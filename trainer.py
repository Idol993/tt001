import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
import numpy as np
import math

from gradient_isolator import GradientIsolator, ParamSliceInfo
from compression import GradientCompressor
from lyapunov import LyapunovAnalyzer
from isolation_report import (
    IsolationReportGenerator,
    CrosstalkMatrixAnalyzer,
    FrequencyMigrationTracker,
    ModuleIsolationReport,
)


class ModularTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        n_layers: int = 2,
        num_classes: int = 10,
        seq_len: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.module_names = ["embedding", "attention", "feedforward", "output"]

        self.embedding = nn.Sequential(
            nn.Embedding(vocab_size, d_model),
            nn.LayerNorm(d_model),
        )

        attention_layers = []
        for i in range(n_layers):
            attention_layers.append((
                f"attn_{i}",
                nn.MultiheadAttention(d_model, n_heads, batch_first=True),
            ))
            attention_layers.append((f"attn_norm_{i}", nn.LayerNorm(d_model)))
        self.attention = nn.Sequential(OrderedDict(attention_layers))

        ff_layers = []
        for i in range(n_layers):
            ff_layers.append((f"ff_{i}_linear1", nn.Linear(d_model, d_ff)))
            ff_layers.append((f"ff_{i}_act", nn.GELU()))
            ff_layers.append((f"ff_{i}_linear2", nn.Linear(d_ff, d_model)))
            ff_layers.append((f"ff_{i}_norm", nn.LayerNorm(d_model)))
        self.feedforward = nn.Sequential(OrderedDict(ff_layers))

        self.output = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            emb = self.embedding[0](x)
            emb = self.embedding[1](emb)
        else:
            emb = x

        attn_out = emb
        for name, layer in self.attention.named_children():
            if "attn_" in name and "norm" not in name:
                residual = attn_out
                attn_out, _ = layer(attn_out, attn_out, attn_out)
                attn_out = attn_out + residual
            else:
                attn_out = layer(attn_out)

        ff_out = attn_out
        for name, layer in self.feedforward.named_children():
            if "linear1" in name:
                residual = ff_out
                ff_out = layer(ff_out)
            elif "linear2" in name:
                ff_out = layer(ff_out)
                ff_out = ff_out + residual
            else:
                ff_out = layer(ff_out)

        pooled = ff_out.transpose(1, 2)
        out = self.output(pooled)
        return out


class CrosstalkFreeTrainer:
    def __init__(
        self,
        model: nn.Module,
        module_names: List[str],
        lr: float = 1e-3,
        n_fft: int = 64,
        hop_length: int = 16,
        num_filters_per_module: int = 6,
        compression_method: str = "topk",
        bandwidth_budget: float = 0.01,
        device: str = "cpu",
        report_interval: int = 10,
        crosstalk_loss_weight: float = 0.1,
        reg_loss_weight: float = 0.01,
    ):
        self.model = model.to(device)
        self.module_names = module_names
        self.device = device
        self.bandwidth_budget = bandwidth_budget
        self.lr = lr
        self._step = 0
        self._report_interval = report_interval
        self._log_interval = report_interval
        self._crosstalk_weight = crosstalk_loss_weight
        self._reg_weight = reg_loss_weight

        self.isolator = GradientIsolator(
            n_fft=n_fft,
            hop_length=hop_length,
            num_modules=len(module_names),
            num_filters_per_module=num_filters_per_module,
            crosstalk_loss_weight=crosstalk_loss_weight,
            reg_weight=reg_loss_weight,
            device=device,
        ).to(device)

        self.compressor = GradientCompressor(
            method=compression_method,
            target_budget_ratio=bandwidth_budget,
        )

        self.lyapunov = LyapunovAnalyzer(
            initial_lr=lr,
            filter_bound_delta=1.0,
            compression_error_bound_eps=0.5,
        )

        self.isolation_reporter = IsolationReportGenerator(
            n_fft=n_fft,
            isolation_threshold=0.5,
        )
        self.crosstalk_analyzer = CrosstalkMatrixAnalyzer(module_names)
        self.freq_tracker = FrequencyMigrationTracker(module_names, n_fft // 2 + 1)

        self.isolator.register_model_structure(model, module_names)

        model_params = list(model.parameters())
        filter_params = list(self.isolator.filter_bank.parameters())
        all_params = model_params + filter_params

        self.optimizer = torch.optim.AdamW(all_params, lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=lr * 0.01
        )

        self._last_param_slices: Dict[str, List[ParamSliceInfo]] = {}
        self._epoch_start_step: int = 0
        self._reports: List[Dict] = []
        self._prev_filter_params: Optional[List[Dict]] = None
        self._prev_filter_params_for_epoch: Optional[List[Dict]] = self.isolator.filter_bank.get_filter_parameters()
        self._filter_stagnant_steps: Dict[str, int] = {name: 0 for name in module_names}
        self._filter_stagnant_threshold = 3
        self._filter_stagnant_min_delta = 1e-4

    def _get_current_lr(self) -> float:
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]
        return self.lr

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: nn.Module,
        generate_report: bool = False,
    ) -> Dict:
        self.model.train()
        self._step += 1
        current_lr = self._get_current_lr()

        inputs_saved = inputs.clone()
        targets_saved = targets.clone()

        self.optimizer.zero_grad()

        with torch.no_grad():
            output_before = self.model(inputs_saved)
            loss_before_step = loss_fn(output_before, targets_saved).item()

        self.lyapunov.record_loss_before_step(loss_before_step)

        output = self.model(inputs)
        loss_main = loss_fn(output, targets)
        loss_main.backward()

        original_grads, param_slices = self.isolator.collect_module_gradients(
            self.model, self.module_names
        )
        self._last_param_slices = param_slices

        module_grad_info = self.isolator.get_last_module_grad_info()

        aux_losses = self.isolator.compute_auxiliary_losses(
            original_grads, self.module_names
        )

        filter_grads_before = {}
        for idx, f in enumerate(self.isolator.filter_bank.filters):
            name = self.module_names[idx] if idx < len(self.module_names) else f"mod_{idx}"
            filter_grads_before[name] = {
                "cf_has_grad": f.log_center_freq.grad is not None,
                "bw_has_grad": f.log_bandwidth.grad is not None,
                "cf_norm_before": f.log_center_freq.grad.norm().item() if f.log_center_freq.grad is not None else 0.0,
                "bw_norm_before": f.log_bandwidth.grad.norm().item() if f.log_bandwidth.grad is not None else 0.0,
            }

        aux_losses["total_aux_loss"].backward()

        filter_param_update_info = {}
        for idx, f in enumerate(self.isolator.filter_bank.filters):
            name = self.module_names[idx] if idx < len(self.module_names) else f"mod_{idx}"
            cf_grad = f.log_center_freq.grad
            bw_grad = f.log_bandwidth.grad
            cf_norm = cf_grad.norm().item() if cf_grad is not None else 0.0
            bw_norm = bw_grad.norm().item() if bw_grad is not None else 0.0
            cf_per_filter = cf_grad.abs().tolist() if cf_grad is not None else [0.0] * f.num_filters
            bw_per_filter = bw_grad.abs().tolist() if bw_grad is not None else [0.0] * f.num_filters
            filter_param_update_info[name] = {
                "center_freq_grad_norm": cf_norm,
                "bandwidth_grad_norm": bw_norm,
                "center_freq_per_filter_grad": cf_per_filter,
                "bandwidth_per_filter_grad": bw_per_filter,
                "cf_grad_zero": cf_norm < 1e-12,
                "bw_grad_zero": bw_norm < 1e-12,
            }

        isolated_grads, iso_stats = self.isolator(
            original_grads,
            self.module_names,
            adapt_filters=(self._step == 1 or self._step % 20 == 0),
        )

        compressed_data, bandwidth = self.compressor.compress_module_gradients(
            isolated_grads
        )
        decompressed_grads = self.compressor.decompress_module_gradients(compressed_data)

        writeback_validation = self.isolator.redistribute_gradients(
            self.model, decompressed_grads, self.module_names, validate=True
        )

        prev_params = self.isolator.filter_bank.get_filter_parameters()
        self.optimizer.step()
        self.scheduler.step()
        after_params = self.isolator.filter_bank.get_filter_parameters()
        filter_deltas = self.isolator.filter_bank.get_filter_delta(self._prev_filter_params)
        self._prev_filter_params = prev_params

        filter_stagnant_info = {}
        for idx, name in enumerate(self.module_names):
            delta = filter_deltas[idx] if idx < len(filter_deltas) else {}
            max_cf = abs(delta.get("max_cf_delta", 0.0))
            max_bw = abs(delta.get("max_bw_delta", 0.0))
            if max_cf < self._filter_stagnant_min_delta and max_bw < self._filter_stagnant_min_delta:
                self._filter_stagnant_steps[name] = self._filter_stagnant_steps.get(name, 0) + 1
            else:
                self._filter_stagnant_steps[name] = 0
            filter_stagnant_info[name] = {
                "stagnant_steps": self._filter_stagnant_steps[name],
                "is_stagnant": self._filter_stagnant_steps[name] >= self._filter_stagnant_threshold,
                "max_cf_delta_abs": max_cf,
                "max_bw_delta_abs": max_bw,
            }

        with torch.no_grad():
            output_after = self.model(inputs_saved)
            loss_after_step = loss_fn(output_after, targets_saved).item()

        lyapunov_record = self.lyapunov.record_step(
            step=self._step,
            loss_after=loss_after_step,
            original_grads=original_grads,
            filtered_grads=isolated_grads,
            compressed_grads=decompressed_grads,
            filter_bank=self.isolator.filter_bank,
            lr=current_lr,
        )

        if iso_stats.get("freq_profiles") is not None:
            self.freq_tracker.record_step(
                self._step,
                iso_stats["freq_profiles"],
                iso_stats.get("peak_info", {}),
            )

        reports = {}
        if generate_report or self._step % self._report_interval == 0:
            reports = self._generate_detailed_reports(
                original_grads,
                isolated_grads,
                decompressed_grads,
                iso_stats,
                writeback_validation,
                filter_param_update_info,
                filter_deltas,
            )

        log = {
            "step": self._step,
            "loss_before": loss_before_step,
            "loss_after": loss_after_step,
            "actual_decrease": lyapunov_record.actual_decrease,
            "expected_decrease": lyapunov_record.expected_decrease,
            "total_loss": (loss_main + aux_losses["total_aux_loss"]).item(),
            "main_loss": loss_main.item(),
            "crosstalk_loss": aux_losses["crosstalk_loss"].item(),
            "reg_loss": aux_losses["reg_loss"].item(),
            "loss_breakdown": aux_losses.get("loss_breakdown", {}),
            "module_grad_info": module_grad_info,
            "filter_update_info": filter_param_update_info,
            "filter_deltas": filter_deltas,
            "filter_stagnant": filter_stagnant_info,
            "bandwidth": {
                "target_ratio": self.bandwidth_budget,
                "actual_ratio": bandwidth.actual_ratio,
                "target_bytes": int(bandwidth.original_bytes * self.bandwidth_budget),
                "actual_bytes": bandwidth.total_bytes,
                "original_bytes": bandwidth.original_bytes,
                "within_budget": bandwidth.within_budget,
                "over_budget_by": bandwidth.details["budget_status"]["over_budget_by"],
                "adjusted_this_step": bandwidth.adjusted_this_step,
                "adjustment_details": bandwidth.adjustment_details,
                "budget_status": bandwidth.details.get("budget_status", {}),
            },
            "lyapunov": {
                "v_before": lyapunov_record.v_before,
                "v_after": lyapunov_record.v_after,
                "contraction_ratio": lyapunov_record.contraction_ratio,
                "sufficient_decrease": lyapunov_record.sufficient_decrease,
                "filter_bounded": lyapunov_record.details["filter_bounded"],
                "filter_perturbation": lyapunov_record.filter_perturbation,
                "compression_perturbation": lyapunov_record.compression_perturbation,
                "total_perturbation": lyapunov_record.total_perturbation,
                "filter_distortion": lyapunov_record.filter_distortion,
                "compression_distortion": lyapunov_record.compression_distortion,
                "grad_norm": math.sqrt(lyapunov_record.grad_norm_sq),
            },
            "writeback_validation": writeback_validation,
            "reports": reports,
        }

        return log

    def _generate_detailed_reports(
        self,
        original_grads: Dict[str, torch.Tensor],
        filtered_grads: Dict[str, torch.Tensor],
        compressed_grads: Dict[str, torch.Tensor],
        iso_stats: Dict,
        writeback_validation: Dict,
        filter_update_info: Dict,
        filter_deltas: List[Dict],
    ) -> Dict:
        reports = {}

        crosstalk_matrix = self.crosstalk_analyzer.compute_crosstalk_matrix(
            original_grads, filtered_grads
        )
        isolation_score = self.crosstalk_analyzer.compute_isolation_score(crosstalk_matrix)

        module_reports = {}
        for idx, name in enumerate(self.module_names):
            if name not in original_grads or name not in filtered_grads:
                continue

            peak_indices = None
            if iso_stats.get("peak_info") and name in iso_stats["peak_info"]:
                peak_indices = iso_stats["peak_info"][name].get("peak_indices")

            crosstalk_leakage = {}
            for j, other_name in enumerate(self.module_names):
                if idx != j:
                    crosstalk_leakage[other_name] = float(crosstalk_matrix[idx, j])

            orig_mag = iso_stats["original_magnitudes"][idx] if idx < len(iso_stats.get("original_magnitudes", [])) else None
            filt_mag = iso_stats["filtered_magnitudes"][idx] if idx < len(iso_stats.get("filtered_magnitudes", [])) else None
            filter_mask = iso_stats["filter_masks"].get(name)

            if orig_mag is not None and filt_mag is not None and filter_mask is not None:
                report = self.isolation_reporter.generate_report(
                    module_name=name,
                    original_grad=original_grads[name],
                    filtered_grad=filtered_grads[name],
                    compressed_grad=compressed_grads.get(name),
                    original_magnitude=orig_mag,
                    filtered_magnitude=filt_mag,
                    filter_mask=filter_mask,
                    peak_indices=peak_indices,
                    crosstalk_leakage=crosstalk_leakage,
                    writeback_validation=writeback_validation,
                )
                module_reports[name] = report

        filter_params = self.isolator.filter_bank.get_filter_parameters()
        filter_diagnostics = {}
        for idx, p in enumerate(filter_params):
            name = self.module_names[idx] if idx < len(self.module_names) else f"module_{idx}"
            delta = filter_deltas[idx] if idx < len(filter_deltas) else {}
            update = filter_update_info.get(name, {})
            filter_diagnostics[name] = {
                "center_frequencies": p["center_frequencies"].tolist(),
                "bandwidths": p["bandwidths"].tolist(),
                "center_freq_delta": delta.get("center_freq_delta", []),
                "bandwidth_delta": delta.get("bandwidth_delta", []),
                "max_cf_delta": delta.get("max_cf_delta", 0.0),
                "max_bw_delta": delta.get("max_bw_delta", 0.0),
                "center_freq_grad_norm": update.get("center_freq_grad_norm", 0.0),
                "bandwidth_grad_norm": update.get("bandwidth_grad_norm", 0.0),
                "cf_per_filter_grad": update.get("center_freq_per_filter_grad", []),
                "bw_per_filter_grad": update.get("bandwidth_per_filter_grad", []),
                "cf_grad_zero": update.get("cf_grad_zero", True),
                "bw_grad_zero": update.get("bw_grad_zero", True),
            }

        reports = {
            "module_reports": module_reports,
            "crosstalk_matrix": crosstalk_matrix,
            "isolation_score": isolation_score,
            "filter_diagnostics": filter_diagnostics,
        }
        self._reports.append(reports)
        return reports

    def train_epoch(
        self,
        dataloader,
        loss_fn: nn.Module,
        epoch: int = 0,
    ) -> List[Dict]:
        logs = []
        self._epoch_start_step = self._step

        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            else:
                inputs, targets = batch.to(self.device), batch.to(self.device)

            generate_report = (batch_idx + 1) % self._log_interval == 0
            log = self.train_step(inputs, targets, loss_fn, generate_report=generate_report)
            logs.append(log)

            if generate_report:
                self._print_step_log(log)

        epoch_summary = self.lyapunov.get_epoch_summary(
            epoch, self._epoch_start_step, self._step + 1
        )
        self._print_epoch_summary(epoch, epoch_summary, logs)

        return logs

    def _print_step_log(self, log: Dict):
        step = log["step"]
        loss = log["loss_before"]
        bw = log["bandwidth"]
        lyap = log["lyapunov"]
        f_info = log["filter_update_info"]
        deltas = log["filter_deltas"]
        grad_info = log["module_grad_info"]
        wb = log["writeback_validation"]
        stagnant = log.get("filter_stagnant", {})

        budget_str = f"TARGET={bw['target_ratio']*100:.2f}% ACTUAL={bw['actual_ratio']*100:.2f}%"
        if bw["within_budget"]:
            budget_str += " [OK]"
        else:
            budget_str += f" [X](+{bw['over_budget_by']*100:.2f}%)"
        if bw.get("adjusted_this_step"):
            budget_str += " [ADJUSTED]"

        contraction_str = f"{lyap['contraction_ratio']:.4f}"
        if lyap["contraction_ratio"] < 1.0:
            contraction_str += " [OK]"
        else:
            contraction_str += " [X]"

        sufficient_str = "[OK]" if lyap["sufficient_decrease"] else "[X]"
        bounded_str = "[OK]" if lyap["filter_bounded"] else "[X]"

        writeback_str = f"{wb['validated_params']}/{wb['total_params']}"
        if wb["validated_params"] == wb["total_params"]:
            writeback_str += " [OK]"
        else:
            writeback_str += " [X]"

        print("=" * 100)
        print(
            f"Step {step:4d} | "
            f"Loss(before/after): {log['loss_before']:.4f}/{log['loss_after']:.4f} "
            f"({log['actual_decrease']:+.4f}) | "
            f"Budget: {budget_str}"
        )
        print(
            f"          "
            f"Contraction: {contraction_str} | "
            f"Sufficient: {sufficient_str} | "
            f"Bounded: {bounded_str} | "
            f"Writeback: {writeback_str}"
        )
        print(
            f"          "
            f"CrosstalkLoss={log['crosstalk_loss']:.6f} | "
            f"RegLoss={log['reg_loss']:.6f} | "
            f"AuxBreakdown(overlap/selfpres/crosssup)={log.get('loss_breakdown', {}).get('overlap_loss', 0):.3f}/"
            f"{log.get('loss_breakdown', {}).get('self_preserve_loss', 0):.3f}/"
            f"{log.get('loss_breakdown', {}).get('cross_suppress_loss', 0):.3f}"
        )

        print("  -- Module Gradient Status --")
        for name in self.module_names:
            gi = grad_info.get(name, {})
            grad_norm = gi.get("norm", 0.0)
            params_w_grad = gi.get("params_with_grad", 0)
            total_p = gi.get("total_params", 0)
            total_elem = gi.get("total_elements", 0)
            all_zero = gi.get("all_zero", False)
            status = "[ZERO!]" if all_zero else ("[OK]" if grad_norm > 0 else "[X]")
            print(f"    {name:<12}: |g|={grad_norm:.4e}, params w/ grad={params_w_grad}/{total_p}, elements={total_elem} {status}")

        print("  -- Filter Learning & Frequency Migration Status --")
        for idx, name in enumerate(self.module_names):
            fi = f_info.get(name, {})
            delta = deltas[idx] if idx < len(deltas) else {}
            cf_norm = fi.get("center_freq_grad_norm", 0.0)
            bw_norm = fi.get("bandwidth_grad_norm", 0.0)
            max_cf_delta = delta.get("max_cf_delta", 0.0)
            max_bw_delta = delta.get("max_bw_delta", 0.0)
            cf_zero = "[CF=0!]" if fi.get("cf_grad_zero", True) else ""
            bw_zero = "[BW=0!]" if fi.get("bw_grad_zero", True) else ""

            stag_info = stagnant.get(name, {})
            stag_steps = stag_info.get("stagnant_steps", 0)
            stag_tag = f"[STAGNANT x{stag_steps}]" if stag_info.get("is_stagnant", False) else ""

            freq_delta = self.freq_tracker.get_last_delta(name)
            if freq_delta["status"] == "no_data":
                freq_str = "freq=[NO DATA]"
            elif freq_delta["status"] == "single_point":
                freq_str = f"dom_freq={freq_delta.get('dominant_freq_last', '?')} (init, no prev)"
            else:
                df = freq_delta["dominant_freq_delta"]
                dc = freq_delta["centroid_delta"]
                df_str = f"{df:+.0f}"
                dc_str = f"{dc:+.2f}"
                freq_str = (f"dom_freq={freq_delta['dominant_freq_prev']}→{freq_delta['dominant_freq_last']}({df_str}), "
                           f"centroid={freq_delta['centroid_prev']:.1f}→{freq_delta['centroid_last']:.1f}({dc_str})")

            print(f"    {name:<12}: |g_cf|={cf_norm:.4e}, |g_bw|={bw_norm:.4e} "
                  f"| Δcf(max)={max_cf_delta:+.4f}, Δbw(max)={max_bw_delta:+.4f} "
                  f"{cf_zero}{bw_zero}{stag_tag}")
            print(f"                  {freq_str}")

        print("  -- Per-Module Gradient Writeback Verification --")
        for name in self.module_names:
            mod_wb = wb.get("per_module", {}).get(name, {})
            total_p = mod_wb.get("total_params", 0)
            valid_p = mod_wb.get("validated_params", 0)
            total_elem = mod_wb.get("total_elements", 0)
            mismatches = mod_wb.get("mismatches", [])
            status = "[OK]" if (total_p > 0 and valid_p == total_p and not mismatches) else "[X]"
            print(f"    {name:<12}: params={valid_p}/{total_p}, elements={total_elem}, mismatches={len(mismatches)} {status}")

        if log.get("reports") and log["reports"].get("module_reports"):
            print("  -- Module Isolation Report --")
            for name, report in log["reports"]["module_reports"].items():
                print(self.isolation_reporter.print_report(report, verbose=False))

            if "crosstalk_matrix" in log["reports"]:
                print(self.crosstalk_analyzer.print_crosstalk_matrix(
                    log["reports"]["crosstalk_matrix"],
                    title=f"Crosstalk Matrix (Isolation Score: {log['reports']['isolation_score']:.4f})"
                ))

        if bw.get("adjusted_this_step") and bw.get("adjustment_details"):
            print("  -- Budget Adjustment Details --")
            adj = bw["adjustment_details"]
            sub = adj.get("sub_adjustments", [adj])
            for s in sub:
                if isinstance(s, dict):
                    print(f"    {s.get('reason', str(s))}")
                    if "previous_ratio" in s and "adjusted_ratio" in s:
                        print(f"      actual_ratio: {s['previous_ratio']*100:.2f}% -> {s['adjusted_ratio']*100:.2f}%")
                    if "old_ratio" in s and "new_ratio" in s:
                        print(f"      topk_ratio: {s['old_ratio']*100:.4f}% -> {s['new_ratio']*100:.4f}%")
                    if "old_topk_ratio" in s and "new_topk_ratio" in s:
                        print(f"      topk_ratio: {s['old_topk_ratio']*100:.4f}% -> {s['new_topk_ratio']*100:.4f}%")
                    if "old_residual_ratio" in s and "new_residual_ratio" in s:
                        print(f"      residual_sample: {s['old_residual_ratio']*100:.2f}% -> {s['new_residual_ratio']*100:.2f}%")
                    if "old_topk_fraction" in s and "new_topk_fraction" in s:
                        print(f"      topk_fraction: {s['old_topk_fraction']*100:.2f}% -> {s['new_topk_fraction']*100:.2f}%")
                    if s.get("at_lower_bound"):
                        print(f"      [NOTE] at lower bound, no further reduction possible")
        print("=" * 100)

    def _print_epoch_summary(self, epoch: int, summary: Dict, logs: List[Dict]):
        if summary.get("status") == "no_data":
            return

        bw_avg = sum(l["bandwidth"]["actual_ratio"] for l in logs) / max(len(logs), 1)
        bw_within = sum(1 for l in logs if l["bandwidth"]["within_budget"]) / max(len(logs), 1)
        bw_adj_count = sum(1 for l in logs if l["bandwidth"].get("adjusted_this_step"))

        print(f"\n{'='*80}")
        print(f"EPOCH {epoch} SUMMARY")
        print(f"{'='*80}")
        print(f"  Steps: {summary['num_steps']}")
        print(f"  Initial Loss: {summary['initial_loss']:.4f} -> Final Loss: {summary['final_loss']:.4f}")
        print(f"  Loss Change: {summary['loss_change']:+.4f}")
        print(f"  Avg Contraction: {summary['avg_contraction']:.4f}")
        print(f"  Sufficient Decrease: {summary['sufficient_decrease_fraction']*100:.1f}% of steps")
        print(f"  Filter Bounded: {summary['filter_bounded_fraction']*100:.1f}% of steps")
        print(f"  Avg Filter Distortion: {summary['avg_filter_distortion']:.4f}")
        print(f"  Avg Compression Distortion: {summary['avg_compression_distortion']:.4f}")
        print(f"  Avg Total Perturbation: {summary['avg_total_perturbation']:.4e}")
        print(f"  Bandwidth: avg={bw_avg*100:.2f}%, within_budget={bw_within*100:.1f}% (target={self.bandwidth_budget*100:.2f}%)")
        if bw_adj_count > 0:
            print(f"  Budget Adjustments this epoch: {bw_adj_count}")

        print(f"\n  Per-Module Gradient Writeback Verification (accumulated over epoch):")
        epoch_wb = {}
        for name in self.module_names:
            epoch_wb[name] = {"total_params": 0, "validated_params": 0, "total_elements": 0, "mismatch_count": 0}
        for log in logs:
            wb = log["writeback_validation"]
            for name in self.module_names:
                mod_wb = wb.get("per_module", {}).get(name, {})
                epoch_wb[name]["total_params"] += mod_wb.get("total_params", 0)
                epoch_wb[name]["validated_params"] += mod_wb.get("validated_params", 0)
                epoch_wb[name]["total_elements"] += mod_wb.get("total_elements", 0)
                epoch_wb[name]["mismatch_count"] += len(mod_wb.get("mismatches", []))
        for name in self.module_names:
            ew = epoch_wb[name]
            status = "[OK]" if (ew["total_params"] > 0 and ew["validated_params"] == ew["total_params"] and ew["mismatch_count"] == 0) else "[X]"
            print(f"    {name:<12}: params={ew['validated_params']}/{ew['total_params']}, "
                  f"elements={ew['total_elements']}, mismatches={ew['mismatch_count']} {status}")

        print(f"\n  Current Filter Parameters (absolute values + this epoch delta):")
        for name, diag in self.get_filter_diagnostics().items():
            centers = [f"{c:.2f}" for c in diag["center_frequencies"]]
            bws = [f"{b:.2f}" for b in diag["bandwidths"]]
            cf_deltas = [f"{d:+.3f}" for d in diag.get("center_freq_delta", [])]
            bw_deltas = [f"{d:+.3f}" for d in diag.get("bandwidth_delta", [])]
            cf_grad = f"{diag.get('center_freq_grad_norm', 0):.4e}"
            bw_grad = f"{diag.get('bandwidth_grad_norm', 0):.4e}"
            moved = any(abs(d) > 1e-4 for d in diag.get("center_freq_delta", []))
            learned = "[LEARNING]" if moved else "[STATIC]"
            print(f"    {name} {learned}:")
            print(f"      Centers:        {centers}")
            print(f"      Center Delta:   {cf_deltas}")
            print(f"      Bandwidths:     {bws}")
            print(f"      Bandwidth Delta:{bw_deltas}")
            print(f"      CF grad norm:   {cf_grad}")
            print(f"      BW grad norm:   {bw_grad}")

        print(f"\n  Frequency Migration (this epoch):")
        for name in self.module_names:
            traj = self.freq_tracker.get_migration_trajectory(name)
            if traj.get("status") == "no_data":
                print(f"    {name:<12}: [NO DATA]")
                continue
            init_d = traj["initial_dominant"]
            final_d = traj["final_dominant"]
            dist = traj["migration_distance"]
            centroids = traj["spectral_centroids"]
            init_c = centroids[0] if centroids else 0
            final_c = centroids[-1] if centroids else 0
            print(f"    {name:<12}: dom_freq {init_d} → {final_d} (Δ={final_d - init_d:+d}, dist={dist})"
                  f" | centroid {init_c:.2f} → {final_c:.2f} (Δ={final_c - init_c:+.2f})")
        print(f"{'='*80}\n")

    def get_filter_diagnostics(self) -> Dict:
        params = self.isolator.filter_bank.get_filter_parameters()
        deltas = self.isolator.filter_bank.get_filter_delta(
            getattr(self, "_prev_filter_params_for_epoch", None)
        )
        diagnostics = {}
        for idx, p in enumerate(params):
            name = self.module_names[idx] if idx < len(self.module_names) else f"module_{idx}"
            f = self.isolator.filter_bank.filters[idx]
            diagnostics[name] = {
                "center_frequencies": p["center_frequencies"].tolist(),
                "bandwidths": p["bandwidths"].tolist(),
                "center_freq_delta": deltas[idx].get("center_freq_delta", []) if idx < len(deltas) else [],
                "bandwidth_delta": deltas[idx].get("bandwidth_delta", []) if idx < len(deltas) else [],
                "center_freq_grad_norm": f.log_center_freq.grad.norm().item() if f.log_center_freq.grad is not None else 0.0,
                "bandwidth_grad_norm": f.log_bandwidth.grad.norm().item() if f.log_bandwidth.grad is not None else 0.0,
            }
        self._prev_filter_params_for_epoch = params
        return diagnostics

    def get_final_lyapunov_proof(self) -> Dict:
        return self.lyapunov.prove_approximate_invariance()

    def get_frequency_migration(self) -> Dict:
        print(self.freq_tracker.print_migration_summary())
        return {
            name: self.freq_tracker.get_migration_trajectory(name)
            for name in self.module_names
        }

    def get_bandwidth_summary(self) -> Dict:
        return self.compressor.get_bandwidth_summary()
