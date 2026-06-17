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
        crosstalk_loss_weight: float = 0.01,
        reg_loss_weight: float = 0.001,
    ):
        self.model = model.to(device)
        self.module_names = module_names
        self.device = device
        self.bandwidth_budget = bandwidth_budget
        self.lr = lr
        self._step = 0
        self._log_interval = 10
        self._report_interval = report_interval
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

        self.optimizer.zero_grad()
        output = self.model(inputs)
        loss_before = loss_fn(output, targets)

        self.lyapunov.record_loss_before_step(loss_before.item())

        loss_before.backward()

        with torch.no_grad():
            original_grads, param_slices = self.isolator.collect_module_gradients(
                self.model, self.module_names
            )
            self._last_param_slices = param_slices

        aux_losses = self.isolator.compute_auxiliary_losses(
            original_grads, self.module_names
        )

        aux_losses["total_aux_loss"].backward()

        isolated_grads, iso_stats = self.isolator(
            original_grads,
            self.module_names,
            adapt_filters=(self._step == 1 or self._step % 50 == 0),
        )

        compressed_data, bandwidth = self.compressor.compress_module_gradients(
            isolated_grads
        )
        decompressed_grads = self.compressor.decompress_module_gradients(compressed_data)

        writeback_validation = self.isolator.redistribute_gradients(
            self.model, decompressed_grads, self.module_names, validate=True
        )

        total_loss = loss_before + aux_losses["total_aux_loss"]

        self.optimizer.step()
        self.scheduler.step()

        with torch.no_grad():
            output_after = self.model(inputs)
            loss_after = loss_fn(output_after, targets).item()

        lyapunov_record = self.lyapunov.record_step(
            step=self._step,
            loss_after=loss_after,
            original_grads=original_grads,
            filtered_grads=isolated_grads,
            compressed_grads=decompressed_grads,
            filter_bank=self.isolator.filter_bank,
            lr=current_lr,
        )

        if iso_stats.get("freq_profiles") and iso_stats.get("peak_info"):
            self.freq_tracker.record_step(
                self._step,
                iso_stats["freq_profiles"],
                iso_stats["peak_info"],
            )

        reports = {}
        if generate_report or self._step % self._report_interval == 0:
            reports = self._generate_detailed_reports(
                original_grads,
                isolated_grads,
                decompressed_grads,
                iso_stats,
                writeback_validation,
            )

        log = {
            "step": self._step,
            "loss_before": loss_before.item(),
            "loss_after": loss_after,
            "actual_decrease": lyapunov_record.actual_decrease,
            "expected_decrease": lyapunov_record.expected_decrease,
            "total_loss": total_loss.item(),
            "crosstalk_loss": aux_losses["crosstalk_loss"].item(),
            "reg_loss": aux_losses["reg_loss"].item(),
            "bandwidth": {
                "target_ratio": self.bandwidth_budget,
                "actual_ratio": bandwidth.actual_ratio,
                "target_bytes": int(bandwidth.original_bytes * self.bandwidth_budget),
                "actual_bytes": bandwidth.total_bytes,
                "original_bytes": bandwidth.original_bytes,
                "within_budget": bandwidth.within_budget,
                "over_budget_by": bandwidth.details["budget_status"]["over_budget_by"],
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
            filter_diagnostics[name] = {
                "center_frequencies": p["center_frequencies"].tolist(),
                "bandwidths": p["bandwidths"].tolist(),
                "center_freq_grads": [],
                "bandwidth_grads": [],
            }
            for f_idx, f in enumerate(self.isolator.filter_bank.filters):
                if f_idx == idx:
                    if f.log_center_freq.grad is not None:
                        filter_diagnostics[name]["center_freq_grads"] = f.log_center_freq.grad.norm(dim=-1).tolist()
                    if f.log_bandwidth.grad is not None:
                        filter_diagnostics[name]["bandwidth_grads"] = f.log_bandwidth.grad.norm(dim=-1).tolist()

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

        budget_str = f"TARGET={bw['target_ratio']*100:.2f}% ACTUAL={bw['actual_ratio']*100:.2f}%"
        if bw["within_budget"]:
            budget_str += " [OK]"
        else:
            budget_str += f" [X](+{bw['over_budget_by']*100:.2f}%)"

        contraction_str = f"{lyap['contraction_ratio']:.4f}"
        if lyap["contraction_ratio"] < 1.0:
            contraction_str += " [OK]"
        else:
            contraction_str += " [X]"

        sufficient_str = "[OK]" if lyap["sufficient_decrease"] else "[X]"
        bounded_str = "[OK]" if lyap["filter_bounded"] else "[X]"

        writeback = log["writeback_validation"]
        writeback_str = f"{writeback['validated_params']}/{writeback['total_params']}"
        if writeback["validated_params"] == writeback["total_params"]:
            writeback_str += " [OK]"
        else:
            writeback_str += " [X]"

        print(
            f"Step {step:4d} | "
            f"Loss: {loss:.4f} | "
            f"Budget: {budget_str} | "
            f"Contraction: {contraction_str} | "
            f"Sufficient: {sufficient_str} | "
            f"Bounded: {bounded_str} | "
            f"Writeback: {writeback_str}"
        )

        if log.get("reports") and log["reports"].get("module_reports"):
            for name, report in log["reports"]["module_reports"].items():
                print(self.isolation_reporter.print_report(report, verbose=False))

            if "crosstalk_matrix" in log["reports"]:
                print(self.crosstalk_analyzer.print_crosstalk_matrix(
                    log["reports"]["crosstalk_matrix"],
                    title=f"Crosstalk Matrix (Isolation Score: {log['reports']['isolation_score']:.4f})"
                ))

    def _print_epoch_summary(self, epoch: int, summary: Dict, logs: List[Dict]):
        if summary.get("status") == "no_data":
            return

        bw_avg = sum(l["bandwidth"]["actual_ratio"] for l in logs) / max(len(logs), 1)
        bw_within = sum(1 for l in logs if l["bandwidth"]["within_budget"]) / max(len(logs), 1)

        print(f"\n{'='*80}")
        print(f"EPOCH {epoch} SUMMARY")
        print(f"{'='*80}")
        print(f"  Steps: {summary['num_steps']}")
        print(f"  Initial Loss: {summary['initial_loss']:.4f} → Final Loss: {summary['final_loss']:.4f}")
        print(f"  Loss Change: {summary['loss_change']:+.4f}")
        print(f"  Avg Contraction: {summary['avg_contraction']:.4f}")
        print(f"  Sufficient Decrease: {summary['sufficient_decrease_fraction']*100:.1f}% of steps")
        print(f"  Filter Bounded: {summary['filter_bounded_fraction']*100:.1f}% of steps")
        print(f"  Avg Filter Distortion: {summary['avg_filter_distortion']:.4f}")
        print(f"  Avg Compression Distortion: {summary['avg_compression_distortion']:.4f}")
        print(f"  Avg Total Perturbation: {summary['avg_total_perturbation']:.4e}")
        print(f"  Bandwidth: avg={bw_avg*100:.2f}%, within_budget={bw_within*100:.1f}% (target={self.bandwidth_budget*100:.2f}%)")

        filter_diag = self.get_filter_diagnostics()
        print(f"\n  Current Filter Parameters:")
        for name, diag in filter_diag.items():
            centers = [f"{c:.1f}" for c in diag["center_frequencies"]]
            bws = [f"{b:.2f}" for b in diag["bandwidths"]]
            cf_grads = [f"{g:.4f}" for g in diag["center_freq_grads"]] if diag["center_freq_grads"] else ["n/a"]
            bw_grads = [f"{g:.4f}" for g in diag["bandwidth_grads"]] if diag["bandwidth_grads"] else ["n/a"]
            print(f"    {name}:")
            print(f"      Centers:   {centers}")
            print(f"      Bandwidths:{bws}")
            print(f"      CF grads:  {cf_grads}")
            print(f"      BW grads:  {bw_grads}")
        print(f"{'='*80}\n")

    def get_filter_diagnostics(self) -> Dict:
        params = self.isolator.filter_bank.get_filter_parameters()
        diagnostics = {}
        for idx, p in enumerate(params):
            name = self.module_names[idx] if idx < len(self.module_names) else f"module_{idx}"
            diagnostics[name] = {
                "center_frequencies": p["center_frequencies"].tolist(),
                "bandwidths": p["bandwidths"].tolist(),
                "center_freq_grads": [],
                "bandwidth_grads": [],
            }
            for f_idx, f in enumerate(self.isolator.filter_bank.filters):
                if f_idx == idx:
                    if f.log_center_freq.grad is not None:
                        diagnostics[name]["center_freq_grads"] = f.log_center_freq.grad.abs().tolist()
                    if f.log_bandwidth.grad is not None:
                        diagnostics[name]["bandwidth_grads"] = f.log_bandwidth.grad.abs().tolist()
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
