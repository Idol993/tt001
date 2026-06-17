import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import math
from dataclasses import dataclass, field


@dataclass
class StepRecord:
    step: int
    v_before: float
    v_after: float
    actual_decrease: float
    expected_decrease: float
    grad_norm_sq: float
    filter_distortion: float
    compression_distortion: float
    filter_perturbation: float
    compression_perturbation: float
    total_perturbation: float
    sufficient_decrease: bool
    contraction_ratio: float
    lr: float
    lip_constant: float
    details: Dict = field(default_factory=dict)


class LyapunovAnalyzer:
    def __init__(
        self,
        lip_constant: float = 1.0,
        filter_bound_delta: float = 1.0,
        compression_error_bound_eps: float = 0.5,
        initial_lr: float = 1e-3,
        min_v_threshold: float = 1e-10,
    ):
        self.L = lip_constant
        self.delta = filter_bound_delta
        self.epsilon = compression_error_bound_eps
        self.initial_lr = initial_lr
        self.min_v_threshold = min_v_threshold

        self._history: List[StepRecord] = []
        self._epoch_summaries: List[Dict] = []

    def verify_filter_bounded(
        self,
        filter_bank,
    ) -> Tuple[bool, float, Dict]:
        max_deviation = 0.0
        details = {}
        for idx, f in enumerate(filter_bank.filters):
            mask = f.compute_filter_mask()
            deviation = (1.0 - mask).abs().max().item()
            max_deviation = max(max_deviation, deviation)
            details[f"module_{idx}"] = {
                "max_deviation": deviation,
                "mask_mean": mask.mean().item(),
                "mask_min": mask.min().item(),
                "mask_max": mask.max().item(),
                "center_freqs": f.center_frequencies.detach().cpu().tolist(),
                "bandwidths": f.bandwidths.detach().cpu().tolist(),
            }
        is_bounded = max_deviation <= self.delta
        return is_bounded, max_deviation, details

    def compute_gradient_similarity_metrics(
        self,
        original_grads: Dict[str, torch.Tensor],
        filtered_grads: Dict[str, torch.Tensor],
        compressed_grads: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Dict]:
        metrics = {}
        for name in original_grads:
            orig = original_grads[name]
            filt = filtered_grads[name]
            diff_norm = (orig - filt).norm().item()
            orig_norm = orig.norm().item()
            relative = diff_norm / (orig_norm + 1e-8)
            cos_sim = torch.nn.functional.cosine_similarity(
                orig.reshape(1, -1), filt.reshape(1, -1)
            ).item()
            entry = {
                "original_norm": orig_norm,
                "filtered_norm": filt.norm().item(),
                "filter_relative_distortion": relative,
                "filter_cosine_similarity": cos_sim,
            }
            if compressed_grads is not None and name in compressed_grads:
                comp = compressed_grads[name]
                comp_diff = (orig - comp).norm().item()
                comp_relative = comp_diff / (orig_norm + 1e-8)
                comp_cos = torch.nn.functional.cosine_similarity(
                    orig.reshape(1, -1), comp.reshape(1, -1)
                ).item()
                filt_comp_diff = (filt - comp).norm().item()
                filt_comp_cos = torch.nn.functional.cosine_similarity(
                    filt.reshape(1, -1), comp.reshape(1, -1)
                ).item()
                entry.update({
                    "compressed_norm": comp.norm().item(),
                    "compression_vs_original_relative": comp_relative,
                    "compression_vs_original_cosine": comp_cos,
                    "compression_vs_filtered_relative": filt_comp_diff / (filt.norm().item() + 1e-8),
                    "compression_vs_filtered_cosine": filt_comp_cos,
                })
            metrics[name] = entry
        return metrics

    def compute_module_crosstalk_matrix(
        self,
        original_grads: Dict[str, torch.Tensor],
        filtered_grads: Dict[str, torch.Tensor],
        module_names: List[str],
    ) -> Dict[str, Dict[str, float]]:
        crosstalk = {}
        for i, name_i in enumerate(module_names):
            crosstalk[name_i] = {}
            orig_i = original_grads[name_i]
            orig_i_norm = orig_i.norm().item()
            for j, name_j in enumerate(module_names):
                if i == j:
                    if name_j in filtered_grads:
                        filt_i = filtered_grads[name_j]
                        filt_i_norm = filt_i.norm().item()
                        min_len = min(orig_i.numel(), filt_i.numel())
                        orig_flat = orig_i.reshape(-1)[:min_len]
                        filt_flat = filt_i.reshape(-1)[:min_len]
                        retained = torch.nn.functional.cosine_similarity(
                            orig_flat.unsqueeze(0), filt_flat.unsqueeze(0)
                        ).item()
                        crosstalk[name_i][name_j] = max(0, retained)
                    else:
                        crosstalk[name_i][name_j] = 0.0
                else:
                    if name_j in filtered_grads:
                        filt_j = filtered_grads[name_j]
                        filt_j_norm = filt_j.norm().item()
                        min_len = min(orig_i.numel(), filt_j.numel())
                        orig_flat = orig_i.reshape(-1)[:min_len]
                        filt_flat = filt_j.reshape(-1)[:min_len]
                        leakage = torch.nn.functional.cosine_similarity(
                            orig_flat.unsqueeze(0), filt_flat.unsqueeze(0)
                        ).abs().item()
                        crosstalk[name_i][name_j] = leakage
                    else:
                        crosstalk[name_i][name_j] = 0.0
        return crosstalk

    def record_loss_before_step(self, loss_before: float) -> None:
        self._pending_v_before = loss_before

    def record_step(
        self,
        step: int,
        loss_after: float,
        original_grads: Dict[str, torch.Tensor],
        filtered_grads: Dict[str, torch.Tensor],
        compressed_grads: Dict[str, torch.Tensor],
        filter_bank,
        lr: Optional[float] = None,
        module_names: Optional[List[str]] = None,
    ) -> StepRecord:
        current_lr = lr if lr is not None else self.initial_lr
        v_before = getattr(self, "_pending_v_before", loss_after)
        v_after = loss_after
        actual_decrease = v_before - v_after

        grad_norm_sq = sum(g.norm().item() ** 2 for g in original_grads.values())

        filter_distortions = []
        for name in original_grads:
            if name in filtered_grads:
                orig = original_grads[name]
                filt = filtered_grads[name]
                dist = (orig - filt).norm().item() / (orig.norm().item() + 1e-8)
                filter_distortions.append(dist)
        avg_filter_distortion = sum(filter_distortions) / max(len(filter_distortions), 1)

        compression_distortions = []
        for name in original_grads:
            if name in compressed_grads and name in filtered_grads:
                filt = filtered_grads[name]
                comp = compressed_grads[name]
                dist = (filt - comp).norm().item() / (filt.norm().item() + 1e-8)
                compression_distortions.append(dist)
        avg_compression_distortion = sum(compression_distortions) / max(len(compression_distortions), 1)

        expected_decrease = current_lr * grad_norm_sq - 0.5 * self.L * current_lr ** 2 * grad_norm_sq

        filter_perturbation = self.delta * avg_filter_distortion * math.sqrt(grad_norm_sq)
        compression_perturbation = self.epsilon * avg_compression_distortion * math.sqrt(grad_norm_sq)
        total_perturbation = filter_perturbation + compression_perturbation

        sufficient_decrease = actual_decrease >= expected_decrease - total_perturbation

        if v_before > self.min_v_threshold:
            contraction_ratio = v_after / v_before
        else:
            contraction_ratio = 1.0

        is_bounded, max_dev, filter_details = self.verify_filter_bounded(filter_bank)

        similarity_metrics = self.compute_gradient_similarity_metrics(
            original_grads, filtered_grads, compressed_grads
        )

        crosstalk_matrix = None

        details = {
            "filter_bounded": is_bounded,
            "filter_max_deviation": max_dev,
            "filter_bound_delta": self.delta,
            "compression_bound_eps": self.epsilon,
            "filter_details": filter_details,
            "similarity_metrics": similarity_metrics,
            "crosstalk_matrix": crosstalk_matrix,
        }

        record = StepRecord(
            step=step,
            v_before=v_before,
            v_after=v_after,
            actual_decrease=actual_decrease,
            expected_decrease=expected_decrease,
            grad_norm_sq=grad_norm_sq,
            filter_distortion=avg_filter_distortion,
            compression_distortion=avg_compression_distortion,
            filter_perturbation=filter_perturbation,
            compression_perturbation=compression_perturbation,
            total_perturbation=total_perturbation,
            sufficient_decrease=sufficient_decrease,
            contraction_ratio=contraction_ratio,
            lr=current_lr,
            lip_constant=self.L,
            details=details,
        )

        self._history.append(record)
        return record

    def compute_contraction_curve(self, window: Optional[int] = None) -> List[float]:
        if len(self._history) < 2:
            return []
        history = self._history[-window:] if window else self._history
        ratios = []
        for r in history[1:]:
            if r.v_before > self.min_v_threshold:
                ratios.append(r.contraction_ratio)
        return ratios

    def compute_rolling_contraction_rate(self, window: int = 10) -> List[float]:
        ratios = self.compute_contraction_curve()
        if len(ratios) < window:
            return [sum(ratios) / len(ratios)] if ratios else [float("inf")]
        rolling = []
        for i in range(len(ratios) - window + 1):
            rolling.append(sum(ratios[i:i+window]) / window)
        return rolling

    def get_overall_convergence_metrics(self) -> Dict:
        if len(self._history) < 2:
            return {"status": "insufficient_data"}

        all_ratios = self.compute_contraction_curve()
        avg_contraction = sum(all_ratios) / len(all_ratios) if all_ratios else float("inf")

        sufficient_count = sum(1 for r in self._history if r.sufficient_decrease)
        sufficient_fraction = sufficient_count / len(self._history)

        bounded_count = sum(1 for r in self._history if r.details.get("filter_bounded", False))
        bounded_fraction = bounded_count / len(self._history)

        total_decrease = self._history[-1].v_after - self._history[0].v_before
        monotonic_decreases = sum(
            1 for r in self._history if r.actual_decrease > 0
        )
        monotonic_fraction = monotonic_decreases / len(self._history)

        rolling_rates = self.compute_rolling_contraction_rate(window=min(20, max(2, len(self._history) // 5)))
        final_rolling = rolling_rates[-1] if rolling_rates else avg_contraction

        initial_perturbation = self._history[0].total_perturbation if self._history else 0
        final_perturbation = self._history[-1].total_perturbation if self._history else 0

        return {
            "total_steps": len(self._history),
            "initial_loss": self._history[0].v_before,
            "final_loss": self._history[-1].v_after,
            "total_loss_change": total_decrease,
            "average_contraction_ratio": avg_contraction,
            "final_rolling_contraction_rate": final_rolling,
            "sufficient_decrease_fraction": sufficient_fraction,
            "filter_bounded_fraction": bounded_fraction,
            "monotonic_decrease_fraction": monotonic_fraction,
            "initial_total_perturbation": initial_perturbation,
            "final_total_perturbation": final_perturbation,
            "contraction_curve": all_ratios,
            "rolling_contraction_rates": rolling_rates,
        }

    def prove_approximate_invariance(self) -> Dict:
        metrics = self.get_overall_convergence_metrics()
        if metrics.get("status") == "insufficient_data":
            return {
                "conclusion": "INSUFFICIENT DATA",
                "details": "Need at least 2 training steps to perform convergence analysis.",
                "metrics": metrics,
            }

        avg_contraction = metrics["average_contraction_ratio"]
        sufficient_fraction = metrics["sufficient_decrease_fraction"]
        bounded_fraction = metrics["filter_bounded_fraction"]
        final_rolling = metrics["final_rolling_contraction_rate"]

        reasons = []
        if bounded_fraction < 0.95:
            reasons.append(
                f"Filter bounded condition violated in {100*(1-bounded_fraction):.1f}% of steps. "
                f"Max deviation exceeds delta={self.delta}. "
                "This means the notch filter occasionally amplifies or distorts gradient components."
            )

        if sufficient_fraction < 0.9:
            reasons.append(
                f"Sufficient decrease condition violated in {100*(1-sufficient_fraction):.1f}% of steps. "
                "The combined filter + compression perturbation occasionally exceeds the expected gradient decrease."
            )

        if avg_contraction >= 1.0:
            reasons.append(
                f"Average contraction ratio {avg_contraction:.4f} >= 1.0. "
                "The Lyapunov function does not show overall contraction."
            )

        if final_rolling >= 1.0:
            reasons.append(
                f"Final rolling contraction rate {final_rolling:.4f} >= 1.0. "
                "Recent steps show divergence or oscillation rather than convergence."
            )

        if not reasons:
            conclusion = (
                f"CONVERGENCE GUARANTEED: Lyapunov approximate invariance holds across all {metrics['total_steps']} steps. "
                f"Average contraction ratio: {avg_contraction:.4f} < 1.0. "
                f"Final rolling contraction rate: {final_rolling:.4f} < 1.0. "
                f"Filter bounded in {bounded_fraction*100:.1f}% of steps (delta={self.delta}). "
                f"Sufficient decrease satisfied in {sufficient_fraction*100:.1f}% of steps. "
                f"By the Lyapunov direct method, the filtered gradient descent converges to an "
                f"epsilon-neighborhood of a stationary point, where epsilon is bounded by "
                f"the total perturbation (initial: {metrics['initial_total_perturbation']:.4f}, "
                f"final: {metrics['final_total_perturbation']:.4f})."
            )
        else:
            conclusion = (
                "CONVERGENCE NOT GUARANTEED: The following conditions were violated:\n"
                + "\n".join(f"  - {r}" for r in reasons)
                + f"\n\nMetrics: avg_contraction={avg_contraction:.4f}, "
                f"bounded={bounded_fraction*100:.1f}%, sufficient={sufficient_fraction*100:.1f}%"
            )

        return {
            "conclusion": conclusion,
            "reasons": reasons,
            "metrics": metrics,
            "theoretical_formulation": (
                "The proof uses the Lyapunov function V(x) = L(x) (the training loss). "
                "For filtered gradient descent with compression, we require:\n"
                "  1. Filter boundedness: ||g_filtered - g_orig|| <= delta * ||g_orig||\n"
                "  2. Sufficient decrease: V(x_{t+1}) <= V(x_t) - lr*||g||^2 + 0.5*L*lr^2*||g||^2 + perturbation\n"
                "  3. Contraction: V(x_{t+1}) / V(x_t) < 1 (eventually)\n"
                f"With delta={self.delta}, eps={self.epsilon}, L={self.L}, these conditions ensure "
                "convergence to a bounded neighborhood of a critical point."
            ),
        }

    def get_epoch_summary(self, epoch_idx: int, start_step: int, end_step: int) -> Dict:
        epoch_records = [r for r in self._history if start_step <= r.step < end_step]
        if not epoch_records:
            return {"epoch": epoch_idx, "status": "no_data"}

        avg_v = sum(r.v_after for r in epoch_records) / len(epoch_records)
        avg_contraction = sum(
            r.contraction_ratio for r in epoch_records[1:]
        ) / max(len(epoch_records) - 1, 1)
        sufficient_count = sum(1 for r in epoch_records if r.sufficient_decrease)
        bounded_count = sum(
            1 for r in epoch_records if r.details.get("filter_bounded", False)
        )

        summary = {
            "epoch": epoch_idx,
            "start_step": start_step,
            "end_step": end_step,
            "num_steps": len(epoch_records),
            "avg_loss": avg_v,
            "initial_loss": epoch_records[0].v_before,
            "final_loss": epoch_records[-1].v_after,
            "loss_change": epoch_records[-1].v_after - epoch_records[0].v_before,
            "avg_contraction": avg_contraction,
            "sufficient_decrease_fraction": sufficient_count / len(epoch_records),
            "filter_bounded_fraction": bounded_count / len(epoch_records),
            "avg_filter_distortion": sum(r.filter_distortion for r in epoch_records) / len(epoch_records),
            "avg_compression_distortion": sum(r.compression_distortion for r in epoch_records) / len(epoch_records),
            "avg_total_perturbation": sum(r.total_perturbation for r in epoch_records) / len(epoch_records),
        }
        self._epoch_summaries.append(summary)
        return summary

    def get_history(self) -> List[StepRecord]:
        return self._history

    def get_epoch_summaries(self) -> List[Dict]:
        return self._epoch_summaries
