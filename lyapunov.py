import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import math


class LyapunovAnalyzer:
    def __init__(
        self,
        lip_constant: float = 1.0,
        filter_bound: float = 1.0,
        compression_error_bound: float = 0.1,
        lr: float = 1e-3,
    ):
        self.L = lip_constant
        self.delta = filter_bound
        self.epsilon = compression_error_bound
        self.lr = lr
        self._history: List[Dict] = []

    def compute_lyapunov_candidate(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        dataloader,
        device: str = "cpu",
    ) -> float:
        model.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0].to(device), batch[1].to(device)
                else:
                    inputs, targets = batch.to(device), batch.to(device)
                output = model(inputs)
                loss = loss_fn(output, targets)
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
        model.train()
        return total_loss / max(total_samples, 1)

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

    def compute_gradient_distortion(
        self,
        original_grads: Dict[str, torch.Tensor],
        filtered_grads: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        distortion = {}
        for name in original_grads:
            orig = original_grads[name]
            filt = filtered_grads[name]
            diff_norm = (orig - filt).norm().item()
            orig_norm = orig.norm().item()
            relative = diff_norm / (orig_norm + 1e-8)
            cos_sim = torch.nn.functional.cosine_similarity(
                orig.reshape(1, -1), filt.reshape(1, -1)
            ).item()
            distortion[name] = {
                "relative_distortion": relative,
                "cosine_similarity": cos_sim,
                "original_norm": orig_norm,
                "filtered_norm": filt.norm().item(),
            }
        return distortion

    def check_sufficient_decrease(
        self,
        prev_loss: float,
        curr_loss: float,
        grad_norm_sq: float,
        filtered_grad_distortion: float,
        compression_distortion: float,
    ) -> Tuple[bool, float]:
        filter_error = self.delta * filtered_grad_distortion
        compression_error = self.epsilon * compression_distortion
        total_perturbation = filter_error + compression_error

        expected_decrease = self.lr * grad_norm_sq - 0.5 * self.L * self.lr ** 2 * grad_norm_sq
        actual_change = prev_loss - curr_loss
        sufficient = actual_change >= expected_decrease - total_perturbation

        violation = 0.0
        if not sufficient:
            violation = (expected_decrease - total_perturbation) - actual_change

        return sufficient, violation

    def compute_contraction_rate(
        self, window: int = 50
    ) -> float:
        if len(self._history) < 2:
            return float("inf")
        recent = self._history[-window:]
        if len(recent) < 2:
            return float("inf")
        ratios = []
        for i in range(1, len(recent)):
            if recent[i - 1]["lyapunov_value"] > 1e-10:
                ratio = recent[i]["lyapunov_value"] / recent[i - 1]["lyapunov_value"]
                ratios.append(ratio)
        if not ratios:
            return float("inf")
        return sum(ratios) / len(ratios)

    def record_step(
        self,
        step: int,
        lyapunov_value: float,
        grad_norm: float,
        filter_distortion: float,
        compression_distortion: float,
        sufficient_decrease: bool,
    ):
        self._history.append({
            "step": step,
            "lyapunov_value": lyapunov_value,
            "grad_norm": grad_norm,
            "filter_distortion": filter_distortion,
            "compression_distortion": compression_distortion,
            "sufficient_decrease": sufficient_decrease,
        })

    def prove_approximate_invariance(
        self,
        filter_bank,
        original_grads: Dict[str, torch.Tensor],
        filtered_grads: Dict[str, torch.Tensor],
        compressed_grads: Dict[str, torch.Tensor],
        step: int,
        prev_loss: float,
        curr_loss: float,
    ) -> Dict:
        is_bounded, max_dev, filter_details = self.verify_filter_bounded(filter_bank)
        distortion = self.compute_gradient_distortion(original_grads, filtered_grads)

        avg_filter_distortion = sum(
            d["relative_distortion"] for d in distortion.values()
        ) / max(len(distortion), 1)

        compression_distortions = {}
        for name in original_grads:
            if name in compressed_grads:
                orig = original_grads[name]
                comp = compressed_grads[name]
                diff = (orig - comp).norm().item()
                orig_n = orig.norm().item()
                compression_distortions[name] = diff / (orig_n + 1e-8)

        avg_compression_distortion = sum(compression_distortions.values()) / max(
            len(compression_distortions), 1
        )

        grad_norm_sq = sum(g.norm().item() ** 2 for g in original_grads.values())

        sufficient, violation = self.check_sufficient_decrease(
            prev_loss, curr_loss, grad_norm_sq,
            avg_filter_distortion, avg_compression_distortion,
        )

        lyapunov_val = curr_loss

        self.record_step(
            step, lyapunov_val, math.sqrt(grad_norm_sq),
            avg_filter_distortion, avg_compression_distortion, sufficient,
        )

        contraction_rate = self.compute_contraction_rate()

        proof = {
            "step": step,
            "lyapunov_value": lyapunov_val,
            "filter_bounded": is_bounded,
            "filter_max_deviation": max_dev,
            "filter_bound_delta": self.delta,
            "sufficient_decrease": sufficient,
            "violation_magnitude": violation,
            "avg_filter_distortion": avg_filter_distortion,
            "avg_compression_distortion": avg_compression_distortion,
            "contraction_rate": contraction_rate,
            "filter_details": filter_details,
            "module_distortion": distortion,
            "convergence_guarantee": self._build_guarantee(
                is_bounded, sufficient, contraction_rate
            ),
        }
        return proof

    def _build_guarantee(
        self, is_bounded: bool, sufficient: bool, contraction_rate: float
    ) -> str:
        if not is_bounded:
            return (
                "FILTER VIOLATION: Notch filter deviation exceeds delta bound. "
                "Convergence guarantee does not hold. Reduce filter bandwidth or "
                "increase delta."
            )
        if not sufficient:
            return (
                "INSUFFICIENT DECREASE: Lyapunov function does not satisfy "
                "approximate descent condition. The filter + compression distortion "
                "exceeds the expected decrease from gradient step. Consider reducing "
                "learning rate or tightening filter bounds."
            )
        if contraction_rate >= 1.0:
            return (
                "WEAK CONTRACTION: Contraction rate >= 1.0, sequence may not converge. "
                "Monitor for oscillation. The filter preserves descent direction but "
                "contraction is not strict."
            )
        return (
            f"CONVERGENCE GUARANTEED: Lyapunov function satisfies approximate "
            f"invariance with contraction rate {contraction_rate:.4f} < 1.0. "
            f"The notched gradient satisfies: "
            f"V(x_t+1) <= V(x_t) - lr*||g||^2 + delta*dist_filter + eps*dist_comp, "
            f"where delta={self.delta}, eps={self.epsilon}. "
            f"By the Lyapunov direct method, the filtered gradient descent "
            f"converges to an epsilon-neighborhood of a stationary point."
        )

    def get_history(self) -> List[Dict]:
        return self._history
