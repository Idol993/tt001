import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import math
import numpy as np


@dataclass
class FrequencyBandAnalysis:
    band_name: str
    freq_range: Tuple[int, int]
    original_energy: float
    filtered_energy: float
    retained_ratio: float
    isolated: bool


@dataclass
class ModuleIsolationReport:
    module_name: str
    original_grad_norm: float
    filtered_grad_norm: float
    compressed_grad_norm: float
    filter_cosine_similarity: float
    compression_cosine_similarity: float
    filter_relative_distortion: float
    compression_relative_distortion: float
    peak_frequencies: List[int]
    filter_retained_bands: List[FrequencyBandAnalysis]
    filter_isolated_bands: List[FrequencyBandAnalysis]
    crosstalk_leakage: Dict[str, float]
    param_slices_verified: int
    param_slices_total: int


class IsolationReportGenerator:
    def __init__(
        self,
        n_fft: int = 64,
        freq_bands: Optional[List[Tuple[str, Tuple[int, int]]]] = None,
        isolation_threshold: float = 0.5,
    ):
        self.n_fft = n_fft
        self.n_freq = n_fft // 2 + 1
        self.isolation_threshold = isolation_threshold

        if freq_bands is None:
            self.freq_bands = [
                ("low_freq", (0, self.n_freq // 4)),
                ("mid_freq", (self.n_freq // 4, self.n_freq // 2)),
                ("high_freq", (self.n_freq // 2, self.n_freq)),
            ]
        else:
            self.freq_bands = freq_bands

    def analyze_frequency_bands(
        self,
        original_magnitude: torch.Tensor,
        filtered_magnitude: torch.Tensor,
        filter_mask: torch.Tensor,
    ) -> Tuple[List[FrequencyBandAnalysis], List[FrequencyBandAnalysis]]:
        original_profile = original_magnitude.mean(dim=-1).detach().cpu().numpy()
        filtered_profile = filtered_magnitude.mean(dim=-1).detach().cpu().numpy()
        mask_np = filter_mask.detach().cpu().numpy()

        retained = []
        isolated = []

        for band_name, (start, end) in self.freq_bands:
            orig_energy = float(np.sum(original_profile[start:end] ** 2))
            filt_energy = float(np.sum(filtered_profile[start:end] ** 2))
            retained_ratio = filt_energy / (orig_energy + 1e-10)
            band_mask = np.mean(mask_np[start:end])
            analysis = FrequencyBandAnalysis(
                band_name=band_name,
                freq_range=(start, end),
                original_energy=orig_energy,
                filtered_energy=filt_energy,
                retained_ratio=retained_ratio,
                isolated=band_mask < self.isolation_threshold,
            )
            if analysis.isolated:
                isolated.append(analysis)
            else:
                retained.append(analysis)

        return retained, isolated

    def generate_report(
        self,
        module_name: str,
        original_grad: torch.Tensor,
        filtered_grad: torch.Tensor,
        compressed_grad: Optional[torch.Tensor],
        original_magnitude: torch.Tensor,
        filtered_magnitude: torch.Tensor,
        filter_mask: torch.Tensor,
        peak_indices: Optional[List[int]] = None,
        crosstalk_leakage: Optional[Dict[str, float]] = None,
        writeback_validation: Optional[Dict] = None,
    ) -> ModuleIsolationReport:
        orig_norm = original_grad.norm().item()
        filt_norm = filtered_grad.norm().item()

        filter_cos = torch.nn.functional.cosine_similarity(
            original_grad.reshape(1, -1),
            filtered_grad.reshape(1, -1),
        ).item()
        filter_dist = (original_grad - filtered_grad).norm().item() / (orig_norm + 1e-8)

        comp_norm = 0.0
        comp_cos = 0.0
        comp_dist = 0.0
        if compressed_grad is not None:
            comp_norm = compressed_grad.norm().item()
            comp_cos = torch.nn.functional.cosine_similarity(
                filtered_grad.reshape(1, -1),
                compressed_grad.reshape(1, -1),
            ).item()
            comp_dist = (filtered_grad - compressed_grad).norm().item() / (filt_norm + 1e-8)

        retained_bands, isolated_bands = self.analyze_frequency_bands(
            original_magnitude, filtered_magnitude, filter_mask
        )

        peaks = peak_indices if peak_indices is not None else []

        verified = writeback_validation.get("validated_params", 0) if writeback_validation else 0
        total = writeback_validation.get("total_params", 0) if writeback_validation else 0

        return ModuleIsolationReport(
            module_name=module_name,
            original_grad_norm=orig_norm,
            filtered_grad_norm=filt_norm,
            compressed_grad_norm=comp_norm,
            filter_cosine_similarity=filter_cos,
            compression_cosine_similarity=comp_cos,
            filter_relative_distortion=filter_dist,
            compression_relative_distortion=comp_dist,
            peak_frequencies=peaks,
            filter_retained_bands=retained_bands,
            filter_isolated_bands=isolated_bands,
            crosstalk_leakage=crosstalk_leakage or {},
            param_slices_verified=verified,
            param_slices_total=total,
        )

    def print_report(
        self,
        report: ModuleIsolationReport,
        verbose: bool = False,
    ) -> str:
        lines = [f"\n{'='*60}"]
        lines.append(f"Module: {report.module_name}")
        lines.append(f"{'='*60}")
        lines.append(f"  Gradient Norms:")
        lines.append(f"    Original:    {report.original_grad_norm:.4e}")
        lines.append(f"    Filtered:    {report.filtered_grad_norm:.4e}")
        lines.append(f"    Compressed:  {report.compressed_grad_norm:.4e}")
        lines.append(f"  Similarity Metrics:")
        lines.append(f"    Filter vs Original Cosine:    {report.filter_cosine_similarity:.4f}")
        lines.append(f"    Filter Distortion:             {report.filter_relative_distortion:.4f}")
        lines.append(f"    Compress vs Filtered Cosine:   {report.compression_cosine_similarity:.4f}")
        lines.append(f"    Compression Distortion:        {report.compression_relative_distortion:.4f}")

        if report.peak_frequencies:
            peaks_str = ", ".join(str(p) for p in report.peak_frequencies[:5])
            lines.append(f"  Dominant Frequency Peaks: {peaks_str}")

        lines.append(f"\n  Frequency Band Analysis:")
        lines.append(f"    RETAINED bands (mask >= {self.isolation_threshold}):")
        for band in report.filter_retained_bands:
            lines.append(f"      {band.band_name:12s} [{band.freq_range[0]:2d}-{band.freq_range[1]:2d}]: "
                        f"{band.retained_ratio*100:5.1f}% energy retained")
        if report.filter_isolated_bands:
            lines.append(f"    ISOLATED bands (mask < {self.isolation_threshold}):")
            for band in report.filter_isolated_bands:
                lines.append(f"      {band.band_name:12s} [{band.freq_range[0]:2d}-{band.freq_range[1]:2d}]: "
                            f"{band.retained_ratio*100:5.1f}% energy retained (NOTCHED)")

        if report.crosstalk_leakage:
            lines.append(f"\n  Crosstalk Leakage to other modules:")
            for target, leakage in sorted(report.crosstalk_leakage.items()):
                if leakage > 0.01:
                    lines.append(f"    -> {target}: {leakage:.4f}")

        if verbose:
            lines.append(f"\n  Gradient Writeback Verification:")
            lines.append(f"    Verified params: {report.param_slices_verified}/{report.param_slices_total}")
            if report.param_slices_verified == report.param_slices_total and report.param_slices_total > 0:
                lines.append(f"    Status: [OK] ALL slices matched correctly")
            else:
                lines.append(f"    Status: [X] Some slices mismatched")

        return "\n".join(lines)


class CrosstalkMatrixAnalyzer:
    def __init__(self, module_names: List[str]):
        self.module_names = module_names

    def compute_crosstalk_matrix(
        self,
        original_grads: Dict[str, torch.Tensor],
        filtered_grads: Dict[str, torch.Tensor],
    ) -> np.ndarray:
        n = len(self.module_names)
        matrix = np.zeros((n, n))
        for i, name_i in enumerate(self.module_names):
            if name_i not in original_grads:
                continue
            orig_i = original_grads[name_i]
            for j, name_j in enumerate(self.module_names):
                if name_j not in filtered_grads:
                    continue
                filt_j = filtered_grads[name_j]
                min_len = min(orig_i.numel(), filt_j.numel())
                orig_flat = orig_i.reshape(-1)[:min_len]
                filt_flat = filt_j.reshape(-1)[:min_len]
                if i == j:
                    similarity = torch.nn.functional.cosine_similarity(
                        orig_flat.unsqueeze(0),
                        filt_flat.unsqueeze(0),
                    ).item()
                    matrix[i, j] = max(0.0, similarity)
                else:
                    leakage = torch.nn.functional.cosine_similarity(
                        orig_flat.unsqueeze(0),
                        filt_flat.unsqueeze(0),
                    ).abs().item()
                    matrix[i, j] = leakage
        return matrix

    def print_crosstalk_matrix(
        self,
        matrix: np.ndarray,
        title: str = "Crosstalk Matrix",
    ) -> str:
        n = len(self.module_names)
        col_width = max(12, max(len(name) for name in self.module_names) + 2)

        lines = [f"\n{'='*60}"]
        lines.append(f"{title}")
        lines.append(f"{'='*60}")
        header = f"{'':{col_width}}"
        for name in self.module_names:
            header += f"{name:>{col_width}}"
        lines.append(header)
        lines.append("-" * (col_width * (n + 1)))

        for i, name_i in enumerate(self.module_names):
            row = f"{name_i:{col_width}}"
            for j in range(n):
                val = matrix[i, j]
                if i == j:
                    row += f"\033[92m{val:>{col_width}.4f}\033[0m"
                elif val > 0.1:
                    row += f"\033[91m{val:>{col_width}.4f}\033[0m"
                else:
                    row += f"{val:>{col_width}.4f}"
            lines.append(row)

        lines.append(f"\n  Diagonal = signal retained (should be high)")
        lines.append(f"  Off-diagonal = crosstalk leakage (should be low)")
        return "\n".join(lines)

    def compute_isolation_score(self, matrix: np.ndarray) -> float:
        n = len(self.module_names)
        signal = 0.0
        leakage = 0.0
        for i in range(n):
            signal += matrix[i, i]
            for j in range(n):
                if i != j:
                    leakage += matrix[i, j]
        total = signal + leakage
        return signal / max(total, 1e-8)


class FrequencyMigrationTracker:
    def __init__(self, module_names: List[str], n_freq: int):
        self.module_names = module_names
        self.n_freq = n_freq
        self._history: Dict[str, List[Dict]] = {name: [] for name in module_names}

    def record_step(
        self,
        step: int,
        freq_profiles: Dict[str, torch.Tensor],
        peak_info: Dict[str, Dict],
    ):
        for name in self.module_names:
            if name in freq_profiles:
                profile = freq_profiles[name].detach().cpu().numpy()
                entry = {
                    "step": step,
                    "freq_profile": profile,
                    "dominant_freq": int(np.argmax(profile)),
                    "spectral_centroid": float(np.sum(np.arange(len(profile)) * profile) / (np.sum(profile) + 1e-8)),
                }
                if name in peak_info and "peak_indices" in peak_info[name]:
                    entry["peak_indices"] = peak_info[name]["peak_indices"]
                self._history[name].append(entry)

    def get_migration_trajectory(self, module_name: str) -> Dict:
        history = self._history.get(module_name, [])
        if not history:
            return {"status": "no_data"}
        steps = [h["step"] for h in history]
        dominant_freqs = [h["dominant_freq"] for h in history]
        centroids = [h["spectral_centroid"] for h in history]
        return {
            "steps": steps,
            "dominant_frequencies": dominant_freqs,
            "spectral_centroids": centroids,
            "initial_dominant": dominant_freqs[0] if dominant_freqs else -1,
            "final_dominant": dominant_freqs[-1] if dominant_freqs else -1,
            "migration_distance": abs(dominant_freqs[-1] - dominant_freqs[0]) if len(dominant_freqs) >= 2 else 0,
        }

    def print_migration_summary(self) -> str:
        lines = [f"\n{'='*60}"]
        lines.append(f"Frequency Migration Summary")
        lines.append(f"{'='*60}")
        for name in self.module_names:
            traj = self.get_migration_trajectory(name)
            if traj.get("status") == "no_data":
                lines.append(f"  {name:<16s}: No data")
                continue
            lines.append(f"  {name:<16s}:")
            lines.append(f"    Initial dominant freq: {traj['initial_dominant']}")
            lines.append(f"    Final dominant freq:   {traj['final_dominant']}")
            lines.append(f"    Migration distance:    {traj['migration_distance']} bins")
            lines.append(f"    Final centroid:        {traj['spectral_centroids'][-1]:.2f}")
        return "\n".join(lines)

    def get_last_delta(self, module_name: str) -> Dict:
        history = self._history.get(module_name, [])
        if not history:
            return {"status": "no_data"}
        if len(history) < 2:
            last = history[-1]
            return {
                "status": "single_point",
                "dominant_freq_last": last["dominant_freq"],
                "centroid_last": last["spectral_centroid"],
                "step_last": last["step"],
            }
        last = history[-1]
        prev = history[-2]
        df = last["dominant_freq"] - prev["dominant_freq"]
        dc = last["spectral_centroid"] - prev["spectral_centroid"]
        return {
            "status": "ok",
            "dominant_freq_prev": prev["dominant_freq"],
            "dominant_freq_last": last["dominant_freq"],
            "dominant_freq_delta": df,
            "centroid_prev": prev["spectral_centroid"],
            "centroid_last": last["spectral_centroid"],
            "centroid_delta": dc,
            "step_prev": prev["step"],
            "step_last": last["step"],
        }
