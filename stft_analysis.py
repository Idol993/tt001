import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional


class STFTGradientAnalyzer(nn.Module):
    def __init__(
        self,
        n_fft: int = 64,
        hop_length: int = 16,
        win_length: Optional[int] = None,
        peak_threshold: float = 0.3,
        min_peak_distance: int = 3,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.peak_threshold = peak_threshold
        self.min_peak_distance = min_peak_distance

    def _reshape_to_1d(self, grad: torch.Tensor) -> torch.Tensor:
        flat = grad.reshape(-1)
        pad_len = (self.n_fft - flat.numel() % self.n_fft) % self.n_fft
        if pad_len > 0:
            flat = torch.nn.functional.pad(flat, (0, pad_len))
        return flat

    def forward_stft(self, grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        signal = self._reshape_to_1d(grad)
        window = torch.hann_window(self.win_length, device=grad.device, dtype=grad.dtype)
        stft_out = torch.stft(
            signal,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
            normalized=True,
        )
        magnitude = stft_out.abs()
        phase = stft_out.angle()
        return magnitude, phase

    def inverse_stft(
        self, magnitude: torch.Tensor, phase: torch.Tensor, orig_size: int
    ) -> torch.Tensor:
        complex_spec = magnitude * torch.exp(1j * phase)
        window = torch.hann_window(self.win_length, device=magnitude.device, dtype=magnitude.dtype)
        signal = torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            length=orig_size,
            normalized=True,
        )
        return signal

    def detect_frequency_peaks(self, magnitude: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        freq_profile = magnitude.mean(dim=-1)
        freq_profile_norm = freq_profile / (freq_profile.max() + 1e-8)

        threshold = self.peak_threshold
        above = freq_profile_norm > threshold

        peaks_mask = torch.zeros_like(above)
        for i in range(len(freq_profile_norm)):
            if not above[i]:
                continue
            left = max(0, i - self.min_peak_distance)
            right = min(len(freq_profile_norm), i + self.min_peak_distance + 1)
            if freq_profile_norm[i] >= freq_profile_norm[left:right].max():
                peaks_mask[i] = True

        peak_indices = peaks_mask.nonzero(as_tuple=True)[0]
        peak_magnitudes = freq_profile_norm[peak_indices]
        return peak_indices, peak_magnitudes

    def analyze_module_gradients(
        self, module_grads: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict]:
        analysis = {}
        for name, grad in module_grads.items():
            magnitude, phase = self.forward_stft(grad)
            peak_indices, peak_mags = self.detect_frequency_peaks(magnitude)
            analysis[name] = {
                "magnitude": magnitude,
                "phase": phase,
                "peak_indices": peak_indices,
                "peak_magnitudes": peak_mags,
                "freq_profile": magnitude.mean(dim=-1),
                "orig_size": grad.numel(),
            }
        return analysis

    def find_overlapping_peaks(
        self,
        analysis: Dict[str, Dict],
        overlap_threshold: float = 0.5,
    ) -> Dict[str, List[int]]:
        n_freq = self.n_fft // 2 + 1
        module_peak_sets = {}
        for name, info in analysis.items():
            peaks = info["peak_indices"]
            peak_set = set()
            for p in peaks:
                p_val = p.item()
                for offset in range(-2, 3):
                    idx = p_val + offset
                    if 0 <= idx < n_freq:
                        peak_set.add(idx)
            module_peak_sets[name] = peak_set

        overlap_map = {}
        module_names = list(analysis.keys())
        for i, name_a in enumerate(module_names):
            conflicting = []
            for j, name_b in enumerate(module_names):
                if i == j:
                    continue
                intersection = module_peak_sets[name_a] & module_peak_sets[name_b]
                if len(intersection) > 0:
                    union = module_peak_sets[name_a] | module_peak_sets[name_b]
                    iou = len(intersection) / (len(union) + 1e-8)
                    if iou > overlap_threshold:
                        conflicting.extend(list(intersection))
            overlap_map[name_a] = list(set(conflicting))
        return overlap_map
