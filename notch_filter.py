import torch
import torch.nn as nn
import math
from typing import Optional, List, Dict, Tuple
from collections import defaultdict


class LearnableNotchFilter(nn.Module):
    def __init__(
        self,
        n_fft: int,
        num_filters: int = 8,
        init_bandwidth: float = 0.5,
        min_bandwidth: float = 0.01,
        max_bandwidth: float = 4.0,
        track_history: bool = True,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.n_freq = n_fft // 2 + 1
        self.num_filters = num_filters
        self.min_bandwidth = min_bandwidth
        self.max_bandwidth = max_bandwidth
        self.track_history = track_history

        init_centers = torch.linspace(0, self.n_freq - 1, num_filters + 2)[1:-1]
        self.log_center_freq = nn.Parameter(
            torch.log(init_centers + 1e-6),
            requires_grad=True,
        )
        self.log_bandwidth = nn.Parameter(
            torch.full((num_filters,), math.log(init_bandwidth)),
            requires_grad=True,
        )

        freq_bins = torch.arange(0, self.n_freq, dtype=torch.float32)
        self.register_buffer("freq_bins", freq_bins)

        if self.track_history:
            self._param_history: List[Dict] = []
            self._grad_history: List[Dict] = []

    @property
    def center_frequencies(self) -> torch.Tensor:
        return torch.exp(self.log_center_freq) - 1e-6

    @property
    def bandwidths(self) -> torch.Tensor:
        raw = torch.exp(self.log_bandwidth)
        return torch.clamp(raw, self.min_bandwidth, self.max_bandwidth)

    def compute_filter_mask(self) -> torch.Tensor:
        centers = self.center_frequencies
        bws = self.bandwidths

        freq_bins = self.freq_bins.unsqueeze(-1)
        centers_expanded = centers.unsqueeze(0)
        bws_expanded = bws.unsqueeze(0)

        squared_dist = (freq_bins - centers_expanded) ** 2
        bw_sq = (bws_expanded / 2.0) ** 2

        notch_response = squared_dist / (squared_dist + bw_sq + 1e-8)
        combined = notch_response.prod(dim=-1)
        combined = torch.clamp(combined, 0.0, 1.0)
        return combined

    def forward(self, magnitude: torch.Tensor) -> torch.Tensor:
        mask = self.compute_filter_mask()
        mask = mask.unsqueeze(-1).expand_as(magnitude)
        return magnitude * mask

    def get_filter_mask_numpy(self) -> torch.Tensor:
        with torch.no_grad():
            return self.compute_filter_mask().detach().cpu()

    def record_history(self, step: int):
        if not self.track_history:
            return
        with torch.no_grad():
            centers = self.center_frequencies.detach().cpu().tolist()
            bws = self.bandwidths.detach().cpu().tolist()
            cf_grad = self.log_center_freq.grad
            bw_grad = self.log_bandwidth.grad
            cf_grad_norm = cf_grad.norm().item() if cf_grad is not None else 0.0
            bw_grad_norm = bw_grad.norm().item() if bw_grad is not None else 0.0
            self._param_history.append({
                "step": step,
                "center_frequencies": centers,
                "bandwidths": bws,
            })
            self._grad_history.append({
                "step": step,
                "center_freq_grad_norm": cf_grad_norm,
                "bandwidth_grad_norm": bw_grad_norm,
            })

    def get_param_history(self) -> List[Dict]:
        return self._param_history

    def get_grad_history(self) -> List[Dict]:
        return self._grad_history

    def initialize_from_peaks(self, peak_indices: torch.Tensor):
        with torch.no_grad():
            peaks = peak_indices.detach().cpu()
            num_peaks = min(len(peaks), self.num_filters)
            new_centers = torch.zeros(self.num_filters)
            for i in range(num_peaks):
                new_centers[i] = float(peaks[i])
            for i in range(num_peaks, self.num_filters):
                new_centers[i] = float(torch.randint(0, self.n_freq, (1,)).item())
            self.log_center_freq.copy_(torch.log(new_centers + 1e-6))


class ModuleNotchFilterBank(nn.Module):
    def __init__(
        self,
        n_fft: int = 64,
        num_modules: int = 4,
        num_filters_per_module: int = 6,
        init_bandwidth: float = 0.5,
        min_bandwidth: float = 0.01,
        max_bandwidth: float = 4.0,
        track_history: bool = True,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.n_freq = n_fft // 2 + 1
        self.num_modules = num_modules
        self.num_filters_per_module = num_filters_per_module
        self.track_history = track_history

        self.filters = nn.ModuleList([
            LearnableNotchFilter(
                n_fft=n_fft,
                num_filters=num_filters_per_module,
                init_bandwidth=init_bandwidth,
                min_bandwidth=min_bandwidth,
                max_bandwidth=max_bandwidth,
                track_history=track_history,
            )
            for _ in range(num_modules)
        ])

    def forward(
        self,
        module_magnitudes: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        filtered = []
        for idx, magnitude in enumerate(module_magnitudes):
            filtered_mag = self.filters[idx](magnitude)
            filtered.append(filtered_mag)
        return filtered

    def record_history(self, step: int):
        for f in self.filters:
            f.record_history(step)

    def get_filter_parameters(self) -> List[Dict]:
        params = []
        for idx, f in enumerate(self.filters):
            params.append({
                "module_idx": idx,
                "center_frequencies": f.center_frequencies.detach().cpu().clone(),
                "bandwidths": f.bandwidths.detach().cpu().clone(),
                "mask": f.get_filter_mask_numpy(),
            })
        return params

    def get_param_history(self) -> List[List[Dict]]:
        return [f.get_param_history() for f in self.filters]

    def get_grad_history(self) -> List[List[Dict]]:
        return [f.get_grad_history() for f in self.filters]

    def compute_crosstalk_loss(
        self,
        module_magnitudes: List[torch.Tensor],
        module_names: List[str],
        crosstalk_weight: float = 0.1,
    ) -> torch.Tensor:
        if self.num_modules < 2:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        masks = [f.compute_filter_mask() for f in self.filters]
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)

        for i in range(self.num_modules):
            for j in range(self.num_modules):
                if i == j:
                    continue
                overlap = masks[i] * masks[j]
                energy = (module_magnitudes[i].mean(dim=-1) * overlap).sum()
                total_loss = total_loss + energy
        return crosstalk_weight * total_loss

    def regularization_loss(self) -> torch.Tensor:
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for f in self.filters:
            bws = f.bandwidths
            centers = f.center_frequencies
            loss = loss + (bws ** 2).mean()
            sorted_centers, _ = torch.sort(centers)
            center_spacing = sorted_centers[1:] - sorted_centers[:-1]
            loss = loss + 0.1 * torch.exp(-10 * center_spacing).mean()
        return loss

    def get_all_masks(self) -> torch.Tensor:
        with torch.no_grad():
            masks = torch.stack([f.get_filter_mask_numpy() for f in self.filters], dim=0)
            return masks
