import torch
import torch.nn as nn
import math
from typing import Optional


class LearnableNotchFilter(nn.Module):
    def __init__(
        self,
        n_fft: int,
        num_filters: int = 8,
        init_bandwidth: float = 0.5,
        min_bandwidth: float = 0.01,
        max_bandwidth: float = 4.0,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.n_freq = n_fft // 2 + 1
        self.num_filters = num_filters
        self.min_bandwidth = min_bandwidth
        self.max_bandwidth = max_bandwidth

        init_centers = torch.linspace(0, self.n_freq - 1, num_filters + 2)[1:-1]
        self.log_center_freq = nn.Parameter(
            torch.log(init_centers + 1e-6)
        )
        self.log_bandwidth = nn.Parameter(
            torch.full((num_filters,), math.log(init_bandwidth))
        )

        freq_bins = torch.arange(0, self.n_freq, dtype=torch.float32)
        self.register_buffer("freq_bins", freq_bins)

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


class ModuleNotchFilterBank(nn.Module):
    def __init__(
        self,
        n_fft: int = 64,
        num_modules: int = 4,
        num_filters_per_module: int = 6,
        init_bandwidth: float = 0.5,
        min_bandwidth: float = 0.01,
        max_bandwidth: float = 4.0,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.num_modules = num_modules

        self.filters = nn.ModuleList([
            LearnableNotchFilter(
                n_fft=n_fft,
                num_filters=num_filters_per_module,
                init_bandwidth=init_bandwidth,
                min_bandwidth=min_bandwidth,
                max_bandwidth=max_bandwidth,
            )
            for _ in range(num_modules)
        ])

    def configure_from_overlap(
        self,
        overlapping_freqs: list,
        module_idx: int,
    ):
        if module_idx >= len(self.filters):
            return
        notch_filter = self.filters[module_idx]
        with torch.no_grad():
            centers = []
            for freq_idx in overlapping_freqs[:notch_filter.num_filters]:
                centers.append(float(freq_idx))
            while len(centers) < notch_filter.num_filters:
                centers.append(0.0)
            notch_filter.log_center_freq.copy_(
                torch.log(torch.tensor(centers, dtype=torch.float32) + 1e-6)
            )

    def forward(
        self,
        module_magnitudes: list,
    ) -> list:
        filtered = []
        for idx, magnitude in enumerate(module_magnitudes):
            filtered_mag = self.filters[idx](magnitude)
            filtered.append(filtered_mag)
        return filtered

    def get_filter_parameters(self) -> list:
        params = []
        for idx, f in enumerate(self.filters):
            params.append({
                "module_idx": idx,
                "center_frequencies": f.center_frequencies.detach(),
                "bandwidths": f.bandwidths.detach(),
            })
        return params

    def regularization_loss(self) -> torch.Tensor:
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for f in self.filters:
            bws = f.bandwidths
            loss = loss + (bws ** 2).mean()
        return loss
