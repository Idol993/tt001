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
        min_bandwidth: float = 0.05,
        max_bandwidth: float = 8.0,
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
            cf_per_filter = cf_grad.abs().tolist() if cf_grad is not None else [0.0] * self.num_filters
            bw_per_filter = bw_grad.abs().tolist() if bw_grad is not None else [0.0] * self.num_filters
            self._param_history.append({
                "step": step,
                "center_frequencies": centers,
                "bandwidths": bws,
            })
            self._grad_history.append({
                "step": step,
                "center_freq_grad_norm": cf_grad_norm,
                "bandwidth_grad_norm": bw_grad_norm,
                "center_freq_per_filter_grad": cf_per_filter,
                "bandwidth_per_filter_grad": bw_per_filter,
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
        min_bandwidth: float = 0.05,
        max_bandwidth: float = 8.0,
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

    def get_filter_delta(self, prev_params: Optional[List[Dict]] = None) -> List[Dict]:
        current = self.get_filter_parameters()
        deltas = []
        for idx, p in enumerate(current):
            entry = {
                "center_freq_delta": [0.0] * self.num_filters_per_module,
                "bandwidth_delta": [0.0] * self.num_filters_per_module,
                "max_cf_delta": 0.0,
                "max_bw_delta": 0.0,
            }
            if prev_params and idx < len(prev_params):
                prev = prev_params[idx]
                cf_curr = p["center_frequencies"].tolist()
                cf_prev = prev["center_frequencies"].tolist()
                bw_curr = p["bandwidths"].tolist()
                bw_prev = prev["bandwidths"].tolist()
                entry["center_freq_delta"] = [c - p_v for c, p_v in zip(cf_curr, cf_prev)]
                entry["bandwidth_delta"] = [c - p_v for c, p_v in zip(bw_curr, bw_prev)]
                entry["max_cf_delta"] = max(abs(d) for d in entry["center_freq_delta"])
                entry["max_bw_delta"] = max(abs(d) for d in entry["bandwidth_delta"])
            deltas.append(entry)
        return deltas

    def get_param_history(self) -> List[List[Dict]]:
        return [f.get_param_history() for f in self.filters]

    def get_grad_history(self) -> List[List[Dict]]:
        return [f.get_grad_history() for f in self.filters]

    def compute_crosstalk_loss(
        self,
        module_magnitudes: List[torch.Tensor],
        module_names: List[str],
        crosstalk_weight: float = 0.1,
    ) -> Tuple[torch.Tensor, Dict]:
        loss_details = {
            "overlap_loss": 0.0,
            "self_preserve_loss": 0.0,
            "cross_suppress_loss": 0.0,
        }
        if self.num_modules < 2:
            return torch.tensor(0.0, device=next(self.parameters()).device), loss_details

        masks = [f.compute_filter_mask() for f in self.filters]
        freq_profiles = [m.mean(dim=-1) for m in module_magnitudes]
        total_energy = sum(p.sum() for p in freq_profiles) + 1e-8

        overlap_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        self_preserve_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        cross_suppress_loss = torch.tensor(0.0, device=next(self.parameters()).device)

        for i in range(self.num_modules):
            prof_i = freq_profiles[i] / (freq_profiles[i].sum() + 1e-8)
            topk_vals, topk_idx = torch.topk(prof_i, min(3, self.n_freq))
            for k_idx in topk_idx:
                self_preserve_loss = self_preserve_loss + (1.0 - masks[i][k_idx]) * prof_i[k_idx]

            for j in range(self.num_modules):
                if i == j:
                    continue
                overlap = masks[i] * masks[j]
                energy_ij = (freq_profiles[i] * overlap).sum()
                overlap_loss = overlap_loss + energy_ij / total_energy

                prof_j = freq_profiles[j] / (freq_profiles[j].sum() + 1e-8)
                topk_vals_j, topk_idx_j = torch.topk(prof_j, min(3, self.n_freq))
                for k_idx in topk_idx_j:
                    cross_suppress_loss = cross_suppress_loss + masks[i][k_idx] * prof_j[k_idx]

        loss_details["overlap_loss"] = overlap_loss.item()
        loss_details["self_preserve_loss"] = self_preserve_loss.item()
        loss_details["cross_suppress_loss"] = cross_suppress_loss.item()

        total = (
            1.0 * overlap_loss
            + 2.0 * self_preserve_loss
            + 1.5 * cross_suppress_loss
        )
        return crosstalk_weight * total, loss_details

    def regularization_loss(self) -> torch.Tensor:
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for f in self.filters:
            bws = f.bandwidths
            centers = f.center_frequencies
            loss = loss + 0.1 * (bws ** 2).mean()
            sorted_centers, _ = torch.sort(centers)
            center_spacing = sorted_centers[1:] - sorted_centers[:-1]
            loss = loss + 0.5 * torch.exp(-5.0 * center_spacing).mean()
            loss = loss + 0.01 * ((centers - self.n_freq / 2) ** 2).mean() / (self.n_freq ** 2)
        return loss

    def get_all_masks(self) -> torch.Tensor:
        with torch.no_grad():
            masks = torch.stack([f.get_filter_mask_numpy() for f in self.filters], dim=0)
            return masks
