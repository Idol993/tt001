import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from stft_analysis import STFTGradientAnalyzer
from notch_filter import ModuleNotchFilterBank


class GradientIsolator(nn.Module):
    def __init__(
        self,
        n_fft: int = 64,
        hop_length: int = 16,
        num_modules: int = 4,
        num_filters_per_module: int = 6,
        overlap_threshold: float = 0.3,
        peak_threshold: float = 0.3,
        adaptive_update_interval: int = 10,
        reg_weight: float = 0.01,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.overlap_threshold = overlap_threshold
        self.adaptive_update_interval = adaptive_update_interval
        self.reg_weight = reg_weight
        self._step_count = 0

        self.analyzer = STFTGradientAnalyzer(
            n_fft=n_fft,
            hop_length=hop_length,
            peak_threshold=peak_threshold,
        )

        self.filter_bank = ModuleNotchFilterBank(
            n_fft=n_fft,
            num_modules=num_modules,
            num_filters_per_module=num_filters_per_module,
        )

    def collect_module_gradients(
        self, model: nn.Module, module_names: List[str]
    ) -> Dict[str, torch.Tensor]:
        grads = {}
        for name, param in model.named_parameters():
            for mod_name in module_names:
                if name.startswith(mod_name) and param.grad is not None:
                    if mod_name not in grads:
                        grads[mod_name] = []
                    grads[mod_name].append(param.grad.detach().clone())
        merged = {}
        for mod_name, grad_list in grads.items():
            merged[mod_name] = torch.cat([g.reshape(-1) for g in grad_list])
        return merged

    def adapt_filter_configuration(
        self, module_grads: Dict[str, torch.Tensor]
    ) -> Dict[str, List[int]]:
        analysis = self.analyzer.analyze_module_gradients(module_grads)
        overlap_map = self.analyzer.find_overlapping_peaks(
            analysis, overlap_threshold=self.overlap_threshold
        )

        module_names = list(module_grads.keys())
        name_to_idx = {name: idx for idx, name in enumerate(module_names)}

        for name, conflicting_freqs in overlap_map.items():
            if name in name_to_idx and len(conflicting_freqs) > 0:
                self.filter_bank.configure_from_overlap(
                    overlapping_freqs=conflicting_freqs,
                    module_idx=name_to_idx[name],
                )
        return overlap_map

    def isolate_gradients(
        self,
        module_grads: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        module_names = list(module_grads.keys())
        magnitudes = []
        phases = []
        orig_sizes = {}

        for name in module_names:
            mag, phase = self.analyzer.forward_stft(module_grads[name])
            magnitudes.append(mag)
            phases.append(phase)
            orig_sizes[name] = module_grads[name].numel()

        filtered_mags = self.filter_bank(magnitudes)

        isolated_grads = {}
        for idx, name in enumerate(module_names):
            reconstructed = self.analyzer.inverse_stft(
                filtered_mags[idx], phases[idx], orig_sizes[name]
            )
            target_shape = module_grads[name].shape
            if reconstructed.numel() >= target_shape.numel():
                isolated_grads[name] = reconstructed[:target_shape.numel()].reshape(
                    target_shape
                )
            else:
                padded = torch.zeros(
                    target_shape.numel(),
                    device=reconstructed.device,
                    dtype=reconstructed.dtype,
                )
                padded[: reconstructed.numel()] = reconstructed
                isolated_grads[name] = padded.reshape(target_shape)

        return isolated_grads

    def forward(
        self,
        module_grads: Dict[str, torch.Tensor],
        adapt_filters: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[int]]]:
        self._step_count += 1

        overlap_map = {}
        if adapt_filters or self._step_count % self.adaptive_update_interval == 1:
            overlap_map = self.adapt_filter_configuration(module_grads)

        isolated_grads = self.isolate_gradients(module_grads)
        return isolated_grads, overlap_map

    def get_regularization_loss(self) -> torch.Tensor:
        return self.reg_weight * self.filter_bank.regularization_loss()

    def redistribute_gradients(
        self,
        model: nn.Module,
        isolated_grads: Dict[str, torch.Tensor],
        module_names: List[str],
    ) -> None:
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            for mod_name in module_names:
                if name.startswith(mod_name) and mod_name in isolated_grads:
                    param_shape = param.grad.shape
                    param_numel = param.grad.numel()
                    isolated = isolated_grads[mod_name]
                    if isolated.numel() >= param_numel:
                        param.grad.copy_(
                            isolated[:param_numel].reshape(param_shape)
                        )
                    break
