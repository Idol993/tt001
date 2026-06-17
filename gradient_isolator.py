import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from stft_analysis import STFTGradientAnalyzer
from notch_filter import ModuleNotchFilterBank


@dataclass
class ParamSliceInfo:
    param_name: str
    module_name: str
    offset: int
    length: int
    shape: torch.Size
    numel: int


class GradientIsolator(nn.Module):
    def __init__(
        self,
        n_fft: int = 64,
        hop_length: int = 16,
        num_modules: int = 4,
        num_filters_per_module: int = 6,
        overlap_threshold: float = 0.3,
        peak_threshold: float = 0.3,
        adaptive_init_interval: int = 100,
        reg_weight: float = 0.01,
        crosstalk_loss_weight: float = 0.05,
        device: str = "cpu",
    ):
        super().__init__()
        self.n_fft = n_fft
        self.overlap_threshold = overlap_threshold
        self.adaptive_init_interval = adaptive_init_interval
        self.reg_weight = reg_weight
        self.crosstalk_loss_weight = crosstalk_loss_weight
        self._step_count = 0
        self.device = device

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

        self._module_param_slices: Dict[str, List[ParamSliceInfo]] = {}
        self._module_total_sizes: Dict[str, int] = {}
        self._last_writeback_validation: Optional[Dict] = None

    def register_model_structure(
        self, model: nn.Module, module_names: List[str]
    ) -> None:
        self._module_param_slices = {name: [] for name in module_names}
        self._module_total_sizes = {name: 0 for name in module_names}
        running_offsets = {name: 0 for name in module_names}

        for name, param in model.named_parameters():
            for mod_name in module_names:
                if name.startswith(mod_name):
                    numel = param.numel()
                    slice_info = ParamSliceInfo(
                        param_name=name,
                        module_name=mod_name,
                        offset=running_offsets[mod_name],
                        length=numel,
                        shape=param.shape,
                        numel=numel,
                    )
                    self._module_param_slices[mod_name].append(slice_info)
                    running_offsets[mod_name] += numel
                    self._module_total_sizes[mod_name] = running_offsets[mod_name]
                    break

    def collect_module_gradients(
        self, model: nn.Module, module_names: List[str]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[ParamSliceInfo]]]:
        if not self._module_param_slices:
            self.register_model_structure(model, module_names)

        grads = {}
        module_grad_norms = {}
        for mod_name in module_names:
            total_size = self._module_total_sizes[mod_name]
            module_grad = torch.zeros(
                total_size,
                device=self.device,
                dtype=next(model.parameters()).dtype,
            )
            slices = self._module_param_slices[mod_name]
            filled_count = 0
            for slice_info in slices:
                param = dict(model.named_parameters())[slice_info.param_name]
                if param.grad is not None:
                    grad_flat = param.grad.detach().reshape(-1)
                    module_grad[slice_info.offset:slice_info.offset + slice_info.length] = grad_flat
                    filled_count += 1
            grads[mod_name] = module_grad
            module_grad_norms[mod_name] = {
                "norm": module_grad.norm().item(),
                "total_params": len(slices),
                "params_with_grad": filled_count,
                "total_elements": total_size,
                "all_zero": module_grad.abs().sum().item() == 0,
            }
        self._last_module_grad_info = module_grad_norms
        return grads, self._module_param_slices

    def get_last_module_grad_info(self) -> Dict:
        return getattr(self, "_last_module_grad_info", {})

    def initialize_filters_from_peaks(
        self, module_grads: Dict[str, torch.Tensor], module_names: List[str]
    ) -> Dict[str, Dict]:
        analysis = self.analyzer.analyze_module_gradients(module_grads)
        name_to_idx = {name: idx for idx, name in enumerate(module_names)}

        peak_info = {}
        for name in module_names:
            info = analysis[name]
            peak_indices = info["peak_indices"]
            if len(peak_indices) > 0:
                mod_idx = name_to_idx[name]
                self.filter_bank.filters[mod_idx].initialize_from_peaks(peak_indices)
            peak_info[name] = {
                "peak_indices": peak_indices.cpu().tolist(),
                "peak_magnitudes": info["peak_magnitudes"].cpu().tolist(),
                "freq_profile": info["freq_profile"].cpu().numpy(),
            }
        return peak_info

    def isolate_gradients(
        self,
        module_grads: Dict[str, torch.Tensor],
        module_names: List[str],
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        magnitudes = []
        phases = []
        orig_sizes = {}
        freq_profiles = {}

        for name in module_names:
            mag, phase = self.analyzer.forward_stft(module_grads[name])
            magnitudes.append(mag)
            phases.append(phase)
            orig_sizes[name] = module_grads[name].numel()
            freq_profiles[name] = mag.mean(dim=-1).detach().cpu()

        filtered_mags = self.filter_bank(magnitudes)

        isolated_grads = {}
        filter_masks = {}
        for idx, name in enumerate(module_names):
            reconstructed = self.analyzer.inverse_stft(
                filtered_mags[idx], phases[idx], orig_sizes[name]
            )
            target_numel = orig_sizes[name]
            if reconstructed.numel() >= target_numel:
                isolated_grads[name] = reconstructed[:target_numel]
            else:
                padded = torch.zeros(
                    target_numel,
                    device=reconstructed.device,
                    dtype=reconstructed.dtype,
                )
                padded[:reconstructed.numel()] = reconstructed
                isolated_grads[name] = padded

            filter_masks[name] = self.filter_bank.filters[idx].get_filter_mask_numpy()

        isolation_stats = {
            "freq_profiles": freq_profiles,
            "filter_masks": filter_masks,
            "original_magnitudes": [m.detach().cpu() for m in magnitudes],
            "filtered_magnitudes": [m.detach().cpu() for m in filtered_mags],
        }

        return isolated_grads, isolation_stats

    def forward(
        self,
        module_grads: Dict[str, torch.Tensor],
        module_names: List[str],
        adapt_filters: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        self._step_count += 1

        peak_info = {}
        if adapt_filters or self._step_count % self.adaptive_init_interval == 1:
            peak_info = self.initialize_filters_from_peaks(module_grads, module_names)

        isolated_grads, isolation_stats = self.isolate_gradients(module_grads, module_names)

        if self.filter_bank.track_history:
            self.filter_bank.record_history(self._step_count)

        stats = {
            "peak_info": peak_info,
            **isolation_stats,
            "step": self._step_count,
        }

        return isolated_grads, stats

    def compute_auxiliary_losses(
        self,
        module_grads: Dict[str, torch.Tensor],
        module_names: List[str],
    ) -> Dict[str, torch.Tensor]:
        magnitudes = []
        for name in module_names:
            grad_with_grad = module_grads[name].clone().detach().requires_grad_(False)
            mag, _ = self.analyzer.forward_stft(grad_with_grad)
            mag_for_loss = mag.detach()
            mag_param_flow, _ = self.analyzer.forward_stft(module_grads[name].clone())
            magnitudes.append(mag_param_flow)

        reg_loss = self.reg_weight * self.filter_bank.regularization_loss()
        crosstalk_loss, loss_breakdown = self.filter_bank.compute_crosstalk_loss(
            magnitudes, module_names, self.crosstalk_loss_weight
        )

        return {
            "reg_loss": reg_loss,
            "crosstalk_loss": crosstalk_loss,
            "total_aux_loss": reg_loss + crosstalk_loss,
            "loss_breakdown": loss_breakdown,
        }

    def get_regularization_loss(self) -> torch.Tensor:
        return self.reg_weight * self.filter_bank.regularization_loss()

    def redistribute_gradients(
        self,
        model: nn.Module,
        isolated_grads: Dict[str, torch.Tensor],
        module_names: List[str],
        validate: bool = True,
    ) -> Dict:
        validation = {
            "total_params": 0,
            "validated_params": 0,
            "shape_matches": True,
            "count_matches": True,
            "mismatches": [],
            "per_module": {},
        }

        for mod_name in module_names:
            mod_stats = {
                "total_params": 0,
                "validated_params": 0,
                "total_elements": 0,
                "mismatches": [],
            }
            if mod_name not in isolated_grads or mod_name not in self._module_param_slices:
                validation["per_module"][mod_name] = mod_stats
                continue

            isolated = isolated_grads[mod_name]
            slices = self._module_param_slices[mod_name]

            for slice_info in slices:
                param = dict(model.named_parameters())[slice_info.param_name]
                if param.grad is None:
                    continue

                mod_stats["total_params"] += 1
                mod_stats["total_elements"] += slice_info.numel
                validation["total_params"] += 1

                slice_start = slice_info.offset
                slice_end = slice_info.offset + slice_info.length

                if slice_end > isolated.numel():
                    msg = f"isolated grad too small: {isolated.numel()} < {slice_end}"
                    validation["mismatches"].append({
                        "param": slice_info.param_name,
                        "reason": msg,
                    })
                    mod_stats["mismatches"].append({
                        "param": slice_info.param_name,
                        "reason": msg,
                    })
                    validation["count_matches"] = False
                    continue

                grad_slice = isolated[slice_start:slice_end]

                if grad_slice.numel() != slice_info.numel:
                    msg = f"numel mismatch: {grad_slice.numel()} != {slice_info.numel}"
                    validation["mismatches"].append({
                        "param": slice_info.param_name,
                        "reason": msg,
                    })
                    mod_stats["mismatches"].append({
                        "param": slice_info.param_name,
                        "reason": msg,
                    })
                    validation["count_matches"] = False
                    continue

                if validate:
                    old_shape = param.grad.shape
                    old_numel = param.grad.numel()

                param.grad.copy_(grad_slice.reshape(slice_info.shape))

                if validate:
                    if param.grad.shape != slice_info.shape:
                        msg = f"shape mismatch after writeback: {param.grad.shape} != {slice_info.shape}"
                        validation["mismatches"].append({
                            "param": slice_info.param_name,
                            "reason": msg,
                        })
                        mod_stats["mismatches"].append({
                            "param": slice_info.param_name,
                            "reason": msg,
                        })
                        validation["shape_matches"] = False
                    elif old_shape != slice_info.shape or old_numel != slice_info.numel:
                        msg = f"shape/numel mismatch: expected {slice_info.shape} ({slice_info.numel} elements), got {old_shape} ({old_numel} elements)"
                        validation["mismatches"].append({
                            "param": slice_info.param_name,
                            "reason": msg,
                        })
                        mod_stats["mismatches"].append({
                            "param": slice_info.param_name,
                            "reason": msg,
                        })
                        validation["shape_matches"] = False
                    else:
                        validation["validated_params"] += 1
                        mod_stats["validated_params"] += 1
                else:
                    validation["validated_params"] += 1
                    mod_stats["validated_params"] += 1

            validation["per_module"][mod_name] = mod_stats

        self._last_writeback_validation = validation
        return validation

    def get_param_slices(self) -> Dict[str, List[ParamSliceInfo]]:
        return self._module_param_slices

    def get_last_validation(self) -> Optional[Dict]:
        return self._last_writeback_validation

    def get_filter_diagnostics(self) -> Dict:
        params = self.filter_bank.get_filter_parameters()
        diagnostics = {}
        for idx, p in enumerate(params):
            diagnostics[f"module_{idx}"] = {
                "center_frequencies": p["center_frequencies"].tolist(),
                "bandwidths": p["bandwidths"].tolist(),
                "mask_min": p["mask"].min().item(),
                "mask_max": p["mask"].max().item(),
                "mask_mean": p["mask"].mean().item(),
            }
        return diagnostics
