import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import math
from dataclasses import dataclass, field


@dataclass
class BandwidthAccount:
    target_budget_ratio: float = 0.01
    original_bytes: int = 0
    data_bytes: int = 0
    metadata_bytes: int = 0
    total_bytes: int = 0
    actual_ratio: float = 0.0
    within_budget: bool = False
    adjusted_this_step: bool = False
    previous_ratio: Optional[float] = None
    adjustment_details: Optional[Dict] = None
    details: Dict = field(default_factory=dict)

    def compute(self) -> float:
        self.total_bytes = self.data_bytes + self.metadata_bytes
        self.actual_ratio = self.total_bytes / max(self.original_bytes, 1)
        self.within_budget = self.actual_ratio <= self.target_budget_ratio
        return self.actual_ratio


def _pack_bits_to_bytes(bits: torch.Tensor) -> torch.Tensor:
    flat = bits.reshape(-1).to(torch.uint8)
    numel = flat.numel()
    num_bytes = math.ceil(numel / 8)
    padded = torch.zeros(num_bytes * 8, dtype=torch.uint8, device=bits.device)
    padded[:numel] = flat
    padded = padded.reshape(num_bytes, 8)
    weights = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=bits.device)
    bytes_tensor = (padded * weights).sum(dim=1).to(torch.uint8)
    return bytes_tensor


def _unpack_bytes_to_bits(bytes_tensor: torch.Tensor, numel: int) -> torch.Tensor:
    num_bytes = bytes_tensor.numel()
    weights = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=bytes_tensor.device)
    bits = (bytes_tensor.unsqueeze(1) & weights.unsqueeze(0)) > 0
    bits = bits.reshape(-1)
    return bits[:numel]


def calculate_index_bytes(numel: int, k: int) -> int:
    bits_per_index = max(1, math.ceil(math.log2(max(numel, 2))))
    bytes_per_index = math.ceil(bits_per_index / 8)
    return k * bytes_per_index


class OneBitSGDCompressor:
    def __init__(
        self,
        momentum_factor: float = 0.9,
        warmup_steps: int = 0,
        target_budget_ratio: float = 0.01,
        auto_adjust: bool = True,
        sample_ratio: float = 1.0,
        min_sample_ratio: float = 0.002,
    ):
        self.momentum_factor = momentum_factor
        self.warmup_steps = warmup_steps
        self.target_budget_ratio = target_budget_ratio
        self.auto_adjust = auto_adjust
        self.sample_ratio = sample_ratio
        self.min_sample_ratio = min_sample_ratio
        self._step = 0
        self._error_buffers: Dict[str, torch.Tensor] = {}
        self._momentum_buffers: Dict[str, torch.Tensor] = {}
        self._last_bandwidth: Optional[BandwidthAccount] = None
        self._last_adjustment: Optional[Dict] = None
        self._adjustment_count = 0

    def compress(
        self, tensor: torch.Tensor, key: str
    ) -> Tuple[torch.Tensor, dict, BandwidthAccount]:
        self._step += 1
        bw = BandwidthAccount(target_budget_ratio=self.target_budget_ratio)
        bw.original_bytes = tensor.numel() * 4

        if key not in self._error_buffers:
            self._error_buffers[key] = torch.zeros_like(tensor)
        if key not in self._momentum_buffers:
            self._momentum_buffers[key] = torch.zeros_like(tensor)

        corrected = tensor + self._error_buffers[key]
        self._momentum_buffers[key] = (
            self.momentum_factor * self._momentum_buffers[key]
            + (1 - self.momentum_factor) * corrected
        )

        if self._step <= self.warmup_steps:
            msg = self._momentum_buffers[key]
            self._error_buffers[key] = corrected - msg
            bw.data_bytes = msg.numel() * 4
            bw.metadata_bytes = 0
            bw.details = {"method": "full_warmup", "warmup": True}
            bw.compute()
            return msg, {"method": "full", "original_shape": tensor.shape}, bw

        numel = tensor.numel()
        flat = self._momentum_buffers[key].reshape(-1)

        if self.sample_ratio >= 1.0 - 1e-9:
            sample_idx = None
            sampled_flat = flat
            sampled_numel = numel
        else:
            sample_k = max(1, int(numel * self.sample_ratio))
            sample_k = min(sample_k, numel)
            _, sample_idx = torch.topk(flat.abs(), sample_k)
            sampled_flat = flat[sample_idx]
            sampled_numel = sample_k

        mean_abs = sampled_flat.abs().mean()
        signs_bool = sampled_flat >= 0
        sign_bytes_tensor = _pack_bits_to_bytes(signs_bool)
        sign_bytes = sign_bytes_tensor.numel()

        if sample_idx is None:
            reconstructed = _unpack_bytes_to_bits(sign_bytes_tensor, sampled_numel)
            reconstructed = reconstructed.to(torch.float32)
            reconstructed[reconstructed == 0] = -1.0
            reconstructed = reconstructed * mean_abs
            reconstructed = reconstructed.reshape(self._momentum_buffers[key].shape)
            index_bytes = 0
        else:
            sampled_recon = _unpack_bytes_to_bits(sign_bytes_tensor, sampled_numel)
            sampled_recon = sampled_recon.to(torch.float32)
            sampled_recon[sampled_recon == 0] = -1.0
            sampled_recon = sampled_recon * mean_abs
            reconstructed = torch.zeros_like(flat)
            reconstructed[sample_idx] = sampled_recon
            reconstructed = reconstructed.reshape(self._momentum_buffers[key].shape)
            index_bytes = calculate_index_bytes(numel, sampled_numel)

        self._error_buffers[key] = corrected - reconstructed

        bw.data_bytes = sign_bytes
        bw.metadata_bytes = 4 + index_bytes
        bw.details = {
            "method": "1bit",
            "sign_bytes": sign_bytes,
            "scale_bytes": 4,
            "index_bytes": index_bytes,
            "numel": numel,
            "sampled_numel": sampled_numel,
            "sample_ratio": self.sample_ratio,
            "mean_abs": mean_abs.item(),
            "actual_ratio": (sign_bytes + 4 + index_bytes) / (numel * 4),
        }
        bw.compute()

        self._last_bandwidth = bw
        return sign_bytes_tensor, {
            "method": "1bit",
            "original_shape": tensor.shape,
            "sign_bytes_packed": sign_bytes_tensor,
            "mean_abs": mean_abs,
            "sample_idx": sample_idx,
            "sampled_numel": sampled_numel,
            "numel": numel,
        }, bw

    def decompress(
        self, compressed: torch.Tensor, meta: dict
    ) -> torch.Tensor:
        if meta["method"] == "1bit" and "sign_bytes_packed" in meta:
            sampled_numel = meta.get("sampled_numel", meta["numel"])
            sampled_recon = _unpack_bytes_to_bits(meta["sign_bytes_packed"], sampled_numel)
            sampled_recon = sampled_recon.to(torch.float32)
            sampled_recon[sampled_recon == 0] = -1.0
            mean_abs = meta["mean_abs"]
            sampled_recon = sampled_recon * mean_abs
            if meta.get("sample_idx") is not None:
                reconstructed = torch.zeros(meta["numel"], dtype=torch.float32, device=compressed.device)
                reconstructed[meta["sample_idx"]] = sampled_recon
            else:
                reconstructed = sampled_recon
            return reconstructed.reshape(meta["original_shape"])
        return compressed.reshape(meta["original_shape"])

    def pop_last_adjustment(self) -> Optional[Dict]:
        adj = self._last_adjustment
        self._last_adjustment = None
        return adj


class TopKCompressor:
    def __init__(
        self,
        compression_ratio: float = 0.005,
        target_budget_ratio: float = 0.01,
        auto_adjust: bool = True,
        min_k: int = 1,
        max_adjustment_factor: float = 0.8,
    ):
        self.compression_ratio = compression_ratio
        self.target_budget_ratio = target_budget_ratio
        self.auto_adjust = auto_adjust
        self.min_k = min_k
        self.max_adjustment_factor = max_adjustment_factor
        self._last_bandwidth: Optional[BandwidthAccount] = None
        self._adjustment_count = 0
        self._last_adjustment: Optional[Dict] = None

    def _compute_k(self, numel: int) -> int:
        k = max(self.min_k, int(numel * self.compression_ratio))
        return min(k, numel)

    def compress(
        self, tensor: torch.Tensor, key: str
    ) -> Tuple[torch.Tensor, dict, BandwidthAccount]:
        bw = BandwidthAccount(target_budget_ratio=self.target_budget_ratio)
        flat = tensor.reshape(-1)
        numel = flat.numel()
        bw.original_bytes = numel * 4
        prev_ratio = self.compression_ratio

        k = self._compute_k(numel)
        topk_vals, topk_indices = torch.topk(flat.abs(), k)
        values = flat[topk_indices]

        index_bytes = calculate_index_bytes(numel, k)
        value_bytes = k * 4

        bw.data_bytes = value_bytes
        bw.metadata_bytes = index_bytes
        bw.details = {
            "method": "topk",
            "k": k,
            "numel": numel,
            "value_bytes": value_bytes,
            "index_bytes": index_bytes,
            "compression_ratio_before": self.compression_ratio,
            "bits_per_index": math.ceil(math.log2(max(numel, 2))),
        }
        bw.compute()

        if self.auto_adjust and not bw.within_budget:
            old_ratio = self.compression_ratio
            max_allowed_total = int(bw.original_bytes * self.target_budget_ratio)
            available = max_allowed_total
            per_item_bytes = 4 + math.ceil(max(1, math.log2(max(numel, 2))) / 8)
            new_k = max(self.min_k, available // per_item_bytes)
            new_ratio = new_k / numel
            self.compression_ratio = max(
                self.min_k / numel,
                min(new_ratio, self.compression_ratio * self.max_adjustment_factor),
            )
            self._adjustment_count += 1
            self._last_adjustment = {
                "step": self._adjustment_count,
                "module": key,
                "old_ratio": old_ratio,
                "new_ratio": self.compression_ratio,
                "old_k": k,
                "new_k": max(self.min_k, int(bw.original_bytes * self.compression_ratio / 4)),
                "reason": f"Budget exceeded: {bw.actual_ratio*100:.2f}% > {bw.target_budget_ratio*100:.2f}%",
            }
            k = self._compute_k(numel)
            topk_vals, topk_indices = torch.topk(flat.abs(), k)
            values = flat[topk_indices]

            index_bytes = calculate_index_bytes(numel, k)
            value_bytes = k * 4
            bw.data_bytes = value_bytes
            bw.metadata_bytes = index_bytes
            bw.adjusted_this_step = True
            bw.previous_ratio = prev_ratio
            bw.adjustment_details = self._last_adjustment
            bw.details.update({
                "k_adjusted": k,
                "adjustment_count": self._adjustment_count,
                "compression_ratio_after": self.compression_ratio,
                "old_compression_ratio": old_ratio,
            })
            bw.compute()

        self._last_bandwidth = bw
        return values, {
            "method": "topk",
            "indices": topk_indices,
            "original_shape": tensor.shape,
            "original_numel": numel,
            "k": k,
        }, bw

    def decompress(self, values: torch.Tensor, meta: dict) -> torch.Tensor:
        reconstructed = torch.zeros(
            meta["original_numel"],
            device=values.device,
            dtype=values.dtype,
        )
        reconstructed[meta["indices"]] = values
        return reconstructed.reshape(meta["original_shape"])

    def pop_last_adjustment(self) -> Optional[Dict]:
        adj = self._last_adjustment
        self._last_adjustment = None
        return adj


class HybridCompressor:
    def __init__(
        self,
        target_budget_ratio: float = 0.01,
        warmup_steps: int = 0,
        momentum_factor: float = 0.9,
        auto_adjust: bool = True,
        topk_fraction: float = 0.5,
        min_topk_fraction: float = 0.05,
        max_adjustment_factor: float = 0.5,
        residual_sample_ratio: Optional[float] = None,
        min_residual_sample_ratio: float = 0.002,
    ):
        self.target_budget_ratio = target_budget_ratio
        self.warmup_steps = warmup_steps
        self.auto_adjust = auto_adjust
        self.topk_fraction = topk_fraction
        self.min_topk_fraction = min_topk_fraction
        self.max_adjustment_factor = max_adjustment_factor
        self.min_residual_sample_ratio = min_residual_sample_ratio
        self._step = 0
        self._last_bandwidth: Optional[BandwidthAccount] = None
        self._last_adjustment: Optional[Dict] = None
        self._adjustment_count = 0

        if residual_sample_ratio is None:
            full_1bit_ratio = 1.0 / 32.0 + 4.0 / 1024.0
            desired_residual_budget = target_budget_ratio * 0.05
            residual_sample_ratio = min(1.0, max(min_residual_sample_ratio, desired_residual_budget / max(full_1bit_ratio, 1e-6)))
        self.residual_sample_ratio = residual_sample_ratio

        initial_topk_ratio = target_budget_ratio * 0.05
        self.onebit = OneBitSGDCompressor(
            momentum_factor=momentum_factor,
            warmup_steps=warmup_steps,
            target_budget_ratio=target_budget_ratio * 0.5,
            auto_adjust=False,
            sample_ratio=self.residual_sample_ratio,
            min_sample_ratio=min_residual_sample_ratio,
        )
        self.topk = TopKCompressor(
            compression_ratio=initial_topk_ratio,
            target_budget_ratio=target_budget_ratio * 0.5,
            auto_adjust=False,
        )

    def _recalculate_budgets(self):
        self.topk.target_budget_ratio = self.target_budget_ratio * self.topk_fraction
        self.onebit.target_budget_ratio = self.target_budget_ratio * (1 - self.topk_fraction)

    def compress(
        self, tensor: torch.Tensor, key: str
    ) -> Tuple[torch.Tensor, dict, BandwidthAccount]:
        self._step += 1
        numel = tensor.numel()

        if self._step <= self.warmup_steps:
            full_tensor = tensor.clone()
            bw = BandwidthAccount(target_budget_ratio=self.target_budget_ratio)
            bw.original_bytes = numel * 4
            bw.data_bytes = numel * 4
            bw.details = {"method": "full_warmup", "warmup": True}
            bw.compute()
            self._last_bandwidth = bw
            return full_tensor, {
                "method": "full",
                "original_shape": tensor.shape,
            }, bw

        topk_vals, topk_meta, topk_bw = self.topk.compress(tensor, f"{key}_topk")
        topk_reconstructed = self.topk.decompress(topk_vals, topk_meta)
        residual = tensor - topk_reconstructed

        onebit_vals, onebit_meta, onebit_bw = self.onebit.compress(residual, f"{key}_1bit")

        bw = BandwidthAccount(target_budget_ratio=self.target_budget_ratio)
        bw.original_bytes = numel * 4
        bw.data_bytes = topk_bw.data_bytes + onebit_bw.data_bytes
        bw.metadata_bytes = topk_bw.metadata_bytes + onebit_bw.metadata_bytes + 8
        bw.details = {
            "method": "hybrid",
            "topk_bw": topk_bw.__dict__,
            "onebit_bw": onebit_bw.__dict__,
        }
        bw.compute()

        previous_ratio = bw.actual_ratio

        if self.auto_adjust and not bw.within_budget:
            old_topk_ratio = self.topk.compression_ratio
            old_res_ratio = self.onebit.sample_ratio
            over_by = bw.actual_ratio - self.target_budget_ratio

            min_topk_ratio = self.topk.min_k / max(numel, 1)
            at_min_topk = self.topk.compression_ratio <= min_topk_ratio + 1e-9
            at_min_res = self.onebit.sample_ratio <= self.onebit.min_sample_ratio + 1e-9

            if at_min_topk and at_min_res:
                self._last_adjustment = {
                    "step": self._step,
                    "module": key,
                    "reason": (f"Budget exceeded: {bw.actual_ratio*100:.2f}% > "
                               f"{bw.target_budget_ratio*100:.2f}%, but at lower bound "
                               f"(min_topk_ratio={min_topk_ratio*100:.4f}%, "
                               f"min_residual_sample={self.onebit.min_sample_ratio*100:.2f}%)"),
                    "at_lower_bound": True,
                    "old_topk_ratio": old_topk_ratio,
                    "new_topk_ratio": old_topk_ratio,
                    "old_residual_ratio": old_res_ratio,
                    "new_residual_ratio": old_res_ratio,
                    "previous_ratio": previous_ratio,
                    "adjusted_ratio": previous_ratio,
                    "adjustment_count": self._adjustment_count,
                }
                bw.adjusted_this_step = False
            else:
                reduction_factor = max(
                    self.max_adjustment_factor,
                    self.target_budget_ratio / max(bw.actual_ratio, 1e-8),
                )

                new_topk_ratio = max(
                    min_topk_ratio,
                    self.topk.compression_ratio * reduction_factor,
                )
                new_res_ratio = max(
                    self.onebit.min_sample_ratio,
                    self.onebit.sample_ratio * reduction_factor,
                )

                actually_changed = (
                    abs(new_topk_ratio - old_topk_ratio) > 1e-12
                    or abs(new_res_ratio - old_res_ratio) > 1e-12
                )

                if not actually_changed:
                    self._last_adjustment = {
                        "step": self._step,
                        "module": key,
                        "reason": (f"Budget exceeded: {bw.actual_ratio*100:.2f}% > "
                                   f"{bw.target_budget_ratio*100:.2f}%, but no further reduction possible"),
                        "at_lower_bound": True,
                        "old_topk_ratio": old_topk_ratio,
                        "new_topk_ratio": old_topk_ratio,
                        "old_residual_ratio": old_res_ratio,
                        "new_residual_ratio": old_res_ratio,
                        "previous_ratio": previous_ratio,
                        "adjusted_ratio": previous_ratio,
                        "adjustment_count": self._adjustment_count,
                    }
                    bw.adjusted_this_step = False
                else:
                    self.topk.compression_ratio = new_topk_ratio
                    self.onebit.sample_ratio = new_res_ratio
                    self._recalculate_budgets()

                    topk_vals, topk_meta, topk_bw = self.topk.compress(tensor, f"{key}_topk_retry")
                    topk_reconstructed = self.topk.decompress(topk_vals, topk_meta)
                    residual = tensor - topk_reconstructed
                    onebit_vals, onebit_meta, onebit_bw = self.onebit.compress(residual, f"{key}_1bit_retry")

                    new_bw = BandwidthAccount(target_budget_ratio=self.target_budget_ratio)
                    new_bw.original_bytes = numel * 4
                    new_bw.data_bytes = topk_bw.data_bytes + onebit_bw.data_bytes
                    new_bw.metadata_bytes = topk_bw.metadata_bytes + onebit_bw.metadata_bytes + 8
                    new_bw.compute()

                    self._adjustment_count += 1
                    self._last_adjustment = {
                        "step": self._step,
                        "module": key,
                        "reason": (f"Hybrid exceeded: {previous_ratio*100:.2f}% > "
                                   f"{bw.target_budget_ratio*100:.2f}% (by {over_by*100:.2f}%)"),
                        "old_topk_ratio": old_topk_ratio,
                        "new_topk_ratio": new_topk_ratio,
                        "old_residual_ratio": old_res_ratio,
                        "new_residual_ratio": new_res_ratio,
                        "previous_ratio": previous_ratio,
                        "adjusted_ratio": new_bw.actual_ratio,
                        "adjustment_count": self._adjustment_count,
                        "at_lower_bound": False,
                    }

                    bw = new_bw
                    bw.adjusted_this_step = True
                    bw.previous_ratio = previous_ratio
                    bw.adjustment_details = self._last_adjustment

        bw.details["topk_ratio_current"] = self.topk.compression_ratio
        bw.details["residual_sample_ratio_current"] = self.onebit.sample_ratio
        bw.details["adjustment_count"] = self._adjustment_count

        self._last_bandwidth = bw
        combined = {
            "topk_vals": topk_vals,
            "topk_meta": topk_meta,
            "onebit_vals": onebit_vals,
            "onebit_meta": onebit_meta,
        }
        dummy = torch.tensor([0.0], device=tensor.device)
        return dummy, {
            "method": "hybrid",
            "combined": combined,
            "original_shape": tensor.shape,
            "topk_ratio": self.topk.compression_ratio,
            "residual_sample_ratio": self.onebit.sample_ratio,
        }, bw

    def decompress(self, compressed: torch.Tensor, meta: dict) -> torch.Tensor:
        if meta["method"] == "hybrid":
            combined = meta["combined"]
            topk_recon = self.topk.decompress(
                combined["topk_vals"], combined["topk_meta"]
            )
            onebit_recon = self.onebit.decompress(
                combined["onebit_vals"], combined["onebit_meta"]
            )
            return topk_recon + onebit_recon
        return compressed.reshape(meta["original_shape"])

    def pop_last_adjustment(self) -> Optional[Dict]:
        adj = self._last_adjustment
        self._last_adjustment = None
        return adj

    def get_current_params(self) -> Dict:
        return {
            "topk_compression_ratio": getattr(self.topk, 'compression_ratio', None),
            "residual_sample_ratio": getattr(self.onebit, 'sample_ratio', None),
            "adjustment_count": self._adjustment_count,
        }


class GradientCompressor:
    def __init__(
        self,
        method: str = "topk",
        target_budget_ratio: float = 0.01,
        warmup_steps: int = 0,
        momentum_factor: float = 0.9,
        auto_adjust: bool = True,
    ):
        self.method = method
        self.target_budget_ratio = target_budget_ratio
        self.auto_adjust = auto_adjust
        self._step = 0

        if method == "1bit":
            self.compressor = OneBitSGDCompressor(
                momentum_factor=momentum_factor,
                warmup_steps=warmup_steps,
                target_budget_ratio=target_budget_ratio,
                auto_adjust=auto_adjust,
            )
        elif method == "topk":
            self.compressor = TopKCompressor(
                compression_ratio=target_budget_ratio * 0.5,
                target_budget_ratio=target_budget_ratio,
                auto_adjust=auto_adjust,
            )
        elif method == "hybrid":
            self.compressor = HybridCompressor(
                target_budget_ratio=target_budget_ratio,
                warmup_steps=warmup_steps,
                momentum_factor=momentum_factor,
                auto_adjust=auto_adjust,
            )
        else:
            raise ValueError(f"Unknown compression method: {method}")

        self._bandwidth_history: List[BandwidthAccount] = []
        self._adjustment_log: List[Dict] = []

    def compress_module_gradients(
        self,
        isolated_grads: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, Tuple[torch.Tensor, dict]], BandwidthAccount]:
        self._step += 1
        compressed = {}
        total_bw = BandwidthAccount(target_budget_ratio=self.target_budget_ratio)
        any_adjusted = False
        prev_overall_ratio = self._bandwidth_history[-1].actual_ratio if self._bandwidth_history else None
        adjustments_this_step = []

        for name, grad in isolated_grads.items():
            compressed_tensor, meta, bw = self.compressor.compress(grad, name)
            compressed[name] = (compressed_tensor, meta)
            total_bw.original_bytes += bw.original_bytes
            total_bw.data_bytes += bw.data_bytes
            total_bw.metadata_bytes += bw.metadata_bytes
            if bw.adjusted_this_step:
                any_adjusted = True
                if bw.adjustment_details:
                    adjustments_this_step.append(bw.adjustment_details)

        total_bw.compute()
        if any_adjusted:
            total_bw.adjusted_this_step = True
            total_bw.previous_ratio = prev_overall_ratio
            total_bw.adjustment_details = {
                "step": self._step,
                "sub_adjustments": adjustments_this_step,
                "previous_overall_ratio": prev_overall_ratio,
            }
            self._adjustment_log.append(total_bw.adjustment_details)

        total_bw.details["module_breakdown"] = {}
        for name, (_, meta_inner) in compressed.items():
            method = meta_inner.get("method", "unknown")
            if method == "topk":
                k = meta_inner.get("k", 0)
                numel = meta_inner.get("original_numel", 1)
                ib = calculate_index_bytes(numel, k)
                vb = k * 4
                total_bw.details["module_breakdown"][name] = {
                    "data_bytes": vb,
                    "metadata_bytes": ib,
                    "actual_ratio": (ib + vb) / max(numel * 4, 1),
                    "k": k,
                    "numel": numel,
                }
            elif method == "1bit":
                numel = meta_inner.get("original_shape", torch.Size([0])).numel()
                sign_bytes = math.ceil(max(numel, 1) / 8)
                scale_bytes = 4
                total_bw.details["module_breakdown"][name] = {
                    "data_bytes": sign_bytes,
                    "metadata_bytes": scale_bytes,
                    "actual_ratio": (sign_bytes + scale_bytes) / max(numel * 4, 1),
                    "numel": numel,
                }
            else:
                total_bw.details["module_breakdown"][name] = {
                    "data_bytes": 0,
                    "metadata_bytes": 0,
                    "actual_ratio": 0.0,
                }
        total_bw.details["budget_status"] = {
            "target_ratio": self.target_budget_ratio,
            "actual_ratio": total_bw.actual_ratio,
            "within_budget": total_bw.within_budget,
            "over_budget_by": max(0, total_bw.actual_ratio - self.target_budget_ratio),
            "adjusted_this_step": total_bw.adjusted_this_step,
            "adjustment_count": len(self._adjustment_log),
        }

        self._bandwidth_history.append(total_bw)
        return compressed, total_bw

    def decompress_module_gradients(
        self,
        compressed: Dict[str, Tuple[torch.Tensor, dict]],
    ) -> Dict[str, torch.Tensor]:
        decompressed = {}
        for name, (tensor, meta) in compressed.items():
            decompressed[name] = self.compressor.decompress(tensor, meta)
        return decompressed

    def get_bandwidth_summary(self) -> Dict:
        if not self._bandwidth_history:
            return {}
        last = self._bandwidth_history[-1]
        avg_ratio = sum(h.actual_ratio for h in self._bandwidth_history) / len(self._bandwidth_history)
        min_ratio = min(h.actual_ratio for h in self._bandwidth_history)
        max_ratio = max(h.actual_ratio for h in self._bandwidth_history)
        within_count = sum(1 for h in self._bandwidth_history if h.within_budget)
        adj_count = sum(1 for h in self._bandwidth_history if h.adjusted_this_step)

        final_ratio = None
        final_topk_fraction = None
        if self.method == "topk":
            final_ratio = getattr(self.compressor, "compression_ratio", None)
        elif self.method == "hybrid":
            final_topk_fraction = getattr(self.compressor, "topk_fraction", None)
            if hasattr(self.compressor, 'topk'):
                final_ratio = getattr(self.compressor.topk, "compression_ratio", None)

        return {
            "method": self.method,
            "last": last.__dict__,
            "average_ratio": avg_ratio,
            "min_ratio": min_ratio,
            "max_ratio": max_ratio,
            "within_budget_fraction": within_count / len(self._bandwidth_history),
            "total_steps": len(self._bandwidth_history),
            "target_budget": self.target_budget_ratio,
            "adjustment_count": adj_count,
            "adjustment_log": self._adjustment_log,
            "compression_ratio": getattr(self.compressor, "compression_ratio", None),
            "final_topk_ratio": final_ratio,
            "final_topk_fraction": final_topk_fraction,
            "final_params": self._get_final_compression_params(),
        }

    def _get_final_compression_params(self) -> Dict:
        if self.method == "topk":
            return {
                "type": "topk",
                "compression_ratio": getattr(self.compressor, "compression_ratio", None),
                "min_k": getattr(self.compressor, "min_k", None),
            }
        elif self.method == "1bit":
            return {
                "type": "1bit",
                "sample_ratio": getattr(self.compressor, "sample_ratio", 1.0),
                "fixed_ratio_full_1bit": "~1/32 + 4-byte scale",
                "approx_ratio_32bit": 1/32 + 4/(1024*4),
            }
        elif self.method == "hybrid":
            return {
                "type": "hybrid",
                "topk_compression_ratio": getattr(self.compressor.topk, "compression_ratio", None) if hasattr(self.compressor, 'topk') else None,
                "residual_sample_ratio": getattr(self.compressor.onebit, "sample_ratio", None) if hasattr(self.compressor, 'onebit') else None,
            }
        return {"type": "unknown"}

    def reset_history(self):
        self._bandwidth_history = []
        self._adjustment_log = []
