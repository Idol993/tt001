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
    ):
        self.momentum_factor = momentum_factor
        self.warmup_steps = warmup_steps
        self.target_budget_ratio = target_budget_ratio
        self.auto_adjust = auto_adjust
        self._step = 0
        self._error_buffers: Dict[str, torch.Tensor] = {}
        self._momentum_buffers: Dict[str, torch.Tensor] = {}
        self._last_bandwidth: Optional[BandwidthAccount] = None
        self._last_adjustment: Optional[Dict] = None

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
        mean_abs = self._momentum_buffers[key].abs().mean()
        signs_packed = torch.packbits(
            self._momentum_buffers[key].reshape(-1) >= 0
        )
        sign_bits = numel
        sign_bytes = math.ceil(sign_bits / 8)

        compressed = (signs_packed.float() * 2 - 127) / 127 * mean_abs

        reconstructed = torch.zeros_like(self._momentum_buffers[key])
        reconstructed[signs_packed.unpackbits()[:numel].bool()] = mean_abs
        reconstructed[~signs_packed.unpackbits()[:numel].bool()] = -mean_abs
        self._error_buffers[key] = corrected - reconstructed

        bw.data_bytes = sign_bytes
        bw.metadata_bytes = 4
        bw.details = {
            "method": "1bit",
            "sign_bytes": sign_bytes,
            "scale_bytes": 4,
            "numel": numel,
            "mean_abs": mean_abs.item(),
            "actual_ratio": (sign_bytes + 4) / (numel * 4),
        }
        bw.compute()

        self._last_bandwidth = bw
        return compressed, {
            "method": "1bit",
            "original_shape": tensor.shape,
            "signs_packed": signs_packed,
            "mean_abs": mean_abs,
        }, bw

    def decompress(
        self, compressed: torch.Tensor, meta: dict
    ) -> torch.Tensor:
        if meta["method"] == "1bit" and "signs_packed" in meta:
            numel = meta["original_shape"].numel()
            signs = meta["signs_packed"].unpackbits()[:numel].bool()
            mean_abs = meta["mean_abs"]
            reconstructed = torch.zeros(
                meta["original_shape"].numel(),
                device=compressed.device,
                dtype=compressed.dtype,
            )
            reconstructed[signs] = mean_abs
            reconstructed[~signs] = -mean_abs
            return reconstructed.reshape(meta["original_shape"])
        return compressed.reshape(meta["original_shape"])


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
        min_topk_fraction: float = 0.1,
        max_adjustment_factor: float = 0.7,
    ):
        self.target_budget_ratio = target_budget_ratio
        self.warmup_steps = warmup_steps
        self.auto_adjust = auto_adjust
        self.topk_fraction = topk_fraction
        self.min_topk_fraction = min_topk_fraction
        self.max_adjustment_factor = max_adjustment_factor
        self._step = 0
        self._last_bandwidth: Optional[BandwidthAccount] = None
        self._last_adjustment: Optional[Dict] = None
        self._adjustment_count = 0

        self.onebit = OneBitSGDCompressor(
            momentum_factor=momentum_factor,
            warmup_steps=warmup_steps,
            target_budget_ratio=target_budget_ratio * (1 - topk_fraction),
            auto_adjust=auto_adjust,
        )
        self.topk = TopKCompressor(
            compression_ratio=target_budget_ratio * topk_fraction * 0.5,
            target_budget_ratio=target_budget_ratio * topk_fraction,
            auto_adjust=auto_adjust,
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
            "topk_fraction_before": self.topk_fraction,
        }
        bw.compute()

        if self.auto_adjust and not bw.within_budget:
            old_topk_fraction = self.topk_fraction
            old_topk_ratio = self.topk.compression_ratio
            over_by = bw.actual_ratio - self.target_budget_ratio
            reduction_factor = max(
                self.max_adjustment_factor,
                self.target_budget_ratio / max(bw.actual_ratio, 1e-8),
            )

            new_topk_fraction = max(
                self.min_topk_fraction,
                self.topk_fraction * reduction_factor,
            )
            new_topk_ratio = max(
                self.topk.min_k / numel if hasattr(self.topk, 'min_k') else 1e-6,
                self.topk.compression_ratio * reduction_factor,
            )
            self.topk_fraction = new_topk_fraction
            self.topk.compression_ratio = new_topk_ratio
            self._recalculate_budgets()

            self._adjustment_count += 1
            self._last_adjustment = {
                "step": self._step,
                "module": key,
                "reason": f"Hybrid exceeded: {bw.actual_ratio*100:.2f}% > {bw.target_budget_ratio*100:.2f}% (by {over_by*100:.2f}%)",
                "old_topk_fraction": old_topk_fraction,
                "new_topk_fraction": new_topk_fraction,
                "old_topk_ratio": old_topk_ratio,
                "new_topk_ratio": new_topk_ratio,
                "adjustment_count": self._adjustment_count,
            }

            topk_vals, topk_meta, topk_bw = self.topk.compress(tensor, f"{key}_topk_retry")
            topk_reconstructed = self.topk.decompress(topk_vals, topk_meta)
            residual = tensor - topk_reconstructed
            onebit_vals, onebit_meta, onebit_bw = self.onebit.compress(residual, f"{key}_1bit_retry")

            bw = BandwidthAccount(target_budget_ratio=self.target_budget_ratio)
            bw.original_bytes = numel * 4
            bw.data_bytes = topk_bw.data_bytes + onebit_bw.data_bytes
            bw.metadata_bytes = topk_bw.metadata_bytes + onebit_bw.metadata_bytes + 8
            bw.adjusted_this_step = True
            bw.previous_ratio = bw.actual_ratio
            bw.adjustment_details = self._last_adjustment
            bw.details = {
                "method": "hybrid",
                "topk_fraction_before": old_topk_fraction,
                "topk_fraction_after": new_topk_fraction,
                "topk_ratio_before": old_topk_ratio,
                "topk_ratio_after": new_topk_ratio,
                "adjustment_count": self._adjustment_count,
            }
            bw.compute()

        bw.details["topk_fraction_current"] = self.topk_fraction
        bw.details["topk_ratio_current"] = getattr(self.topk, 'compression_ratio', None)

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
            "topk_fraction": self.topk_fraction,
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
            "topk_fraction": self.topk_fraction,
            "topk_compression_ratio": getattr(self.topk, 'compression_ratio', None),
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

        return {
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
        }

    def reset_history(self):
        self._bandwidth_history = []
        self._adjustment_log = []
