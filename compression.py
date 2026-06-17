import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import math


class OneBitSGDCompressor:
    def __init__(self, momentum_factor: float = 0.9, warmup_steps: int = 100):
        self.momentum_factor = momentum_factor
        self.warmup_steps = warmup_steps
        self._step = 0
        self._error_buffers: Dict[str, torch.Tensor] = {}
        self._momentum_buffers: Dict[str, torch.Tensor] = {}

    def compress(self, tensor: torch.Tensor, key: str) -> Tuple[torch.Tensor, dict]:
        self._step += 1

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
            return msg, {"method": "full", "original_shape": tensor.shape}

        mean_abs = self._momentum_buffers[key].abs().mean()
        signs = (self._momentum_buffers[key] >= 0).float()
        compressed = signs * 2 - 1
        compressed = compressed * mean_abs

        self._error_buffers[key] = corrected - compressed
        compressed_bytes = signs.numel() / 8
        original_bytes = tensor.numel() * 4
        compression_ratio = compressed_bytes / original_bytes

        return compressed, {
            "method": "1bit",
            "original_shape": tensor.shape,
            "compression_ratio": compression_ratio,
            "mean_abs": mean_abs.item(),
        }

    def decompress(
        self, compressed: torch.Tensor, meta: dict
    ) -> torch.Tensor:
        return compressed.reshape(meta["original_shape"])


class TopKCompressor:
    def __init__(self, compression_ratio: float = 0.01):
        self.compression_ratio = compression_ratio

    def compress(self, tensor: torch.Tensor, key: str) -> Tuple[torch.Tensor, dict]:
        flat = tensor.reshape(-1)
        k = max(1, int(flat.numel() * self.compression_ratio))

        topk_vals, topk_indices = torch.topk(flat.abs(), k)
        values = flat[topk_indices]

        compression_ratio = (k * (4 + 4)) / (flat.numel() * 4)
        return values, {
            "method": "topk",
            "indices": topk_indices,
            "original_shape": tensor.shape,
            "original_numel": flat.numel(),
            "compression_ratio": compression_ratio,
        }

    def decompress(self, values: torch.Tensor, meta: dict) -> torch.Tensor:
        reconstructed = torch.zeros(
            meta["original_numel"],
            device=values.device,
            dtype=values.dtype,
        )
        reconstructed[meta["indices"]] = values
        return reconstructed.reshape(meta["original_shape"])


class HybridCompressor:
    def __init__(
        self,
        bandwidth_budget: float = 0.01,
        warmup_steps: int = 100,
        momentum_factor: float = 0.9,
    ):
        self.bandwidth_budget = bandwidth_budget
        self.warmup_steps = warmup_steps
        self.onebit = OneBitSGDCompressor(
            momentum_factor=momentum_factor,
            warmup_steps=warmup_steps,
        )
        self.topk = TopKCompressor(compression_ratio=budget_to_topk_ratio(bandwidth_budget))

    def compress(self, tensor: torch.Tensor, key: str) -> Tuple[torch.Tensor, dict]:
        numel = tensor.numel()
        approx_topk_ratio = self.bandwidth_budget * 0.5
        k = max(1, int(numel * approx_topk_ratio))
        topk_bytes = k * 8
        onebit_bytes = numel / 8
        total_bytes = topk_bytes + onebit_bytes
        budget_bytes = numel * 4 * self.bandwidth_budget

        if total_bytes <= budget_bytes:
            topk_vals, topk_meta = self.topk.compress(tensor, f"{key}_topk")
            onebit_vals, onebit_meta = self.onebit.compress(tensor, f"{key}_1bit")
            meta = {
                "method": "hybrid",
                "topk_meta": topk_meta,
                "onebit_meta": onebit_meta,
                "original_shape": tensor.shape,
            }
            combined = torch.cat([topk_vals, onebit_vals.reshape(-1)[:k]])
            return combined, meta
        else:
            return self.onebit.compress(tensor, key)

    def decompress(self, compressed: torch.Tensor, meta: dict) -> torch.Tensor:
        if meta["method"] == "hybrid":
            topk_reconstructed = self.topk.decompress(
                compressed[: meta["topk_meta"]["indices"].numel()],
                meta["topk_meta"],
            )
            onebit_reconstructed = self.onebit.decompress(compressed, meta["onebit_meta"])
            return (topk_reconstructed + onebit_reconstructed) / 2.0
        else:
            return self.onebit.decompress(compressed, meta)


def budget_to_topk_ratio(bandwidth_budget: float) -> float:
    return max(0.001, bandwidth_budget * 0.5)


class GradientCompressor:
    def __init__(
        self,
        method: str = "topk",
        bandwidth_budget: float = 0.01,
        warmup_steps: int = 100,
        momentum_factor: float = 0.9,
    ):
        self.method = method
        if method == "1bit":
            self.compressor = OneBitSGDCompressor(
                momentum_factor=momentum_factor,
                warmup_steps=warmup_steps,
            )
        elif method == "topk":
            self.compressor = TopKCompressor(
                compression_ratio=budget_to_topk_ratio(bandwidth_budget)
            )
        elif method == "hybrid":
            self.compressor = HybridCompressor(
                bandwidth_budget=bandwidth_budget,
                warmup_steps=warmup_steps,
                momentum_factor=momentum_factor,
            )
        else:
            raise ValueError(f"Unknown compression method: {method}")

    def compress_module_gradients(
        self,
        isolated_grads: Dict[str, torch.Tensor],
    ) -> Dict[str, Tuple[torch.Tensor, dict]]:
        compressed = {}
        total_bytes = 0
        total_original = 0
        for name, grad in isolated_grads.items():
            compressed_tensor, meta = self.compressor.compress(grad, name)
            compressed[name] = (compressed_tensor, meta)
            total_bytes += compressed_tensor.numel() * 4
            total_original += grad.numel() * 4
        actual_ratio = total_bytes / (total_original + 1e-8)
        return compressed, actual_ratio

    def decompress_module_gradients(
        self,
        compressed: Dict[str, Tuple[torch.Tensor, dict]],
    ) -> Dict[str, torch.Tensor]:
        decompressed = {}
        for name, (tensor, meta) in compressed.items():
            decompressed[name] = self.compressor.decompress(tensor, meta)
        return decompressed
