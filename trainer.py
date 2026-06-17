import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

from gradient_isolator import GradientIsolator
from compression import GradientCompressor
from lyapunov import LyapunovAnalyzer


class ModularTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        n_layers: int = 2,
        num_classes: int = 10,
        seq_len: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.module_names = ["embedding", "attention", "feedforward", "output"]

        self.embedding = nn.Sequential(
            nn.Embedding(vocab_size, d_model),
            nn.LayerNorm(d_model),
        )

        attention_layers = []
        for i in range(n_layers):
            attention_layers.append((
                f"attn_{i}",
                nn.MultiheadAttention(d_model, n_heads, batch_first=True),
            ))
            attention_layers.append((f"attn_norm_{i}", nn.LayerNorm(d_model)))
        self.attention = nn.Sequential(OrderedDict(attention_layers))

        ff_layers = []
        for i in range(n_layers):
            ff_layers.append((f"ff_{i}_linear1", nn.Linear(d_model, d_ff)))
            ff_layers.append((f"ff_{i}_act", nn.GELU()))
            ff_layers.append((f"ff_{i}_linear2", nn.Linear(d_ff, d_model)))
            ff_layers.append((f"ff_{i}_norm", nn.LayerNorm(d_model)))
        self.feedforward = nn.Sequential(OrderedDict(ff_layers))

        self.output = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            emb = self.embedding[0](x)
            emb = self.embedding[1](emb)
        else:
            emb = x

        attn_out = emb
        for name, layer in self.attention.named_children():
            if "attn_" in name and "norm" not in name:
                residual = attn_out
                attn_out, _ = layer(attn_out, attn_out, attn_out)
                attn_out = attn_out + residual
            else:
                attn_out = layer(attn_out)

        ff_out = attn_out
        for name, layer in self.feedforward.named_children():
            if "linear1" in name:
                residual = ff_out
                ff_out = layer(ff_out)
            elif "linear2" in name:
                ff_out = layer(ff_out)
                ff_out = ff_out + residual
            else:
                ff_out = layer(ff_out)

        pooled = ff_out.transpose(1, 2)
        out = self.output(pooled)
        return out


class CrosstalkFreeTrainer:
    def __init__(
        self,
        model: nn.Module,
        module_names: List[str],
        lr: float = 1e-3,
        n_fft: int = 64,
        hop_length: int = 16,
        num_filters_per_module: int = 6,
        compression_method: str = "topk",
        bandwidth_budget: float = 0.01,
        lyapunov_lr: Optional[float] = None,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.module_names = module_names
        self.device = device

        self.isolator = GradientIsolator(
            n_fft=n_fft,
            hop_length=hop_length,
            num_modules=len(module_names),
            num_filters_per_module=num_filters_per_module,
        ).to(device)

        self.compressor = GradientCompressor(
            method=compression_method,
            bandwidth_budget=bandwidth_budget,
        )

        self.lyapunov = LyapunovAnalyzer(
            lr=lr,
            filter_bound=1.0,
            compression_error_bound=0.1,
        )

        model_params = list(model.parameters())
        filter_params = list(self.isolator.filter_bank.parameters())
        all_params = model_params + filter_params

        self.optimizer = torch.optim.AdamW(all_params, lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=lr * 0.01
        )

        self._step = 0
        self._log_interval = 10

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: nn.Module,
    ) -> Dict:
        self.model.train()
        self._step += 1

        self.optimizer.zero_grad()
        output = self.model(inputs)
        loss = loss_fn(output, targets)
        loss.backward()

        with torch.no_grad():
            original_grads = self.isolator.collect_module_gradients(
                self.model, self.module_names
            )

        isolated_grads, overlap_map = self.isolator(
            original_grads, adapt_filters=(self._step % 50 == 1)
        )

        compressed_data, compression_ratio = self.compressor.compress_module_gradients(
            isolated_grads
        )
        decompressed_grads = self.compressor.decompress_module_gradients(compressed_data)

        self.isolator.redistribute_gradients(
            self.model, decompressed_grads, self.module_names
        )

        reg_loss = self.isolator.get_regularization_loss()
        total_loss = loss + reg_loss

        self.optimizer.step()
        self.scheduler.step()

        proof = self.lyapunov.prove_approximate_invariance(
            filter_bank=self.isolator.filter_bank,
            original_grads=original_grads,
            filtered_grads=isolated_grads,
            compressed_grads=decompressed_grads,
            step=self._step,
            prev_loss=loss.item(),
            curr_loss=total_loss.item(),
        )

        log = {
            "step": self._step,
            "loss": loss.item(),
            "total_loss": total_loss.item(),
            "reg_loss": reg_loss.item(),
            "compression_ratio": compression_ratio,
            "overlap_map": {k: len(v) for k, v in overlap_map.items()},
            "convergence_guarantee": proof["convergence_guarantee"],
            "filter_bounded": proof["filter_bounded"],
            "sufficient_decrease": proof["sufficient_decrease"],
            "avg_filter_distortion": proof["avg_filter_distortion"],
            "contraction_rate": proof["contraction_rate"],
        }

        return log

    def train_epoch(
        self,
        dataloader,
        loss_fn: nn.Module,
        epoch: int = 0,
    ) -> List[Dict]:
        logs = []
        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            else:
                inputs, targets = batch.to(self.device), batch.to(self.device)

            log = self.train_step(inputs, targets, loss_fn)
            logs.append(log)

            if (batch_idx + 1) % self._log_interval == 0:
                print(
                    f"Epoch {epoch} | Step {log['step']} | "
                    f"Loss: {log['loss']:.4f} | "
                    f"Compression: {log['compression_ratio']:.4f} | "
                    f"Filter distortion: {log['avg_filter_distortion']:.4f} | "
                    f"Contraction: {log['contraction_rate']:.4f} | "
                    f"Guarantee: {log['convergence_guarantee'][:60]}..."
                )
        return logs

    def get_filter_diagnostics(self) -> Dict:
        params = self.isolator.filter_bank.get_filter_parameters()
        diagnostics = {}
        for p in params:
            idx = p["module_idx"]
            name = self.module_names[idx] if idx < len(self.module_names) else f"module_{idx}"
            diagnostics[name] = {
                "center_frequencies": p["center_frequencies"].tolist(),
                "bandwidths": p["bandwidths"].tolist(),
            }
        return diagnostics
