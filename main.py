import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from trainer import CrosstalkFreeTrainer, ModularTransformer


def create_synthetic_dataset(
    vocab_size: int = 1000,
    d_model: int = 128,
    seq_len: int = 32,
    num_samples: int = 500,
    num_classes: int = 10,
):
    inputs = torch.randint(0, vocab_size, (num_samples, seq_len))
    targets = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(inputs, targets)


def main():
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = ModularTransformer(
        vocab_size=1000,
        d_model=128,
        n_heads=4,
        d_ff=256,
        n_layers=2,
        num_classes=10,
        seq_len=32,
    )

    module_names = model.module_names
    print(f"Modules: {module_names}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    trainer = CrosstalkFreeTrainer(
        model=model,
        module_names=module_names,
        lr=1e-3,
        n_fft=64,
        hop_length=16,
        num_filters_per_module=6,
        compression_method="topk",
        bandwidth_budget=0.01,
        device=device,
    )

    dataset = create_synthetic_dataset(num_samples=500)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()

    num_epochs = 3
    print(f"\nStarting crosstalk-free training for {num_epochs} epochs...")
    print(f"Communication bandwidth budget: <1%")
    print("=" * 80)

    all_logs = []
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        logs = trainer.train_epoch(dataloader, loss_fn, epoch=epoch)
        all_logs.extend(logs)

        if logs:
            epoch_loss = sum(l["loss"] for l in logs) / len(logs)
            epoch_comp = sum(l["compression_ratio"] for l in logs) / len(logs)
            epoch_dist = sum(l["avg_filter_distortion"] for l in logs) / len(logs)
            bounded_count = sum(1 for l in logs if l["filter_bounded"])
            sufficient_count = sum(1 for l in logs if l["sufficient_decrease"])
            print(f"\n  Epoch Summary:")
            print(f"    Avg Loss: {epoch_loss:.4f}")
            print(f"    Avg Compression Ratio: {epoch_comp:.4f}")
            print(f"    Avg Filter Distortion: {epoch_dist:.4f}")
            print(f"    Filter Bounded: {bounded_count}/{len(logs)} steps")
            print(f"    Sufficient Decrease: {sufficient_count}/{len(logs)} steps")

    print("\n" + "=" * 80)
    print("Training complete. Filter diagnostics:")
    diagnostics = trainer.get_filter_diagnostics()
    for mod_name, info in diagnostics.items():
        centers = [f"{c:.1f}" for c in info["center_frequencies"]]
        bws = [f"{b:.2f}" for b in info["bandwidths"]]
        print(f"  {mod_name}: centers={centers}, bandwidths={bws}")

    lyapunov_history = trainer.lyapunov.get_history()
    if lyapunov_history:
        final = lyapunov_history[-1]
        print(f"\nFinal Lyapunov Analysis:")
        print(f"  Lyapunov value: {final['lyapunov_value']:.4f}")
        print(f"  Filter distortion: {final['filter_distortion']:.4f}")
        print(f"  Compression distortion: {final['compression_distortion']:.4f}")
        print(f"  Sufficient decrease: {final['sufficient_decrease']}")


if __name__ == "__main__":
    main()
