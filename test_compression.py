import torch
import sys
sys.path.insert(0, '.')
from compression import GradientCompressor

test_tensor = torch.randn(10000)  # 40000 bytes
target = 0.005  # 0.5% budget = 200 bytes

for method in ["topk", "1bit", "hybrid"]:
    print(f"\n=== Testing {method} compression ===")
    comp = GradientCompressor(method=method, target_budget_ratio=target, auto_adjust=True)
    grads = {"test": test_tensor}

    for step in range(5):
        compressed, bw = comp.compress_module_gradients(grads)
        adj_str = " [ADJUSTED]" if bw.adjusted_this_step else ""
        print(f"  Step {step+1}: actual={bw.actual_ratio*100:.3f}%, target={target*100:.2f}%, "
              f"within={bw.within_budget}{adj_str}")
        if bw.adjusted_this_step and bw.adjustment_details:
            det = bw.adjustment_details
            subs = det.get("sub_adjustments", [det])
            for s in subs:
                if isinstance(s, dict):
                    print(f"    Reason: {s.get('reason', '')}")
                    if "previous_ratio" in s and "adjusted_ratio" in s:
                        print(f"    ratio: {s['previous_ratio']*100:.2f}% -> {s['adjusted_ratio']*100:.2f}%")
                    if "old_ratio" in s and "new_ratio" in s:
                        print(f"    topk_ratio: {s['old_ratio']*100:.4f}% -> {s['new_ratio']*100:.4f}%")
                    if "old_topk_ratio" in s and "new_topk_ratio" in s:
                        print(f"    topk_ratio: {s['old_topk_ratio']*100:.4f}% -> {s['new_topk_ratio']*100:.4f}%")
                    if "old_residual_ratio" in s and "new_residual_ratio" in s:
                        print(f"    residual_sample: {s['old_residual_ratio']*100:.2f}% -> {s['new_residual_ratio']*100:.2f}%")
                    if "old_topk_fraction" in s and "new_topk_fraction" in s:
                        print(f"    topk_frac: {s['old_topk_fraction']*100:.2f}% -> {s['new_topk_fraction']*100:.2f}%")

    summary = comp.get_bandwidth_summary()
    print(f"\n  Summary:")
    print(f"    avg={summary['average_ratio']*100:.3f}%, max={summary['max_ratio']*100:.3f}%")
    print(f"    adjustments={summary['adjustment_count']}")
    print(f"    final_params: {summary.get('final_params', {})}")
