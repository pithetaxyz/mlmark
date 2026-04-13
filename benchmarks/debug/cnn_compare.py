"""
ResNet-50 inference benchmark — comparison across optimization modes.
Run with: python bench_compare.py
"""

import torch
import time
import torchvision.models as models

BATCHES = {"small": 32, "medium": 128, "large": 512}
WARMUP  = 50
RUNS    = 100
DTYPE   = torch.float16
DEVICE  = "cuda"

def make_model():
    return models.resnet50(weights=None).to(device=DEVICE, dtype=DTYPE).eval()

def timed(fn, batch, warmup=WARMUP, runs=RUNS):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            fn()
    torch.cuda.synchronize()
    return round(batch * runs / (time.perf_counter() - t), 1)

def sep(char="-", width=72):
    print(char * width)

MODES = ["Baseline", "cudnn.benchmark", "torch.compile (eager)", "CUDA Graph"]

print("=" * 72)
print(f"  ResNet-50 Inference — {torch.cuda.get_device_name(0)}")
print(f"  PyTorch {torch.__version__}  |  dtype: {DTYPE}")
print("=" * 72)
print(f"  {'Mode':<28} {'small (32)':>12} {'medium (128)':>14} {'large (512)':>13}")
sep()

all_results = {m: {} for m in MODES}

for tier, batch in BATCHES.items():
    x_shape = (batch, 3, 224, 224)

    # 1. Baseline
    torch.backends.cudnn.benchmark = False
    m = make_model()
    x = torch.randn(*x_shape, device=DEVICE, dtype=DTYPE)
    all_results["Baseline"][tier] = timed(lambda: m(x), batch)

    # 2. cudnn.benchmark
    torch.backends.cudnn.benchmark = True
    m = make_model()
    x = torch.randn(*x_shape, device=DEVICE, dtype=DTYPE)
    all_results["cudnn.benchmark"][tier] = timed(lambda: m(x), batch)

    # 3. torch.compile (eager — Triton unavailable on Windows)
    torch.backends.cudnn.benchmark = True
    m = make_model()
    compiled = torch.compile(m, backend="eager")
    x = torch.randn(*x_shape, device=DEVICE, dtype=DTYPE)
    all_results["torch.compile (eager)"][tier] = timed(lambda: compiled(x), batch)

    # 4. CUDA Graph
    torch.backends.cudnn.benchmark = True
    m = make_model()
    static_x = torch.randn(*x_shape, device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        for _ in range(WARMUP):
            m(static_x)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with torch.no_grad():
            _ = m(static_x)
    torch.cuda.synchronize()

    t = time.perf_counter()
    for _ in range(RUNS):
        g.replay()
    torch.cuda.synchronize()
    all_results["CUDA Graph"][tier] = round(batch * RUNS / (time.perf_counter() - t), 1)

# Print results table
for mode in MODES:
    r = all_results[mode]
    print(f"  {mode:<28} {r['small']:>12.1f} {r['medium']:>14.1f} {r['large']:>13.1f}")

# Print speedup table vs baseline
sep()
print(f"\n  Speedup vs Baseline")
sep()
print(f"  {'Mode':<28} {'small (32)':>12} {'medium (128)':>14} {'large (512)':>13}")
sep()
for mode in MODES[1:]:
    row = []
    for tier in BATCHES:
        base = all_results["Baseline"][tier]
        val  = all_results[mode][tier]
        row.append(f"{val/base:.2f}x")
    print(f"  {mode:<28} {row[0]:>12} {row[1]:>14} {row[2]:>13}")

print("=" * 72)