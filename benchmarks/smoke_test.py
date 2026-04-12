"""
Quick smoke test — one pass through all benchmarks with tiny sizes.
Purpose: verify GPU + CPU paths work before the full run.
"""
import torch
import torch.nn as nn
import time
import math

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPES = [("FP32", torch.float32)]
if torch.cuda.is_available():
    DTYPES.append(("FP16", torch.float16))

ok = 0
fail = 0

def check(label, fn):
    global ok, fail
    try:
        fn()
        print(f"  [OK]   {label}")
        ok += 1
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        fail += 1

print(f"\n=== Smoke Test  (device: {DEVICE}) ===\n")

# --- matmul ---
for tag, dtype in DTYPES:
    def _matmul(dtype=dtype):
        a = torch.randn(256, 256, device=DEVICE, dtype=dtype)
        torch.matmul(a, a)
        if DEVICE != "cpu": torch.cuda.synchronize()
    check(f"matmul GPU {tag}", _matmul)

def _matmul_cpu():
    a = torch.randn(256, 256)
    torch.matmul(a, a)
check("matmul CPU FP32", _matmul_cpu)

# --- CNN (single conv layer) ---
for tag, dtype in DTYPES:
    def _cnn(dtype=dtype):
        m = nn.Conv2d(3, 16, 3, padding=1).to(device=DEVICE, dtype=dtype).eval()
        x = torch.randn(1, 3, 64, 64, device=DEVICE, dtype=dtype)
        with torch.no_grad(): m(x)
        if DEVICE != "cpu": torch.cuda.synchronize()
    check(f"CNN GPU {tag}", _cnn)

def _cnn_cpu():
    m = nn.Conv2d(3, 16, 3, padding=1).eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad(): m(x)
check("CNN CPU FP32", _cnn_cpu)

# --- transformer (2-layer, tiny) ---
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(64)
        self.ff = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64))
    def forward(self, x): return x + self.ff(self.ln(x))

for tag, dtype in DTYPES:
    def _tfm(dtype=dtype):
        m = nn.Sequential(Block(), Block()).to(device=DEVICE, dtype=dtype).eval()
        x = torch.randn(1, 32, 64, device=DEVICE, dtype=dtype)
        with torch.no_grad(): m(x)
        if DEVICE != "cpu": torch.cuda.synchronize()
    check(f"transformer GPU {tag}", _tfm)

def _tfm_cpu():
    m = nn.Sequential(Block(), Block()).eval()
    x = torch.randn(1, 32, 64)
    with torch.no_grad(): m(x)
check("transformer CPU FP32", _tfm_cpu)

# --- memory bandwidth (small) ---
if torch.cuda.is_available():
    def _h2d():
        t = torch.randn(1024*1024, device="cpu")  # 4 MB
        t.to("cuda"); torch.cuda.synchronize()
    check("memory H2D", _h2d)

    def _d2h():
        t = torch.randn(1024*1024, device="cuda")
        t.to("cpu"); torch.cuda.synchronize()
    check("memory D2H", _d2h)

    def _d2d():
        t = torch.randn(1024*1024, device="cuda")
        t.clone(); torch.cuda.synchronize()
    check("memory D2D", _d2d)

print(f"\nResult: {ok} passed, {fail} failed\n")
