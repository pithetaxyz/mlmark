"""
Transformer attention benchmark — GPT-2-scale synthetic forward pass.
"""
import torch
import torch.nn as nn
import time
import json
import math

WARMUP = 5
RUNS = 20
BATCH_SIZE = 1
D_MODEL, N_HEADS, N_LAYERS, D_FF = 768, 12, 12, 3072

SIZE_TIERS = {"small": 128, "medium": 512, "large": 2048}


class _MHA(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_head = D_MODEL // N_HEADS
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.out = nn.Linear(D_MODEL, D_MODEL)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, N_HEADS, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.d_head), dim=-1)
        return self.out((attn @ v).transpose(1, 2).reshape(B, T, C))


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = _MHA()
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.ff = nn.Sequential(nn.Linear(D_MODEL, D_FF), nn.GELU(), nn.Linear(D_FF, D_MODEL))

    def forward(self, x):
        return x + self.ff(self.ln2(x + self.attn(self.ln1(x))))


def _make_model():
    return nn.Sequential(*[_Block() for _ in range(N_LAYERS)])


def run_one(tier: str, device: str, dtype: torch.dtype) -> dict:
    seq_len = SIZE_TIERS[tier]
    model = _make_model().to(device=device, dtype=dtype).eval()
    x = torch.randn(BATCH_SIZE, seq_len, D_MODEL, device=device, dtype=dtype)

    with torch.no_grad():
        for _ in range(WARMUP):
            model(x)
        if device != "cpu":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(RUNS):
            model(x)
        if device != "cpu":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    ms = elapsed / RUNS * 1000
    return {
        "benchmark": "transformer_gpt2",
        "tier": tier,
        "seq_len": seq_len,
        "device": device,
        "dtype": str(dtype),
        "ms_per_run": round(ms, 2),
        "tokens_per_sec": round(seq_len * RUNS / elapsed, 1),
    }


if __name__ == "__main__":
    results = []
    params_m = sum(p.numel() for p in _make_model().parameters()) / 1e6
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"=== Transformer Benchmark — GPT-2 scale ({params_m:.0f}M params)  ({device_name}) ===")
    configs = [("cpu", torch.float32, "CPU FP32")]
    if torch.cuda.is_available():
        configs += [("cuda", torch.float32, "GPU FP32"), ("cuda", torch.float16, "GPU FP16")]
    for tier in SIZE_TIERS:
        for device, dtype, label in configs:
            try:
                r = run_one(tier, device, dtype)
                print(f"  {label:12s}  {tier:8s}  seq={r['seq_len']:5d}  {r['ms_per_run']:8.2f} ms  {r['tokens_per_sec']:8.1f} tok/s")
                results.append(r)
            except Exception as e:
                print(f"  {label:12s}  {tier:8s}  ERROR: {e}")
    print(json.dumps(results, indent=2))
