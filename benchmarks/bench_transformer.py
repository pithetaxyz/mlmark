"""
Transformer attention benchmark — synthetic forward pass mimicking LLM inference.
Tests seq lengths 128, 512, 2048 with a GPT-2-scale config (12 layers, 768 hidden, 12 heads).
"""
import torch
import torch.nn as nn
import time
import json
import math
import sys

WARMUP = 5
RUNS = 20

# GPT-2 small scale
N_LAYERS = 12
D_MODEL = 768
N_HEADS = 12
D_FF = 3072
SEQ_LENGTHS = [128, 512, 2048]
BATCH_SIZE = 1


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scale = math.sqrt(self.d_head)
        attn = torch.softmax(q @ k.transpose(-2, -1) / scale, dim=-1)
        return self.out((attn @ v).transpose(1, 2).reshape(B, T, C))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def bench(model, device, dtype, seq_len):
    model = model.to(device=device, dtype=dtype).eval()
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
    tok_per_sec = seq_len * RUNS / elapsed
    return round(ms, 2), round(tok_per_sec, 1)


def main():
    model = MiniGPT(N_LAYERS, D_MODEL, N_HEADS, D_FF)
    param_m = sum(p.numel() for p in model.parameters()) / 1e6
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    results = {
        "benchmark": "transformer_gpt2_scale",
        "device": device_name,
        "model_params_M": round(param_m, 1),
        "results": []
    }
    print(f"  Model: {param_m:.1f}M params  ({N_LAYERS} layers, d={D_MODEL}, heads={N_HEADS})")

    configs = [("cpu", torch.float32, "CPU FP32")]
    if torch.cuda.is_available():
        configs += [
            ("cuda", torch.float32, "GPU FP32"),
            ("cuda", torch.float16, "GPU FP16"),
        ]

    for device, dtype, label in configs:
        for seq_len in SEQ_LENGTHS:
            try:
                ms, tps = bench(model, device, dtype, seq_len)
                row = {"label": label, "seq_len": seq_len, "ms_per_run": ms, "tokens_per_sec": tps}
                results["results"].append(row)
                print(f"  {label:12s}  seq={seq_len:5d}  {ms:8.2f} ms/fwd  {tps:8.1f} tok/s")
            except Exception as e:
                print(f"  {label:12s}  seq={seq_len:5d}  ERROR: {e}", file=sys.stderr)

    return results


if __name__ == "__main__":
    print("=== Transformer Inference Benchmark (GPT-2 scale) ===")
    r = main()
    print(json.dumps(r, indent=2))
