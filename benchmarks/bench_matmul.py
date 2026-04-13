"""
Matrix multiplication benchmark — TFLOPS for FP32 and FP16.
"""
import torch
import time
import json

WARMUP = 5
RUNS = 20

SIZE_TIERS = {"small": 512, "medium": 2048, "large": 8192}


def _matmul(a, b, device):
    """Dispatch to the right matmul op depending on dtype."""
    if hasattr(torch, "float8_e4m3fn") and a.dtype == torch.float8_e4m3fn:
        scale = torch.tensor(1.0, device=device)
        return torch._scaled_mm(a, b.t(), scale_a=scale, scale_b=scale,
                                out_dtype=torch.float16)
    elif hasattr(torch, "float4_e2m1fn_x2") and a.dtype == torch.float4_e2m1fn_x2:
        return torch._int_mm(a.view(torch.int8), b.view(torch.int8).t())
    return torch.matmul(a, b)


def run_one(tier: str, device: str, dtype: torch.dtype) -> dict:
    N = SIZE_TIERS[tier]
    if dtype == getattr(torch, "float4_e2m1fn_x2", None):
        a = torch.randint(0, 256, (N, N // 2), device=device, dtype=torch.uint8).view(dtype)
        b = torch.randint(0, 256, (N, N // 2), device=device, dtype=torch.uint8).view(dtype)
    else:
        a = torch.randn(N, N, device=device).to(dtype)
        b = torch.randn(N, N, device=device).to(dtype)

    for _ in range(WARMUP):
        _matmul(a, b, device)
    if device != "cpu":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(RUNS):
        _matmul(a, b, device)
    if device != "cpu":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tflops = 2 * N ** 3 * RUNS / elapsed / 1e12
    return {
        "benchmark": "matmul",
        "tier": tier,
        "size": N,
        "device": device,
        "dtype": str(dtype),
        "tflops": round(tflops, 3),
        "ms_per_run": round(elapsed / RUNS * 1000, 2),
    }


if __name__ == "__main__":
    results = []
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"=== Matrix Multiplication Benchmark  ({device_name}) ===")
    configs = [("cpu", torch.float32, "CPU FP32")]
    if torch.cuda.is_available():
        configs += [("cuda", torch.float32, "GPU FP32"), ("cuda", torch.float16, "GPU FP16")]
        if hasattr(torch, "float8_e4m3fn"):
            configs += [("cuda", torch.float8_e4m3fn, "GPU FP8")]
        if hasattr(torch, "float4_e2m1fn_x2"):
            configs += [("cuda", torch.float4_e2m1fn_x2, "GPU FP4")]

    for tier in SIZE_TIERS:
        for device, dtype, label in configs:
            try:
                r = run_one(tier, device, dtype)
                print(f"  {label:12s}  {tier:8s}  N={r['size']:5d}  {r['tflops']:7.3f} TFLOPS  {r['ms_per_run']:8.2f} ms")
                results.append(r)
            except Exception as e:
                print(f"  {label:12s}  {tier:8s}  ERROR: {e}")
    print(json.dumps(results, indent=2))
