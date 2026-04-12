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
    if a.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) if hasattr(torch, "float8_e4m3fn") else ():
        scale = torch.tensor(1.0, device=device)
        return torch._scaled_mm(a, b.t(), scale_a=scale, scale_b=scale,
                                out_dtype=torch.float16)
    return torch.matmul(a, b)


def run_one(tier: str, device: str, dtype: torch.dtype) -> dict:
    N = SIZE_TIERS[tier]
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
    for tier in SIZE_TIERS:
        for device, dtype, label in configs:
            try:
                r = run_one(tier, device, dtype)
                print(f"  {label:12s}  {tier:8s}  N={r['size']:5d}  {r['tflops']:7.3f} TFLOPS  {r['ms_per_run']:8.2f} ms")
                results.append(r)
            except Exception as e:
                print(f"  {label:12s}  {tier:8s}  ERROR: {e}")
    print(json.dumps(results, indent=2))
