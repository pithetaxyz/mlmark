"""
Matrix multiplication benchmark — measures TFLOPS for FP32 and FP16.
Tests a range of matrix sizes typical in ML (linear layers, attention).
"""
import torch
import time
import json
import sys

WARMUP = 5
RUNS = 20
SIZES = [512, 1024, 2048, 4096, 8192]


def bench(device, dtype, N):
    a = torch.randn(N, N, device=device, dtype=dtype)
    b = torch.randn(N, N, device=device, dtype=dtype)

    for _ in range(WARMUP):
        torch.matmul(a, b)
    if device != "cpu":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(RUNS):
        torch.matmul(a, b)
    if device != "cpu":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    flops = 2 * N ** 3 * RUNS
    tflops = flops / elapsed / 1e12
    return round(tflops, 3), round(elapsed / RUNS * 1000, 2)


def main():
    results = {"benchmark": "matmul", "device": str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"), "results": []}

    devices = [("cpu", torch.float32, "CPU FP32")]
    if torch.cuda.is_available():
        devices += [
            ("cuda", torch.float32, "GPU FP32"),
            ("cuda", torch.float16, "GPU FP16"),
        ]

    for device, dtype, label in devices:
        for N in SIZES:
            try:
                tflops, ms = bench(device, dtype, N)
                row = {"label": label, "size": N, "tflops": tflops, "ms_per_run": ms}
                results["results"].append(row)
                print(f"  {label:12s}  N={N:5d}  {tflops:7.3f} TFLOPS  {ms:7.2f} ms/run")
            except Exception as e:
                print(f"  {label:12s}  N={N:5d}  ERROR: {e}", file=sys.stderr)

    return results


if __name__ == "__main__":
    print("=== Matrix Multiplication Benchmark ===")
    r = main()
    print(json.dumps(r, indent=2))
