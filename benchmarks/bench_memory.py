"""
GPU memory bandwidth benchmark — H2D, D2H, D2D transfer throughput.
"""
import torch
import time
import json

RUNS = 20

SIZE_TIERS = {"small": 64, "medium": 512, "large": 2048}  # MB


def run_one(tier: str, device: str, dtype: torch.dtype) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available")

    bytes_per_elem = {
        torch.float32: 4, torch.float16: 2, torch.bfloat16: 2,
    }.get(dtype, getattr(dtype, "itemsize", 1))

    size_mb = SIZE_TIERS[tier]
    n = (size_mb * 1024 * 1024) // bytes_per_elem

    results = {"benchmark": "memory_bandwidth", "tier": tier, "size_mb": size_mb,
               "dtype": str(dtype), "transfers": {}}

    # For dtypes that can't be created on CPU (e.g. FP8), allocate on GPU and use
    # view-as-uint8 for the CPU-side buffer so H2D/D2H measure the right byte count.
    cpu_friendly = dtype in (torch.float32, torch.float16, torch.bfloat16)

    transfers = [("cpu", "cuda", "H2D"), ("cuda", "cpu", "D2H"), ("cuda", "cuda", "D2D")]
    for src, dst, label in transfers:
        if src == "cpu":
            if cpu_friendly:
                t = torch.randn(n, dtype=dtype, device="cpu")
            else:
                # same byte count as the target dtype
                t = torch.randint(0, 256, (n,), dtype=torch.uint8, device="cpu")
        else:
            # allocate in fp32 on GPU then cast — works for fp8 since GPU supports the view
            t = torch.randn(n, device="cuda").to(dtype)
        if src != "cpu":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(RUNS):
            t2 = t.to(dst)
            if dst != "cpu":
                torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        gbps = (size_mb / 1024 * RUNS) / elapsed
        results["transfers"][label] = round(gbps, 2)

    return results


if __name__ == "__main__":
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"=== Memory Bandwidth Benchmark  ({device_name}) ===")
    all_results = []
    for tier in SIZE_TIERS:
        try:
            r = run_one(tier, "cuda", torch.float32)
            for label, gbps in r["transfers"].items():
                print(f"  {tier:8s}  {label:4s}  {r['size_mb']:5d} MB  {gbps:7.2f} GB/s")
            all_results.append(r)
        except Exception as e:
            print(f"  {tier:8s}  ERROR: {e}")
    print(json.dumps(all_results, indent=2))
