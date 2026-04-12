"""
GPU memory bandwidth benchmark — measures host-to-device, device-to-host,
and device-to-device transfer throughput. Critical for iGPU (shared RAM) systems.
"""
import torch
import time
import json

RUNS = 20
SIZES_MB = [64, 256, 512, 1024, 2048]


def bandwidth(src, dst, size_mb, label):
    n = (size_mb * 1024 * 1024) // 4  # float32 elements
    t = torch.randn(n, dtype=torch.float32, device=src)
    if src != "cpu":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(RUNS):
        t2 = t.to(dst)
        torch.cuda.synchronize() if dst != "cpu" else None
    elapsed = time.perf_counter() - t0

    gb = size_mb / 1024 * RUNS
    gbps = gb / elapsed
    return round(gbps, 2)


def main():
    if not torch.cuda.is_available():
        print("  No GPU available.")
        return {}

    device_name = torch.cuda.get_device_name(0)
    total_vram = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    results = {
        "benchmark": "memory_bandwidth",
        "device": device_name,
        "vram_total_gb": total_vram,
        "results": []
    }
    print(f"  Device: {device_name}  VRAM: {total_vram} GB")

    transfers = [
        ("cpu", "cuda", "H2D (host→GPU)"),
        ("cuda", "cpu",  "D2H (GPU→host)"),
        ("cuda", "cuda", "D2D (GPU→GPU)"),
    ]

    for size_mb in SIZES_MB:
        for src, dst, label in transfers:
            try:
                gbps = bandwidth(src, dst, size_mb, label)
                row = {"label": label, "size_mb": size_mb, "gbps": gbps}
                results["results"].append(row)
                print(f"  {label:20s}  {size_mb:5d} MB  {gbps:7.2f} GB/s")
            except Exception as e:
                print(f"  {label:20s}  {size_mb:5d} MB  ERROR: {e}")

    return results


if __name__ == "__main__":
    print("=== Memory Bandwidth Benchmark ===")
    r = main()
    print(json.dumps(r, indent=2))
