"""
CNN inference benchmark — ResNet-50 images/sec throughput.
"""
import torch
import torch.nn as nn
import time
import json

WARMUP = 10
RUNS = 50
INPUT_SIZE = (3, 224, 224)

SIZE_TIERS = {"small": 1, "medium": 8, "large": 32}


def _make_model():
    try:
        import torchvision.models as models
        return models.resnet50(weights=None)
    except ImportError:
        return nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(64, 1000)
        )


def run_one(tier: str, device: str, dtype: torch.dtype) -> dict:
    batch_size = SIZE_TIERS[tier]
    model = _make_model().to(device=device, dtype=dtype).eval()
    x = torch.randn(batch_size, *INPUT_SIZE, device=device, dtype=dtype)

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

    return {
        "benchmark": "cnn_resnet50",
        "tier": tier,
        "batch_size": batch_size,
        "device": device,
        "dtype": str(dtype),
        "images_per_sec": round(batch_size * RUNS / elapsed, 1),
        "ms_per_batch": round(elapsed / RUNS * 1000, 2),
    }


if __name__ == "__main__":
    results = []
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"=== CNN Inference Benchmark — ResNet-50  ({device_name}) ===")
    configs = [("cpu", torch.float32, "CPU FP32")]
    if torch.cuda.is_available():
        configs += [("cuda", torch.float32, "GPU FP32"), ("cuda", torch.float16, "GPU FP16")]
    for tier in SIZE_TIERS:
        for device, dtype, label in configs:
            try:
                r = run_one(tier, device, dtype)
                print(f"  {label:12s}  {tier:8s}  batch={r['batch_size']:2d}  {r['images_per_sec']:8.1f} img/s  {r['ms_per_batch']:8.2f} ms")
                results.append(r)
            except Exception as e:
                print(f"  {label:12s}  {tier:8s}  ERROR: {e}")
    print(json.dumps(results, indent=2))
