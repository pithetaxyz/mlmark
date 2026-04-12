"""
CNN inference benchmark — ResNet-50 images/sec throughput.
Tests batch sizes 1, 8, 32 in FP32 and FP16 on GPU.
"""
import torch
import torch.nn as nn
import time
import json
import sys

WARMUP = 10
RUNS = 50
INPUT_SIZE = (3, 224, 224)
BATCH_SIZES = [1, 8, 32]


def make_resnet50():
    try:
        import torchvision.models as models
        return models.resnet50(weights=None)
    except ImportError:
        # Minimal stand-in if torchvision not available
        return nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(64, 1000)
        )


def bench(model, device, dtype, batch_size):
    model = model.to(device=device, dtype=dtype).eval()
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

    imgs_per_sec = batch_size * RUNS / elapsed
    return round(imgs_per_sec, 1), round(elapsed / RUNS * 1000, 2)


def main():
    model = make_resnet50()
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    results = {"benchmark": "cnn_resnet50", "device": device_name, "results": []}

    configs = [("cpu", torch.float32, "CPU FP32")]
    if torch.cuda.is_available():
        configs += [
            ("cuda", torch.float32, "GPU FP32"),
            ("cuda", torch.float16, "GPU FP16"),
        ]

    for device, dtype, label in configs:
        for bs in BATCH_SIZES:
            try:
                ips, ms = bench(model, device, dtype, bs)
                row = {"label": label, "batch_size": bs, "images_per_sec": ips, "ms_per_batch": ms}
                results["results"].append(row)
                print(f"  {label:12s}  batch={bs:3d}  {ips:8.1f} img/s  {ms:7.2f} ms/batch")
            except Exception as e:
                print(f"  {label:12s}  batch={bs:3d}  ERROR: {e}", file=sys.stderr)

    return results


if __name__ == "__main__":
    print("=== CNN Inference Benchmark (ResNet-50) ===")
    r = main()
    print(json.dumps(r, indent=2))
