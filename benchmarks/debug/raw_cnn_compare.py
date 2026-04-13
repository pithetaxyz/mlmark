"""
ResNet-50 from scratch — no torchvision dependency.
Matches the original paper architecture exactly.
Run with: python resnet50_scratch.py
"""

import torch
import torch.nn as nn
import time


# ── Building blocks ───────────────────────────────────────────────────────────

class ConvBnRelu(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride=stride,
                              padding=padding, groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """
    ResNet-50 bottleneck: 1x1 → 3x3 → 1x1 with optional shortcut projection.
    expansion=4 means the output channels are 4× the base width.
    """
    expansion = 4

    def __init__(self, in_c, base_c, stride=1, downsample=None):
        super().__init__()
        self.conv1 = ConvBnRelu(in_c,          base_c,          1)
        self.conv2 = ConvBnRelu(base_c,         base_c,          3, stride=stride, padding=1)
        self.conv3 = nn.Conv2d( base_c,         base_c * 4,      1, bias=False)
        self.bn3   = nn.BatchNorm2d(base_c * 4)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet50(nn.Module):
    """
    ResNet-50: [3, 4, 6, 3] bottleneck blocks across 4 stages.
    Output: (batch, 1000) logits.
    """
    def __init__(self, num_classes=1000):
        super().__init__()
        self.in_c = 64

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        # Residual stages
        self.layer1 = self._make_stage(64,  3, stride=1)
        self.layer2 = self._make_stage(128, 4, stride=2)
        self.layer3 = self._make_stage(256, 6, stride=2)
        self.layer4 = self._make_stage(512, 3, stride=2)

        # Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512 * Bottleneck.expansion, num_classes),
        )

        # Weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_stage(self, base_c, n_blocks, stride):
        out_c = base_c * Bottleneck.expansion
        downsample = None
        if stride != 1 or self.in_c != out_c:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )
        layers = [Bottleneck(self.in_c, base_c, stride=stride, downsample=downsample)]
        self.in_c = out_c
        for _ in range(1, n_blocks):
            layers.append(Bottleneck(self.in_c, base_c))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(x)


# ── Benchmark ─────────────────────────────────────────────────────────────────

def timed(model, x, device, warmup=20, runs=50):
    batch = x.shape[0]
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    return round(batch * runs / (time.perf_counter() - t), 1)


def sep(w=60): print("-" * w)

BATCHES = {"small": 32, "medium": 128, "large": 512}
CONFIGS = [
    ("cpu", torch.float32, "CPU FP32"),
]
if torch.cuda.is_available():
    CONFIGS += [
        ("cuda", torch.float32, "GPU FP32"),
        ("cuda", torch.float16, "GPU FP16"),
    ]

print("=" * 60)
print(f"  ResNet-50 (scratch) — CNN Inference Benchmark")
if torch.cuda.is_available():
    print(f"  GPU : {torch.cuda.get_device_name(0)}")
print(f"  PyTorch : {torch.__version__}")
print("=" * 60)
print(f"  {'':20} {'small (32)':>10} {'medium (128)':>13} {'large (512)':>12}")
sep()

for device, dtype, label in CONFIGS:
    row = []
    for tier, batch in BATCHES.items():
        model = ResNet50().to(device=device, dtype=dtype).eval()
        x = torch.randn(batch, 3, 224, 224, device=device, dtype=dtype)
        img_s = timed(model, x, device)
        row.append(img_s)
    print(f"  {label:20} {row[0]:>10.1f} {row[1]:>13.1f} {row[2]:>12.1f}")

sep()
print("Done.")