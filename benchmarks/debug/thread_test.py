import torch
import torchvision.models as models
import time

model = models.resnet50(weights=None).eval()
x = torch.randn(32, 3, 224, 224)

for n in [1, 2, 4, 8, 12, 16, 20, 24]:
    torch.set_num_threads(n)
    with torch.no_grad():
        for _ in range(10): model(x)
    t = time.perf_counter()
    with torch.no_grad():
        for _ in range(30): model(x)
    print(f"threads={n:>3}: {32*30/(time.perf_counter()-t):>8.1f} img/s")