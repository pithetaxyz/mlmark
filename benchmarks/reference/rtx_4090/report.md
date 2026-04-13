# mlmark — Benchmark Report

**Device:** NVIDIA GeForce RTX 4090  
**Date:** 2026-04-12 19:27  

---

## Matrix Multiplication (TFLOPS)

| Tier | Size | CPU FP32 | GPU FP32 | GPU FP16 | GPU FP8 | GPU FP4 |
|------|------|----------|----------|----------|---------|---------|
| small | 512 | 0.695 | 18.891 | 16.580 | 6.185 | — |
| medium | 2048 | 1.172 | 49.022 | 149.410 | 127.518 | — |
| large | 8192 | 1.042 | 53.942 | 160.880 | 326.635 | — |

## CNN Inference — ResNet-50 (img/s)

| Tier | Batch | CPU FP32 | GPU FP32 | GPU FP16 |
|------|-------|----------|----------|----------|
| small | 8 | 65.9 | 2632.0 | 1935.4 |
| medium | 32 | 60.4 | 2526.4 | 4905.2 |
| large | 128 | 49.0 | 2331.0 | 4511.3 |

## Transformer Inference — GPT-2 scale (tokens/s)

| Tier | Seq len | CPU FP32 | GPU FP32 | GPU FP16 |
|------|---------|----------|----------|----------|
| small | 128 | 2440 | 21604 | 32985 |
| medium | 512 | 3711 | 61053 | 104433 |
| large | 2048 | 2414 | 65121 | 159694 |

## Memory Bandwidth (GB/s)

| Tier | Size | H2D | D2H | D2D |
|------|------|-----|-----|-----|
| small | 64 MB | 10.79 | 11.95 | 18712.6 |
| medium | 512 MB | 11.05 | 10.71 | 132275.14 |
| large | 2048 MB | 10.94 | 5.03 | 750.44 |

> **H2D** Host→GPU  **D2H** GPU→Host  **D2D** GPU internal — on iGPU all share physical RAM, so H2D/D2H are CPU-controller bound (~2-3 GB/s) while D2D is a fast memcopy (~290 GB/s).

## Peak Results

- **Matmul:** 326.635 TFLOPS  (GPU FP8, N=8192)
- **CNN ResNet-50:** 4905.2 img/s  (GPU FP16, batch=32)
- **Transformer:** 159694 tok/s  (GPU FP16, seq=2048)
- **Memory D2D:** 132275.1 GB/s  (GPU, 512 MB)
