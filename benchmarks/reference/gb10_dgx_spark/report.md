# mlmark — Benchmark Report

**Device:** NVIDIA GB10  
**Date:** 2026-04-13 06:45  

---

## Matrix Multiplication (TFLOPS)

| Tier | Size | CPU FP32 | GPU FP32 | GPU FP16 | GPU FP8 | GPU FP4 |
|------|------|----------|----------|----------|---------|---------|
| small | 512 | 0.413 | 5.038 | 7.510 | 2.380 | 5.967 |
| medium | 2048 | 0.731 | 33.926 | 54.739 | 60.823 | 160.303 |
| large | 8192 | 0.791 | 38.182 | 60.656 | 134.389 | 314.485 |

## CNN Inference — ResNet-50 (img/s)

| Tier | Batch | CPU FP32 | GPU FP32 | GPU FP16 |
|------|-------|----------|----------|----------|
| small | 8 | 21.7 | 677.1 | 1326.1 |
| medium | 32 | 18.9 | 550.0 | 1149.0 |
| large | 128 | 23.7 | 561.2 | 1054.3 |

## Transformer Inference — GPT-2 scale (tokens/s)

| Tier | Seq len | CPU FP32 | GPU FP32 | GPU FP16 |
|------|---------|----------|----------|----------|
| small | 128 | 1748 | 42955 | 76210 |
| medium | 512 | 1633 | 69041 | 126484 |
| large | 2048 | 1044 | 23821 | 45231 |

## Memory Bandwidth (GB/s)

| Tier | Size | H2D | D2H | D2D |
|------|------|-----|-----|-----|
| small | 64 MB | 29.36 | 2.94 | 61.42 |
| medium | 512 MB | 42.41 | 3.75 | 71.81 |
| large | 2048 MB | 45.61 | 4.32 | 76.08 |

> **H2D** Host→GPU  **D2H** GPU→Host  **D2D** GPU internal — on iGPU all share physical RAM, so H2D/D2H are CPU-controller bound (~2-3 GB/s) while D2D is a fast memcopy (~290 GB/s).

## Peak Results

- **Matmul:** 314.485 TFLOPS  (GPU FP4, N=8192)
- **CNN ResNet-50:** 1326.1 img/s  (GPU FP16, batch=8)
- **Transformer:** 126484 tok/s  (GPU FP16, seq=512)
- **Memory D2D:** 76.1 GB/s  (GPU, 2048 MB)
