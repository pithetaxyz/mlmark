# mlmark — Benchmark Report

**Device:** NVIDIA GeForce RTX 3050 OEM  
**Date:** 2026-04-12 13:04  

---

## Matrix Multiplication (TFLOPS)

| Tier | Size | CPU FP32 | GPU FP32 | GPU FP16 | GPU FP8 | GPU FP4 |
|------|------|----------|----------|----------|---------|---------|
| small | 512 | 0.352 | 3.519 | 10.661 | — | — |
| medium | 2048 | 0.459 | 4.975 | 14.569 | — | — |
| large | 8192 | 0.431 | 4.893 | 16.636 | — | — |

## CNN Inference — ResNet-50 (img/s)

| Tier | Batch | CPU FP32 | GPU FP32 | GPU FP16 |
|------|-------|----------|----------|----------|
| small | 32 | 831.1 | 5955.6 | 8072.2 |
| medium | 128 | 754.0 | 6084.0 | 8459.4 |
| large | 512 | 371.3 | 6056.1 | 8399.2 |

## Transformer Inference — GPT-2 scale (tokens/s)

| Tier | Seq len | CPU FP32 | GPU FP32 | GPU FP16 |
|------|---------|----------|----------|----------|
| small | 128 | 1728 | 16768 | 29935 |
| medium | 512 | 1916 | 18028 | 48489 |
| large | 2048 | 1190 | 10520 | 21771 |

## Memory Bandwidth (GB/s)

| Tier | Size | H2D | D2H | D2D |
|------|------|-----|-----|-----|
| small | 64 MB | 10.93 | 9.93 | 12742.11 |
| medium | 512 MB | 10.48 | 10.19 | 62539.07 |
| large | 2048 MB | 7.53 | 4.3 | 388.91 |

> **H2D** Host→GPU  **D2H** GPU→Host  **D2D** GPU internal — on iGPU all share physical RAM, so H2D/D2H are CPU-controller bound (~2-3 GB/s) while D2D is a fast memcopy (~290 GB/s).

## Peak Results

- **Matmul:** 16.636 TFLOPS  (GPU FP16, N=8192)
- **CNN ResNet-50:** 8459.4 img/s  (GPU FP16, batch=128)
- **Transformer:** 48489 tok/s  (GPU FP16, seq=512)
- **Memory D2D:** 62539.1 GB/s  (GPU, 512 MB)
