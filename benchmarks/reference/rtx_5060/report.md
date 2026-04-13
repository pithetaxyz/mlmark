# mlmark — Benchmark Report

**Device:** NVIDIA GeForce RTX 5060  
**Date:** 2026-04-12 20:02  

---

## Matrix Multiplication (TFLOPS)

| Tier | Size | CPU FP32 | GPU FP32 | GPU FP16 | GPU FP8 | GPU FP4 |
|------|------|----------|----------|----------|---------|---------|
| small | 512 | 0.468 | 7.640 | 15.472 | 4.529 | — |
| medium | 2048 | 0.799 | 13.357 | 40.336 | 68.799 | — |
| large | 8192 | 0.825 | 13.630 | 39.118 | 81.173 | — |

## CNN Inference — ResNet-50 (img/s)

| Tier | Batch | CPU FP32 | GPU FP32 | GPU FP16 |
|------|-------|----------|----------|----------|
| small | 8 | 47.8 | 784.3 | 1440.3 |
| medium | 32 | 50.6 | 800.5 | 1508.0 |
| large | 128 | 42.5 | 803.2 | 1414.9 |

## Transformer Inference — GPT-2 scale (tokens/s)

| Tier | Seq len | CPU FP32 | GPU FP32 | GPU FP16 |
|------|---------|----------|----------|----------|
| small | 128 | 2744 | 25739 | 47730 |
| medium | 512 | 2992 | 37467 | 130355 |
| large | 2048 | 1762 | 23787 | 64169 |

## Memory Bandwidth (GB/s)

| Tier | Size | H2D | D2H | D2D |
|------|------|-----|-----|-----|
| small | 64 MB | 13.82 | 4.87 | 14108.35 |
| medium | 512 MB | 12.71 | 11.24 | 112233.43 |
| large | 2048 MB | 12.85 | 2.8 | 543.78 |

> **H2D** Host→GPU  **D2H** GPU→Host  **D2D** GPU internal — on iGPU all share physical RAM, so H2D/D2H are CPU-controller bound (~2-3 GB/s) while D2D is a fast memcopy (~290 GB/s).

## Peak Results

- **Matmul:** 81.173 TFLOPS  (GPU FP8, N=8192)
- **CNN ResNet-50:** 1508.0 img/s  (GPU FP16, batch=32)
- **Transformer:** 130355 tok/s  (GPU FP16, seq=512)
- **Memory D2D:** 112233.4 GB/s  (GPU, 512 MB)
