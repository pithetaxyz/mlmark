# mlmark — Benchmark Report

**Device:** NVIDIA GeForce RTX 4090  
**Date:** 2026-04-13 00:04  

---

## Matrix Multiplication (TFLOPS)

| Tier | Size | CPU FP32 | GPU FP32 | GPU FP16 | GPU FP8 | GPU FP4 |
|------|------|----------|----------|----------|---------|---------|
| small | 512 | 0.743 | 19.544 | 14.601 | 3.277 | — |
| medium | 2048 | 1.469 | 29.440 | 85.304 | 113.242 | — |
| large | 8192 | 1.213 | 54.157 | 159.990 | 328.838 | — |

## CNN Inference — ResNet-50 (img/s)

| Tier | Batch | CPU FP32 | GPU FP32 | GPU FP16 |
|------|-------|----------|----------|----------|
| small | 8 | 66.1 | 2478.8 | 1821.6 |
| medium | 32 | 66.4 | 2683.8 | 4475.8 |
| large | 128 | 51.0 | 2283.1 | 4318.9 |

## Transformer Inference — GPT-2 scale (tokens/s)

| Tier | Seq len | CPU FP32 | GPU FP32 | GPU FP16 |
|------|---------|----------|----------|----------|
| small | 128 | 2642 | 26421 | 31728 |
| medium | 512 | 3945 | 114708 | 167698 |
| large | 2048 | 2568 | 64452 | 160022 |

## Memory Bandwidth (GB/s)

| Tier | Size | H2D | D2H | D2D |
|------|------|-----|-----|-----|
| small | 64 MB | 9.8 | 11.49 | 154.4 |
| medium | 512 MB | 11.18 | 10.91 | 287.09 |
| large | 2048 MB | 11.38 | 4.8 | 207.7 |

> **H2D** Host→GPU  **D2H** GPU→Host  **D2D** GPU internal — on iGPU all share physical RAM, so H2D/D2H are CPU-controller bound (~2-3 GB/s) while D2D is a fast memcopy (~290 GB/s).

## Peak Results

- **Matmul:** 328.838 TFLOPS  (GPU FP8, N=8192)
- **CNN ResNet-50:** 4475.8 img/s  (GPU FP16, batch=32)
- **Transformer:** 167698 tok/s  (GPU FP16, seq=512)
- **Memory D2D:** 287.1 GB/s  (GPU, 512 MB)
