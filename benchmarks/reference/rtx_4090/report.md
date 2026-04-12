# mlmark — Benchmark Report

**Device:** NVIDIA GeForce RTX 4090  
**Date:** 2026-04-12 09:25  

---

## Matrix Multiplication (TFLOPS)

| Tier | Size | CPU FP32 | GPU FP32 | GPU FP16 |
|------|------|----------|----------|----------|
| small | 512 | 0.868 | 10.023 | 15.089 |
| medium | 2048 | 0.980 | 43.341 | 110.703 |
| large | 8192 | 1.215 | 56.162 | 165.749 |

## CNN Inference — ResNet-50 (img/s)

| Tier | Batch | CPU FP32 | GPU FP32 | GPU FP16 |
|------|-------|----------|----------|----------|
| small | 1 | 44.0 | 334.6 | 248.3 |
| medium | 8 | 70.3 | 2549.8 | 2056.3 |
| large | 32 | 51.1 | 2755.1 | 5161.1 |

## Transformer Inference — GPT-2 scale (tokens/s)

| Tier | Seq len | CPU FP32 | GPU FP32 | GPU FP16 |
|------|---------|----------|----------|----------|
| small | 128 | 3142 | 20517 | 44308 |
| medium | 512 | 3606 | 50121 | 205964 |
| large | 2048 | 1885 | 73177 | 182559 |

## Memory Bandwidth (GB/s)

| Tier | Size | H2D | D2H | D2D |
|------|------|-----|-----|-----|
| small | 64 MB | 12.72 | 6.85 | 6132.32 |
| medium | 512 MB | 10.2 | 6.44 | 13595.05 |
| large | 2048 MB | 11.63 | 4.86 | 14878.89 |

> **H2D** Host→GPU  **D2H** GPU→Host  **D2D** GPU internal — on iGPU all share physical RAM, so H2D/D2H are CPU-controller bound (~2-3 GB/s) while D2D is a fast memcopy (~290 GB/s).

## Peak Results

- **Matmul:** 165.749 TFLOPS  (GPU FP16, N=8192)
- **CNN ResNet-50:** 5161.1 img/s  (GPU FP16, batch=32)
- **Transformer:** 205964 tok/s  (GPU FP16, seq=512)
- **Memory D2D:** 15692.5 GB/s  (GPU, 2048 MB)
