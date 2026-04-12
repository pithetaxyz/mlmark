# mlmark — Benchmark Report

**Device:** NVIDIA GeForce RTX 4090  
**Date:** 2026-04-12 19:13  

---

## Matrix Multiplication (TFLOPS)

| Tier | Size | CPU FP32 | GPU FP32 | GPU FP16 |
|------|------|----------|----------|----------|
| small | 512 | 0.633 | 15.399 | 16.232 |
| medium | 2048 | 0.962 | 55.305 | 146.141 |
| large | 8192 | 1.144 | 54.205 | 160.916 |

## CNN Inference — ResNet-50 (img/s)

| Tier | Batch | CPU FP32 | GPU FP32 | GPU FP16 |
|------|-------|----------|----------|----------|
| small | 1 | 42.9 | 269.6 | 252.9 |
| medium | 8 | 58.0 | 2517.5 | 1995.5 |
| large | 32 | 48.7 | 2664.8 | 4922.3 |

## Transformer Inference — GPT-2 scale (tokens/s)

| Tier | Seq len | CPU FP32 | GPU FP32 | GPU FP16 |
|------|---------|----------|----------|----------|
| small | 128 | 3050 | 39356 | 50105 |
| medium | 512 | 3438 | 94710 | 169030 |
| large | 2048 | 1779 | 70947 | 181327 |

## Memory Bandwidth (GB/s)

| Tier | Size | H2D | D2H | D2D |
|------|------|-----|-----|-----|
| small | 64 MB | 11.56 | 6.72 | 5788.51 |
| medium | 512 MB | 12.29 | 6.17 | 12901.24 |
| large | 2048 MB | 11.56 | 4.79 | 13301.48 |

> **H2D** Host→GPU  **D2H** GPU→Host  **D2D** GPU internal — on iGPU all share physical RAM, so H2D/D2H are CPU-controller bound (~2-3 GB/s) while D2D is a fast memcopy (~290 GB/s).

## Peak Results

- **Matmul:** 314.367 TFLOPS  (GPU FP8, N=8192)
- **CNN ResNet-50:** 4922.3 img/s  (GPU FP16, batch=32)
- **Transformer:** 181327 tok/s  (GPU FP16, seq=2048)
- **Memory D2D:** 13301.5 GB/s  (GPU, 2048 MB)
