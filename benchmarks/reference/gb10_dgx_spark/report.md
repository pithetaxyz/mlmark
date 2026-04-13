# mlmark — Benchmark Report

**Device:** NVIDIA GB10  
**Date:** 2026-04-13 05:06  

---

## Matrix Multiplication (TFLOPS)

| Tier | Size | CPU FP32 | GPU FP32 | GPU FP16 | GPU FP8 | GPU FP4 |
|------|------|----------|----------|----------|---------|---------|
| small | 512 | 0.303 | 10.934 | 6.558 | 0.541 | — |
| medium | 2048 | 0.744 | 32.660 | 66.222 | 60.647 | — |
| large | 8192 | 0.772 | 42.508 | 61.105 | 133.702 | — |

## CNN Inference — ResNet-50 (img/s)

| Tier | Batch | CPU FP32 | GPU FP32 | GPU FP16 |
|------|-------|----------|----------|----------|
| small | 8 | 20.1 | 680.1 | 1299.8 |
| medium | 32 | 19.2 | 565.0 | 1181.3 |
| large | 128 | 23.7 | 579.4 | 1086.1 |

## Transformer Inference — GPT-2 scale (tokens/s)

| Tier | Seq len | CPU FP32 | GPU FP32 | GPU FP16 |
|------|---------|----------|----------|----------|
| small | 128 | 1873 | 43786 | 78164 |
| medium | 512 | 2158 | 72441 | 129268 |
| large | 2048 | 1059 | 24406 | 46545 |

## Memory Bandwidth (GB/s)

| Tier | Size | H2D | D2H | D2D |
|------|------|-----|-----|-----|
| small | 64 MB | 29.79 | 0.93 | 280.94 |
| medium | 512 MB | 41.87 | 4.41 | 285.12 |
| large | 2048 MB | 45.34 | 4.34 | 379.52 |

> **H2D** Host→GPU  **D2H** GPU→Host  **D2D** GPU internal — on iGPU all share physical RAM, so H2D/D2H are CPU-controller bound (~2-3 GB/s) while D2D is a fast memcopy (~290 GB/s).

## Peak Results

- **Matmul:** 133.702 TFLOPS  (GPU FP8, N=8192)
- **CNN ResNet-50:** 1299.8 img/s  (GPU FP16, batch=8)
- **Transformer:** 129268 tok/s  (GPU FP16, seq=512)
- **Memory D2D:** 379.5 GB/s  (GPU, 2048 MB)
