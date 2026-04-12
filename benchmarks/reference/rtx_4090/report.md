# mlmark — Benchmark Report

**Device:** NVIDIA GeForce RTX 4090  
**Date:** 2026-04-12 19:27  

---

## Matrix Multiplication (TFLOPS)

| Tier | Size | CPU FP32 | GPU FP32 | GPU FP16 | GPU FP8 | GPU FP4 |
|------|------|----------|----------|----------|---------|---------|
| small | 512 | 0.772 | 18.423 | 16.538 | 2.446 | — |
| medium | 2048 | 1.002 | 25.120 | 146.797 | 169.235 | — |
| large | 8192 | 1.165 | 52.946 | 153.144 | 318.655 | — |

## CNN Inference — ResNet-50 (img/s)

| Tier | Batch | CPU FP32 | GPU FP32 | GPU FP16 |
|------|-------|----------|----------|----------|
| small | 1 | 41.0 | 322.2 | 178.6 |
| medium | 8 | 58.9 | 2140.1 | 1455.8 |
| large | 32 | 45.4 | 2670.1 | 5041.2 |

## Transformer Inference — GPT-2 scale (tokens/s)

| Tier | Seq len | CPU FP32 | GPU FP32 | GPU FP16 |
|------|---------|----------|----------|----------|
| small | 128 | 3044 | 16295 | 44134 |
| medium | 512 | 3320 | 99304 | 176411 |
| large | 2048 | 1838 | 71397 | 179865 |

## Memory Bandwidth (GB/s)

| Tier | Size | H2D | D2H | D2D |
|------|------|-----|-----|-----|
| small | 64 MB | 10.38 | 6.36 | 6513.44 |
| medium | 512 MB | 10.93 | 6.14 | 12916.58 |
| large | 2048 MB | 11.6 | 5.03 | 13040.4 |

> **H2D** Host→GPU  **D2H** GPU→Host  **D2D** GPU internal — on iGPU all share physical RAM, so H2D/D2H are CPU-controller bound (~2-3 GB/s) while D2D is a fast memcopy (~290 GB/s).

## Peak Results

- **Matmul:** 318.655 TFLOPS  (GPU FP8, N=8192)
- **CNN ResNet-50:** 5041.2 img/s  (GPU FP16, batch=32)
- **Transformer:** 179865 tok/s  (GPU FP16, seq=2048)
- **Memory D2D:** 13040.4 GB/s  (GPU, 2048 MB)
