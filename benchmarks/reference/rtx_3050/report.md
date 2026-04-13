# mlmark — Benchmark Report

**Device:** NVIDIA GeForce RTX 3050 OEM  
**Date:** 2026-04-12 21:44  

---

## Matrix Multiplication (TFLOPS)

| Tier | Size | CPU FP32 | GPU FP32 | GPU FP16 | GPU FP8 | GPU FP4 |
|------|------|----------|----------|----------|---------|---------|
| small | 512 | 0.325 | 3.502 | 10.631 | — | — |
| medium | 2048 | 0.388 | 4.915 | 14.337 | — | — |
| large | 8192 | 0.545 | 4.911 | 16.831 | — | — |

## CNN Inference — ResNet-50 (img/s)

| Tier | Batch | CPU FP32 | GPU FP32 | GPU FP16 |
|------|-------|----------|----------|----------|
| small | 8 | 28.7 | 333.5 | 578.3 |
| medium | 32 | 32.1 | 367.2 | 662.3 |
| large | 128 | 26.0 | 387.9 | 698.7 |

## Transformer Inference — GPT-2 scale (tokens/s)

| Tier | Seq len | CPU FP32 | GPU FP32 | GPU FP16 |
|------|---------|----------|----------|----------|
| small | 128 | 1624 | 16030 | 22411 |
| medium | 512 | 1834 | 17811 | 47574 |
| large | 2048 | 1334 | 10482 | 21826 |

## Memory Bandwidth (GB/s)

| Tier | Size | H2D | D2H | D2D |
|------|------|-----|-----|-----|
| small | 64 MB | 8.61 | 10.64 | 12413.11 |
| medium | 512 MB | 10.26 | 8.4 | 119474.4 |
| large | 2048 MB | 10.18 | 3.87 | 480.39 |

> **H2D** Host→GPU  **D2H** GPU→Host  **D2D** GPU internal — on iGPU all share physical RAM, so H2D/D2H are CPU-controller bound (~2-3 GB/s) while D2D is a fast memcopy (~290 GB/s).

## Peak Results

- **Matmul:** 16.831 TFLOPS  (GPU FP16, N=8192)
- **CNN ResNet-50:** 698.7 img/s  (GPU FP16, batch=128)
- **Transformer:** 47574 tok/s  (GPU FP16, seq=512)
- **Memory D2D:** 119474.4 GB/s  (GPU, 512 MB)
