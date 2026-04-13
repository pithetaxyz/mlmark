# mlmark — Benchmark Report

**Device:** NVIDIA GeForce RTX 3050 OEM  
**Date:** 2026-04-12 23:58  

---

## Matrix Multiplication (TFLOPS)

| Tier | Size | CPU FP32 | GPU FP32 | GPU FP16 | GPU FP8 | GPU FP4 |
|------|------|----------|----------|----------|---------|---------|
| small | 512 | 0.350 | 3.506 | 7.077 | — | — |
| medium | 2048 | 0.488 | 4.959 | 14.278 | — | — |
| large | 8192 | 0.541 | 4.924 | 16.896 | — | — |

## CNN Inference — ResNet-50 (img/s)

| Tier | Batch | CPU FP32 | GPU FP32 | GPU FP16 |
|------|-------|----------|----------|----------|
| small | 8 | 31.7 | 333.5 | 578.4 |
| medium | 32 | 30.9 | 370.2 | 669.8 |
| large | 128 | 26.4 | 388.3 | 698.9 |

## Transformer Inference — GPT-2 scale (tokens/s)

| Tier | Seq len | CPU FP32 | GPU FP32 | GPU FP16 |
|------|---------|----------|----------|----------|
| small | 128 | 1798 | 16651 | 27348 |
| medium | 512 | 1807 | 18011 | 48129 |
| large | 2048 | 1305 | 10493 | 21910 |

## Memory Bandwidth (GB/s)

| Tier | Size | H2D | D2H | D2D |
|------|------|-----|-----|-----|
| small | 64 MB | 10.26 | 11.07 | 77.55 |
| medium | 512 MB | 10.59 | 8.11 | 77.66 |
| large | 2048 MB | 10.16 | 4.2 | 22.12 |

> **H2D** Host→GPU  **D2H** GPU→Host  **D2D** GPU internal — on iGPU all share physical RAM, so H2D/D2H are CPU-controller bound (~2-3 GB/s) while D2D is a fast memcopy (~290 GB/s).

## Peak Results

- **Matmul:** 16.896 TFLOPS  (GPU FP16, N=8192)
- **CNN ResNet-50:** 698.9 img/s  (GPU FP16, batch=128)
- **Transformer:** 48129 tok/s  (GPU FP16, seq=512)
- **Memory D2D:** 77.7 GB/s  (GPU, 512 MB)
