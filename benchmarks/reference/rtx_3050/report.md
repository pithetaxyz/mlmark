# mlmark — Benchmark Report

**Device:** NVIDIA GeForce RTX 3050 OEM  
**Date:** 2026-04-12 19:59  

---

## Matrix Multiplication (TFLOPS)

| Tier | Size | CPU FP32 | GPU FP32 | GPU FP16 | GPU FP8 | GPU FP4 |
|------|------|----------|----------|----------|---------|---------|
| small | 512 | 0.350 | 3.529 | 9.694 | — | — |
| medium | 2048 | 0.420 | 4.963 | 14.374 | — | — |
| large | 8192 | 0.505 | 4.910 | 16.886 | — | — |

## CNN Inference — ResNet-50 (img/s)

| Tier | Batch | CPU FP32 | GPU FP32 | GPU FP16 |
|------|-------|----------|----------|----------|
| small | 8 | 804.4 | 5283.8 | 8182.5 |
| medium | 32 | 830.1 | 5984.1 | 8330.0 |
| large | 128 | 735.4 | 6029.9 | 8394.0 |

## Transformer Inference — GPT-2 scale (tokens/s)

| Tier | Seq len | CPU FP32 | GPU FP32 | GPU FP16 |
|------|---------|----------|----------|----------|
| small | 128 | 1992 | 16870 | 35969 |
| medium | 512 | 1816 | 17995 | 48402 |
| large | 2048 | 1246 | 10506 | 21921 |

## Memory Bandwidth (GB/s)

| Tier | Size | H2D | D2H | D2D |
|------|------|-----|-----|-----|
| small | 64 MB | 10.69 | 10.4 | 13185.65 |
| medium | 512 MB | 10.58 | 8.74 | 119047.7 |
| large | 2048 MB | 10.33 | 4.52 | 529.21 |

> **H2D** Host→GPU  **D2H** GPU→Host  **D2D** GPU internal — on iGPU all share physical RAM, so H2D/D2H are CPU-controller bound (~2-3 GB/s) while D2D is a fast memcopy (~290 GB/s).

## Peak Results

- **Matmul:** 16.886 TFLOPS  (GPU FP16, N=8192)
- **CNN ResNet-50:** 8394.0 img/s  (GPU FP16, batch=128)
- **Transformer:** 48402 tok/s  (GPU FP16, seq=512)
- **Memory D2D:** 119047.7 GB/s  (GPU, 512 MB)
