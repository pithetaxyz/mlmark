# mlmark — Benchmark Report

**Device:** NVIDIA GeForce RTX 5060  
**Date:** 2026-04-13 00:08  

---

## Matrix Multiplication (TFLOPS)

| Tier | Size | CPU FP32 | GPU FP32 | GPU FP16 | GPU FP8 | GPU FP4 |
|------|------|----------|----------|----------|---------|---------|
| small | 512 | 0.523 | 7.405 | 14.054 | 5.844 | 4.874 |
| medium | 2048 | 0.825 | 13.129 | 40.503 | 69.087 | 201.867 |
| large | 8192 | 0.857 | 13.905 | 39.316 | 81.193 | 287.144 |

## CNN Inference — ResNet-50 (img/s)

| Tier | Batch | CPU FP32 | GPU FP32 | GPU FP16 |
|------|-------|----------|----------|----------|
| small | 8 | 51.5 | 791.4 | 1312.7 |
| medium | 32 | 51.2 | 804.9 | 1514.5 |
| large | 128 | 43.3 | 806.1 | 1418.3 |

## Transformer Inference — GPT-2 scale (tokens/s)

| Tier | Seq len | CPU FP32 | GPU FP32 | GPU FP16 |
|------|---------|----------|----------|----------|
| small | 128 | 2563 | 26344 | 40036 |
| medium | 512 | 3042 | 38100 | 106988 |
| large | 2048 | 1816 | 23964 | 64441 |

## Memory Bandwidth (GB/s)

| Tier | Size | H2D | D2H | D2D |
|------|------|-----|-----|-----|
| small | 64 MB | 13.83 | 5.16 | 97.47 |
| medium | 512 MB | 13.47 | 11.66 | 141.41 |
| large | 2048 MB | 13.16 | 3.01 | 38.27 |

> **H2D** Host→GPU  **D2H** GPU→Host  **D2D** GPU internal — on iGPU all share physical RAM, so H2D/D2H are CPU-controller bound (~2-3 GB/s) while D2D is a fast memcopy (~290 GB/s).

## Peak Results

- **Matmul:** 287.144 TFLOPS  (GPU FP4, N=8192)
- **CNN ResNet-50:** 1514.5 img/s  (GPU FP16, batch=32)
- **Transformer:** 106988 tok/s  (GPU FP16, seq=512)
- **Memory D2D:** 141.4 GB/s  (GPU, 512 MB)
