# GPU Kernel Optimization with PyTorch and Triton

A correctness-first project exploring GPU matrix multiplication (GEMM) and memory-efficient scaled dot-product attention using Triton.

> **Status:** implementation-stage starter. The kernels and benchmark scripts have not yet been validated on a supported NVIDIA GPU. No performance improvement is claimed.

## What is included

- Tiled Triton GEMM for 2D tensors.
- Forward-only FlashAttention-style kernel using tiled computation and online softmax.
- PyTorch reference operations and GPU correctness tests.
- Benchmark scripts that check correctness before reporting latency and measured speedup.
- Google Colab setup notes.

## Repository layout

```text
src/kernels.py                 Triton GEMM and attention kernels
src/reference.py               PyTorch references
benchmarks/benchmark_gemm.py   GEMM benchmark
benchmarks/benchmark_attention.py  Attention benchmark
tests/test_kernels.py          GPU correctness tests
notebooks/colab_quickstart.md  Colab instructions
requirements.txt
```

## Run on Google Colab

1. Open Colab and select **Runtime → Change runtime type → T4 GPU** (or another available supported NVIDIA GPU).
2. Clone this repository and install dependencies:

```bash
!git clone https://github.com/Ampplex/GPU-Kernel-Optimization-.git
%cd GPU-Kernel-Optimization-
!pip install -r requirements.txt
```

3. Run correctness checks and benchmarks:

```bash
!pytest -q
!python benchmarks/benchmark_gemm.py --m 1024 --n 1024 --k 1024 --dtype fp16
!python benchmarks/benchmark_attention.py --batch 1 --heads 4 --seq-len 512 --head-dim 64 --dtype fp16
```

The scripts exit with a clear message if CUDA is unavailable. A supported NVIDIA GPU is required to execute Triton kernels and obtain GPU performance measurements.

## Implementation notes

### Tiled GEMM

Computes `C = A @ B` by loading blocks of the input matrices and accumulating in FP32 before storing in the requested dtype. The wrapper accepts compatible 2D CUDA tensors in FP16, BF16, or FP32.

### FlashAttention-style forward pass

Computes non-causal scaled dot-product attention without materializing the full sequence-by-sequence score matrix. It uses online softmax statistics across key/value blocks. This starter implementation is **forward-only** and does not include a custom backward kernel.

## Benchmarking methodology

The benchmark scripts validate outputs against PyTorch before timing, warm up through Triton's benchmarking utility, and print the GPU model, workload shape, dtype, baseline latency, custom-kernel latency, and observed ratio. Repeat measurements across shapes and retain the environment details. A custom kernel is not guaranteed to outperform optimized PyTorch libraries.

## Current limitations

- No custom CUDA C++ kernel is included.
- No asynchronous CUDA copy pipeline or custom backward attention kernel is included.
- No speedup is claimed until measurements are run and recorded on actual NVIDIA hardware.
- Results depend on GPU architecture, software versions, shapes, precision, and launch configuration.

## Resume description

Implemented tiled GEMM and a forward-only FlashAttention-style kernel in Triton, with PyTorch reference comparisons and reproducible benchmark scripts for numerical correctness and latency evaluation. Add quantified performance results only after validating them on hardware.
