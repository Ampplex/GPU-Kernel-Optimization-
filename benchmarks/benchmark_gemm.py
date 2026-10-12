import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import triton.testing
from src.kernels import triton_gemm


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--m', type=int, default=1024)
    p.add_argument('--n', type=int, default=1024)
    p.add_argument('--k', type=int, default=1024)
    p.add_argument('--dtype', choices=['fp16', 'bf16', 'fp32'], default='fp16')
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA GPU not available. Run this benchmark in a GPU runtime such as Colab.')
    dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}[args.dtype]
    a = torch.randn((args.m, args.k), device='cuda', dtype=dtype)
    b = torch.randn((args.k, args.n), device='cuda', dtype=dtype)
    ref = a @ b
    out = triton_gemm(a, b)
    torch.testing.assert_close(out, ref, rtol=2e-2 if dtype != torch.float32 else 1e-3,
                               atol=2e-2 if dtype != torch.float32 else 1e-3)
    torch_ms = triton.testing.do_bench(lambda: a @ b)
    triton_ms = triton.testing.do_bench(lambda: triton_gemm(a, b))
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'Shape: A={tuple(a.shape)}, B={tuple(b.shape)} | dtype={args.dtype}')
    print(f'PyTorch GEMM: {torch_ms:.4f} ms')
    print(f'Triton GEMM:  {triton_ms:.4f} ms')
    print(f'Speedup (PyTorch / Triton): {torch_ms / triton_ms:.3f}x')


if __name__ == '__main__':
    main()
