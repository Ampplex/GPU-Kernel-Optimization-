import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F
import triton.testing
from src.kernels import triton_attention


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--seq-len', type=int, default=512)
    p.add_argument('--head-dim', type=int, default=64)
    p.add_argument('--dtype', choices=['fp16', 'bf16', 'fp32'], default='fp16')
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA GPU not available. Run this benchmark in a GPU runtime such as Colab.')
    dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}[args.dtype]
    shape = (args.batch, args.heads, args.seq_len, args.head_dim)
    q, k, v = [torch.randn(shape, device='cuda', dtype=dtype) for _ in range(3)]
    ref = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    out = triton_attention(q, k, v)
    torch.testing.assert_close(out, ref, rtol=3e-2, atol=3e-2)
    torch_ms = triton.testing.do_bench(lambda: F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False))
    triton_ms = triton.testing.do_bench(lambda: triton_attention(q, k, v))
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'Shape: {shape} | dtype={args.dtype}')
    print(f'PyTorch SDPA: {torch_ms:.4f} ms')
    print(f'Triton attention: {triton_ms:.4f} ms')
    print(f'Speedup (PyTorch / Triton): {torch_ms / triton_ms:.3f}x')


if __name__ == '__main__':
    main()
