import pytest
import torch
import torch.nn.functional as F

from src.kernels import triton_gemm, triton_attention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a supported NVIDIA GPU")


def test_gemm_fp16():
    a = torch.randn((127, 96), device="cuda", dtype=torch.float16)
    b = torch.randn((96, 83), device="cuda", dtype=torch.float16)
    torch.testing.assert_close(triton_gemm(a, b), a @ b, rtol=2e-2, atol=2e-2)


def test_attention_fp16():
    q = torch.randn((1, 2, 65, 32), device="cuda", dtype=torch.float16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    actual = triton_attention(q, k, v)
    expected = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
