"""PyTorch reference implementations used for correctness checks."""
import math
import torch
import torch.nn.functional as F


def gemm_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b


def attention_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Scaled dot-product attention for tensors shaped [B, H, S, D]."""
    return F.scaled_dot_product_attention(
        q, k, v, dropout_p=0.0, is_causal=False, scale=1.0 / math.sqrt(q.shape[-1])
    )
