"""Triton GPU kernels. Requires a CUDA-enabled PyTorch build and a supported NVIDIA GPU."""
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel(
    A, B, C,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kb in range(triton.cdiv(K, BK)):
        kk = kb * BK + rk
        a = tl.load(A + rm[:, None] * stride_am + kk[None, :] * stride_ak,
                    (rm[:, None] < M) & (kk[None, :] < K), other=0)
        b = tl.load(B + kk[:, None] * stride_bk + rn[None, :] * stride_bn,
                    (kk[:, None] < K) & (rn[None, :] < N), other=0)
        acc += tl.dot(a, b)
    c = acc.to(C.dtype.element_ty)
    tl.store(C + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
             c, (rm[:, None] < M) & (rn[None, :] < N))


def triton_gemm(a: torch.Tensor, b: torch.Tensor, *, tile_m=32, tile_n=64, tile_k=32) -> torch.Tensor:
    """Tiled GEMM for contiguous 2-D CUDA tensors with compatible shapes."""
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("triton_gemm requires CUDA tensors on an NVIDIA GPU")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("Expected A[M,K] and B[K,N]")
    if a.device != b.device:
        raise ValueError("A and B must be on the same device")
    if a.dtype != b.dtype or a.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("A and B must have the same dtype: fp16, bf16, or fp32")
    a, b = a.contiguous(), b.contiguous()
    m, k = a.shape
    _, n = b.shape
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(m, tile_m), triton.cdiv(n, tile_n))
    _gemm_kernel[grid](a, b, c, m, n, k, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                       c.stride(0), c.stride(1), tile_m, tile_n, tile_k)
    return c


@triton.jit
def _attention_fwd_kernel(
    Q, K, V, O,
    S: tl.constexpr, D: tl.constexpr, HEADS: tl.constexpr,
    stride_qb: tl.constexpr, stride_qh: tl.constexpr, stride_qs: tl.constexpr, stride_qd: tl.constexpr,
    stride_kb: tl.constexpr, stride_kh: tl.constexpr, stride_ks: tl.constexpr, stride_kd: tl.constexpr,
    stride_vb: tl.constexpr, stride_vh: tl.constexpr, stride_vs: tl.constexpr, stride_vd: tl.constexpr,
    stride_ob: tl.constexpr, stride_oh: tl.constexpr, stride_os: tl.constexpr, stride_od: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    SCALE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch = pid_bh // HEADS
    head = pid_bh % HEADS
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    q_ptrs = Q + batch * stride_qb + head * stride_qh + rows[:, None] * stride_qs + cols_d[None, :] * stride_qd
    q = tl.load(q_ptrs, (rows[:, None] < S) & (cols_d[None, :] < D), other=0)
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for start_n in range(triton.cdiv(S, BLOCK_N)):
        key_rows = start_n * BLOCK_N + offs_n
        k_ptrs = K + batch * stride_kb + head * stride_kh + key_rows[None, :] * stride_ks + cols_d[:, None] * stride_kd
        k = tl.load(k_ptrs, (key_rows[None, :] < S) & (cols_d[:, None] < D), other=0)
        scores = tl.dot(q, k) * SCALE
        scores = tl.where(key_rows[None, :] < S, scores, -float("inf"))
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        p = tl.exp(scores - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_ij = alpha * l_i + tl.sum(p, axis=1)
        v_ptrs = V + batch * stride_vb + head * stride_vh + key_rows[:, None] * stride_vs + cols_d[None, :] * stride_vd
        v = tl.load(v_ptrs, (key_rows[:, None] < S) & (cols_d[None, :] < D), other=0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l_i = l_ij
        m_i = m_ij
    out = acc / l_i[:, None]
    o_ptrs = O + batch * stride_ob + head * stride_oh + rows[:, None] * stride_os + cols_d[None, :] * stride_od
    tl.store(o_ptrs, out, (rows[:, None] < S) & (cols_d[None, :] < D))


def triton_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, block_m=32, block_n=64) -> torch.Tensor:
    """Forward-only non-causal attention for Q/K/V shaped [B,H,S,D].

    Uses online softmax and does not materialize the full [S,S] score matrix.
    Head dimension must be <= 128 and all tensors must share dtype/device/shape.
    """
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("triton_attention requires CUDA tensors on an NVIDIA GPU")
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Q, K, and V must have the same [B,H,S,D] shape")
    if q.device != k.device or q.device != v.device or q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("Q, K, and V must share device and dtype")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Supported dtypes: fp16, bf16, fp32")
    b, h, s, d = q.shape
    if d > 128:
        raise ValueError("This starter kernel supports head dimensions up to 128")
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = torch.empty_like(q)
    block_d = triton.next_power_of_2(d)
    grid = (triton.cdiv(s, block_m), b * h)
    _attention_fwd_kernel[grid](
        q, k, v, out, s, d, h,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        block_m, block_n, block_d, 1.0 / math.sqrt(d),
        num_warps=4,
    )
    return out
