import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    # Batch and head indices
    off_z = off_hz // H
    off_h = off_hz % H

    # Offset pointers for batch/head
    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    o_offset = off_z * stride_oz + off_h * stride_oh

    # Block pointers
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + k_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = V + v_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

    # Load Q explicitly with a mask since seqlen might not be multiple of block size
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # initialize accumulator
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Inner loop
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # Load K, V
        k = tl.load(k_ptrs + start_n * stride_kn, mask=(start_n + offs_n)[:, None] < N_CTX, other=0.0)
        v = tl.load(v_ptrs + start_n * stride_vn, mask=(start_n + offs_n)[:, None] < N_CTX, other=0.0)

        # compute qk
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= sm_scale

        # Incorporate causal mask
        # We only mask out if start_n + offs_n > offs_m (i.e. attending to future tokens)
        mask = offs_m[:, None] >= (start_n + offs_n)[None, :]
        qk = tl.where(mask, qk, float("-inf"))

        # compute scaling (m_ij)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        
        # compute P
        p = tl.math.exp(qk)

        # scaling factors (l_ij)
        l_ij = tl.sum(p, 1)

        # compute alpha
        alpha = tl.math.exp(m_i - m_ij)
        
        # scale accumulator
        acc = acc * alpha[:, None]
        
        # update l_i and m_i
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        # execute dot product with v
        # Cast P to float16 since inputs to dot must be float16/bfloat16
        p = p.to(tl.float16)
        acc += tl.dot(p, v)

    # Normalize output
    acc = acc / l_i[:, None]
    
    # Store
    tl.store(o_ptrs, acc.to(tl.float16), mask=offs_m[:, None] < N_CTX)

def triton_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float):
    # q, k, v shape: (B, H, T, D)
    B, H, T, D = q.size()
    
    # Output tensor must be pre-allocated and contiguous
    out = torch.empty_like(q)
    
    BLOCK_M = 128
    BLOCK_N = 128
    
    # Calculate grid layout
    # grid[0] handles sequence length M tiling
    # grid[1] handles batch * heads
    grid = (triton.cdiv(T, BLOCK_M), B * H, 1)

    _fwd_kernel[grid](
        q, k, v, sm_scale,
        out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, T,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=D,
        num_warps=4,
        num_stages=2,
    )
    
    return out
