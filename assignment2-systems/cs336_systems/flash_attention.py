"""FlashAttention-2 (forward: PyTorch-tiled and Triton; backward: PyTorch+compile).

Interface matches the tests: apply(Q, K, V, is_causal) with Q:(batch, Nq, d),
K/V:(batch, Nk, d). The autograd.Function saves (Q, K, V, O, L) where L is the
per-query log-sum-exp of shape (batch, Nq); the tests fish L out of saved_tensors.
Causal masking adds -1e6 to masked scores (matching the reference).
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


# --------------------------------------------------------------------------- #
# Shared backward (recomputation). Plain PyTorch + torch.compile, no Triton.
# --------------------------------------------------------------------------- #
def _flash_backward(Q, K, V, O, dO, L, is_causal: bool):
    # Compute in fp32 for numerical stability and to avoid mixed-dtype matmuls
    # (L is fp32 while Q/K/V may be bf16); cast grads back to the input dtype.
    in_dtype = Q.dtype
    scale = 1.0 / math.sqrt(Q.shape[-1])
    Qf, Kf, Vf, Of, dOf = (t.float() for t in (Q, K, V, O, dO))
    S = torch.einsum("bqd,bkd->bqk", Qf, Kf) * scale
    if is_causal:
        nq, nk = Q.shape[-2], K.shape[-2]
        q_idx = torch.arange(nq, device=Q.device)[:, None]
        k_idx = torch.arange(nk, device=Q.device)[None, :]
        S = torch.where(q_idx >= k_idx, S, torch.full_like(S, -1e6))
    # Recover softmax probabilities from the saved log-sum-exp.
    P = torch.exp(S - L[..., None])                      # (b, nq, nk)
    dV = torch.einsum("bqk,bqd->bkd", P, dOf)            # (b, nk, d)
    dP = torch.einsum("bqd,bkd->bqk", dOf, Vf)           # (b, nq, nk)
    D = torch.sum(dOf * Of, dim=-1)                      # (b, nq)
    dS = P * (dP - D[..., None])                         # (b, nq, nk)
    dQ = torch.einsum("bqk,bkd->bqd", dS, Kf) * scale
    dK = torch.einsum("bqk,bqd->bkd", dS, Qf) * scale
    return dQ.to(in_dtype), dK.to(in_dtype), dV.to(in_dtype)


_flash_backward_compiled = torch.compile(_flash_backward)


# --------------------------------------------------------------------------- #
# Pure-PyTorch tiled forward (Algorithm 1) — reference to debug the Triton kernel.
# --------------------------------------------------------------------------- #
class FlashAttentionPyTorch(torch.autograd.Function):
    Q_TILE = 16
    K_TILE = 16

    @staticmethod
    def forward(ctx, Q, K, V, is_causal: bool = False):
        B, Nq, d = Q.shape
        Nk = K.shape[1]
        Bq, Bk = FlashAttentionPyTorch.Q_TILE, FlashAttentionPyTorch.K_TILE
        scale = 1.0 / math.sqrt(d)
        O = torch.empty_like(Q)
        L = torch.empty(B, Nq, device=Q.device, dtype=torch.float32)

        for i in range(0, Nq, Bq):
            Qi = Q[:, i:i + Bq]                                  # (B, Bq, d)
            Oi = torch.zeros(B, Qi.shape[1], d, device=Q.device, dtype=torch.float32)
            li = torch.zeros(B, Qi.shape[1], device=Q.device, dtype=torch.float32)
            mi = torch.full((B, Qi.shape[1]), float("-inf"), device=Q.device, dtype=torch.float32)
            for j in range(0, Nk, Bk):
                Kj = K[:, j:j + Bk]
                Vj = V[:, j:j + Bk]
                Sij = torch.einsum("bqd,bkd->bqk", Qi, Kj).float() * scale   # (B, Bq, Bk)
                if is_causal:
                    q_idx = (i + torch.arange(Qi.shape[1], device=Q.device))[:, None]
                    k_idx = (j + torch.arange(Kj.shape[1], device=Q.device))[None, :]
                    Sij = Sij + torch.where(q_idx >= k_idx, 0.0, -1e6)
                m_new = torch.maximum(mi, Sij.max(dim=-1).values)            # (B, Bq)
                P = torch.exp(Sij - m_new[..., None])                        # (B, Bq, Bk)
                corr = torch.exp(mi - m_new)                                 # (B, Bq)
                li = corr * li + P.sum(dim=-1)
                Oi = corr[..., None] * Oi + torch.einsum("bqk,bkd->bqd", P, Vj.float())
                mi = m_new
            Oi = Oi / li[..., None]
            O[:, i:i + Bq] = Oi.to(O.dtype)
            L[:, i:i + Bq] = mi + torch.log(li)

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        dQ, dK, dV = _flash_backward_compiled(Q, K, V, O, dO, L, ctx.is_causal)
        return dQ, dK, dV, None


# --------------------------------------------------------------------------- #
# Triton forward kernel (Algorithm 1).
# --------------------------------------------------------------------------- #
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb, shape=(N_QUERIES, D), strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0), block_shape=(Q_TILE_SIZE, D), order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb, shape=(N_KEYS, D), strides=(stride_kk, stride_kd),
        offsets=(0, 0), block_shape=(K_TILE_SIZE, D), order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb, shape=(N_KEYS, D), strides=(stride_vk, stride_vd),
        offsets=(0, 0), block_shape=(K_TILE_SIZE, D), order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob, shape=(N_QUERIES, D), strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0), block_shape=(Q_TILE_SIZE, D), order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb, shape=(N_QUERIES,), strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,), block_shape=(Q_TILE_SIZE,), order=(0,),
    )

    Qi = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")  # (Bq, D)
    Oi = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    mi = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)

    q_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

    n_k_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
    for j in range(n_k_tiles):
        Kj = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")   # (Bk, D)
        Vj = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")   # (Bk, D)
        Sij = tl.dot(Qi, tl.trans(Kj)) * scale                                  # (Bq, Bk)
        if is_causal:
            k_offsets = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            mask = q_offsets[:, None] >= k_offsets[None, :]
            Sij = Sij + tl.where(mask, 0.0, -1e6)
        m_new = tl.maximum(mi, tl.max(Sij, axis=-1))                            # (Bq,)
        P = tl.exp(Sij - m_new[:, None])                                        # (Bq, Bk)
        corr = tl.exp(mi - m_new)                                               # (Bq,)
        li = corr * li + tl.sum(P, axis=-1)
        Oi = Oi * corr[:, None] + tl.dot(P.to(Vj.dtype), Vj)
        mi = m_new
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    Oi = Oi / li[:, None]
    Li = mi + tl.log(li)
    tl.store(O_block_ptr, Oi.to(O_block_ptr.type.element_ty), boundary_check=(0,))
    tl.store(L_block_ptr, Li, boundary_check=(0,))


class FlashAttentionTriton(torch.autograd.Function):
    Q_TILE = 16
    K_TILE = 16

    @staticmethod
    def forward(ctx, Q, K, V, is_causal: bool = False):
        B, Nq, d = Q.shape
        Nk = K.shape[1]
        Bq, Bk = FlashAttentionTriton.Q_TILE, FlashAttentionTriton.K_TILE
        scale = 1.0 / math.sqrt(d)
        O = torch.empty_like(Q)
        L = torch.empty(B, Nq, device=Q.device, dtype=torch.float32)
        grid = (triton.cdiv(Nq, Bq), B)
        flash_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            Nq, Nk, scale,
            D=d, Q_TILE_SIZE=Bq, K_TILE_SIZE=Bk, is_causal=is_causal,
        )
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        dQ, dK, dV = _flash_backward_compiled(Q, K, V, O, dO, L, ctx.is_causal)
        return dQ, dK, dV, None
