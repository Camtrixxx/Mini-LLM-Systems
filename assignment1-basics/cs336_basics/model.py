"""Transformer language model components (CS336 Assignment 1, section 3)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import einsum, rearrange
from torch import Tensor


# --------------------------------------------------------------------------- #
# Basic building blocks
# --------------------------------------------------------------------------- #


class Linear(nn.Module):
    """Linear transform y = W x (no bias). Weight is stored as (d_out, d_in)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Tensor) -> Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    """Token embedding lookup. Weight is stored as (num_embeddings, embedding_dim)."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean subtraction, no bias)."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)  # upcast to avoid overflow in x^2
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms * self.weight).to(in_dtype)


def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


def softmax(x: Tensor, dim: int) -> Tensor:
    """Numerically stable softmax along `dim`."""
    x = x - x.amax(dim=dim, keepdim=True)
    exp = torch.exp(x)
    return exp / exp.sum(dim=dim, keepdim=True)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network: w2(SiLU(w1 x) * w3 x)."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class SiLUFFN(nn.Module):
    """Ungated FFN baseline for the SwiGLU ablation: w2(SiLU(w1 x))."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)))


# --------------------------------------------------------------------------- #
# Positional embeddings and attention
# --------------------------------------------------------------------------- #


class RotaryPositionalEmbedding(nn.Module):
    """RoPE (Su et al., 2021). Pre-caches cos/sin tables up to `max_seq_len`.

    Dimension pairs are (x_{2i}, x_{2i+1}) with frequency theta^{-2i/d_k}.
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert d_k % 2 == 0, "RoPE dimension must be even"
        inv_freq = theta ** (
            -torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k
        )
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = einsum(positions, inv_freq, "seq, half -> seq half")
        # persistent=False: these are derived caches, not model weights.
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        """x: (..., seq_len, d_k); token_positions: (..., seq_len)."""
        cos = self.cos[token_positions]  # (..., seq, d_k / 2)
        sin = self.sin[token_positions]
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rotated = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
        return rearrange(rotated, "... half two -> ... (half two)")


def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V.

    mask is boolean with True = "may attend".
    """
    d_k = Q.shape[-1]
    scores = einsum(Q, K, "... q d_k, ... k d_k -> ... q k") / math.sqrt(d_k)
    # Mask + softmax in float32: under autocast(bf16) + torch.compile, the
    # fused backward of a bf16 softmax with -inf entries produces NaNs.
    scores = scores.to(torch.float32)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = softmax(scores, dim=-1).to(V.dtype)
    return einsum(weights, V, "... q k, ... k d_v -> ... q d_v")


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention, optionally with RoPE."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope = rope
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        seq_len = x.shape[-2]

        q = rearrange(self.q_proj(x), "... seq (h d) -> ... h seq d", h=self.num_heads)
        k = rearrange(self.k_proj(x), "... seq (h d) -> ... h seq d", h=self.num_heads)
        v = rearrange(self.v_proj(x), "... seq (h d) -> ... h seq d", h=self.num_heads)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            # Insert the head dim so positions broadcast across heads.
            positions = token_positions.unsqueeze(-2)
            q = self.rope(q, positions)
            k = self.rope(k, positions)

        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )
        out = scaled_dot_product_attention(q, k, v, mask=causal_mask)
        out = rearrange(out, "... h seq d -> ... seq (h d)")
        return self.output_proj(out)


# --------------------------------------------------------------------------- #
# Transformer
# --------------------------------------------------------------------------- #


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: x + Attn(LN(x)), then x + FFN(LN(x))."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RotaryPositionalEmbedding | None = None,
        norm_position: str = "pre",  # "pre" | "post" | "none" (ablations)
        ffn_type: str = "swiglu",  # "swiglu" | "silu" (ablations)
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        assert norm_position in ("pre", "post", "none")
        self.norm_position = norm_position
        if norm_position != "none":
            self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
            self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model, num_heads, rope=rope, device=device, dtype=dtype
        )
        ffn_cls = SwiGLU if ffn_type == "swiglu" else SiLUFFN
        self.ffn = ffn_cls(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        if self.norm_position == "pre":
            x = x + self.attn(self.ln1(x), token_positions=token_positions)
            x = x + self.ffn(self.ln2(x))
        elif self.norm_position == "post":
            x = self.ln1(x + self.attn(x, token_positions=token_positions))
            x = self.ln2(x + self.ffn(x))
        else:  # no layer norm
            x = x + self.attn(x, token_positions=token_positions)
            x = x + self.ffn(x)
        return x


class TransformerLM(nn.Module):
    """Decoder-only Transformer language model with RoPE and pre-norm blocks."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        norm_position: str = "pre",
        ffn_type: str = "swiglu",
        use_rope: bool = True,
        tie_embeddings: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        rope = (
            RotaryPositionalEmbedding(
                theta=rope_theta,
                d_k=d_model // num_heads,
                max_seq_len=context_length,
                device=device,
            )
            if use_rope
            else None  # NoPE ablation: no positional information at all
        )
        self.layers = nn.ModuleList(
            TransformerBlock(
                d_model,
                num_heads,
                d_ff,
                rope=rope,
                norm_position=norm_position,
                ffn_type=ffn_type,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
        if tie_embeddings:
            # Weight tying (Vaswani et al., 2017 §3.4): share input/output
            # embeddings; re-init with a smaller std suited to both roles.
            nn.init.trunc_normal_(
                self.token_embeddings.weight, mean=0.0, std=0.02, a=-0.06, b=0.06
            )
            self.lm_head.weight = self.token_embeddings.weight

    def forward(self, token_ids: Tensor) -> Tensor:
        """token_ids: (batch, seq_len) -> logits: (batch, seq_len, vocab_size)."""
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)
