"""Problem (leaderboard): optimized full training step (forward + loss + backward
+ AdamW) for the 8B config, timed with triton.testing.do_bench.

Optimizations wired in (toggle with flags):
  --flash        : swap naive attention for our Triton FlashAttention-2 (causal),
                   removing the O(S^2) score matrix so ctx=32768 is feasible.
  --checkpoint   : activation-checkpoint each TransformerBlock (trade compute for memory).
  --compile      : torch.compile the model.
  bf16 autocast is always on (leaderboard runs at BF16 + causal).

The official leaderboard measures the exact 8B Config on 2x B200. On our A800s the
full config's AdamW state (~91 GB) doesn't fit on one GPU, so --num-layers scales
the depth down to a feasible size for local measurement; --no-optim times only
forward+backward (which fits the full depth with flash+checkpointing).
"""

from __future__ import annotations

import argparse

import torch
import triton

import cs336_basics.model as basics_model
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.flash_attention import FlashAttentionTriton


class Config:
    ctx_len = 32768
    vocab_size = 151936
    d_model = 4096
    d_ff = 11008
    num_layers = 34
    num_heads = 32
    batch_size = 2


def flash_sdpa(Q, K, V, mask=None):
    """Drop-in for scaled_dot_product_attention: (…, heads, seq, d) causal flash."""
    *lead, s, d = Q.shape
    Qr, Kr, Vr = (t.reshape(-1, t.shape[-2], d) for t in (Q, K, V))
    out = FlashAttentionTriton.apply(Qr, Kr, Vr, True)
    return out.reshape(*lead, s, d)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-layers", type=int, default=Config.num_layers)
    p.add_argument("--ctx-len", type=int, default=Config.ctx_len)
    p.add_argument("--batch-size", type=int, default=Config.batch_size)
    p.add_argument("--flash", action="store_true")
    p.add_argument("--checkpoint", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--no-optim", action="store_true", help="time forward+backward only")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if args.flash:
        basics_model.scaled_dot_product_attention = flash_sdpa

    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    model = BasicsTransformerLM(
        vocab_size=Config.vocab_size, context_length=args.ctx_len, d_model=Config.d_model,
        num_layers=args.num_layers, num_heads=Config.num_heads, d_ff=Config.d_ff,
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params {n_params/1e9:.2f}B, layers {args.num_layers}, ctx {args.ctx_len}", flush=True)

    if args.checkpoint:
        from torch.utils.checkpoint import checkpoint
        for blk in model.layers:
            fwd = blk.forward
            blk.forward = (lambda f: (lambda *a, **k: checkpoint(f, *a, use_reentrant=False, **k)))(fwd)
    if args.compile:
        model = torch.compile(model)

    optimizer = None if args.no_optim else AdamW(model.parameters())
    x = torch.randint(0, Config.vocab_size, (args.batch_size, args.ctx_len), device=args.device)
    y = torch.randint(0, Config.vocab_size, (args.batch_size, args.ctx_len), device=args.device)

    def train_step():
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = cross_entropy(model(x), y)
        loss.backward()
        if optimizer is not None:
            optimizer.step()

    try:
        ms = triton.testing.do_bench(train_step, warmup=200, rep=1000)
        peak = torch.cuda.max_memory_allocated() / 1024**3
        mode = "fwd+bwd" if args.no_optim else "full step"
        print(f"[{mode}] {ms:.1f} ms/step  peak {peak:.1f} GB  "
              f"(flash={args.flash} ckpt={args.checkpoint} compile={args.compile})", flush=True)
    except torch.OutOfMemoryError:
        print("OOM", flush=True)


if __name__ == "__main__":
    main()
