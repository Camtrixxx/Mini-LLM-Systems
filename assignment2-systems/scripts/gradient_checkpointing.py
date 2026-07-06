"""Problem (gradient_checkpointing)(b): peak-memory vs checkpoint block size.

xl model, batch 4, seq_len 2048, single training step. We group the model's N
TransformerBlocks into consecutive chunks of size K and wrap each chunk in a
single (non-nested) torch.utils.checkpoint call. Peak memory is minimized near
K = sqrt(N): larger K stores fewer checkpoints but recomputes/keeps more live
activations per group; smaller K keeps less live but stores more checkpoints.
"""

from __future__ import annotations

import argparse
import json

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.model_configs import MODEL_SIZES


class CheckpointedGroup(nn.Module):
    """Runs a consecutive group of TransformerBlocks under one checkpoint call."""

    def __init__(self, blocks: list[nn.Module]):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def _run(self, x):
        for b in self.blocks:
            x = b(x)
        return x

    def forward(self, x):
        return checkpoint(self._run, x, use_reentrant=False)


def regroup(model: BasicsTransformerLM, group_size: int) -> None:
    layers = list(model.layers)
    groups = [CheckpointedGroup(layers[i:i + group_size]) for i in range(0, len(layers), group_size)]
    model.layers = nn.ModuleList(groups)


def measure(group_size: int | None, ctx: int, batch: int, device: str) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cfg = MODEL_SIZES["xl"]
    model = BasicsTransformerLM(
        vocab_size=10_000, context_length=ctx, d_model=cfg.d_model,
        num_layers=cfg.num_layers, num_heads=cfg.num_heads, d_ff=cfg.d_ff,
    ).to(device)
    n_layers = cfg.num_layers
    if group_size is not None:
        regroup(model, group_size)
    opt = AdamW(model.parameters())
    x = torch.randint(0, 10_000, (batch, ctx), device=device)
    y = torch.randint(0, 10_000, (batch, ctx), device=device)
    try:
        for _ in range(2):  # a couple steps so allocator/optimizer state settle
            loss = cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1024**3
        out = {"group_size": group_size, "n_layers": n_layers, "peak_mem_gb": round(peak, 2)}
    except torch.OutOfMemoryError:
        out = {"group_size": group_size, "n_layers": n_layers, "oom": True}
    del model, opt
    torch.cuda.empty_cache()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    # None = no checkpointing baseline; then group sizes bracketing sqrt(32)~5.66.
    for gs in [None, 1, 2, 4, 6, 8, 16, 32]:
        r = measure(gs, args.ctx, args.batch, args.device)
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
