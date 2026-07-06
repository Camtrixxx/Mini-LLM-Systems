"""Problem (fsdp_accounting)(b): peak memory of the xl model on 2 GPUs under FSDP
vs. plain replicated (DDP-style) training. Launch with torchrun --nproc_per_node=2;
--fsdp selects the sharded path.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.ddp import DDPIndividual
from cs336_systems.fsdp import FSDP
from cs336_systems.model_configs import MODEL_SIZES


def gb() -> float:
    return torch.cuda.max_memory_allocated() / 1024**3


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fsdp", action="store_true")
    p.add_argument("--size", default="xl")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--iters", type=int, default=5)
    args = p.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = f"cuda:{local}"
    torch.manual_seed(0)

    cfg = MODEL_SIZES[args.size]
    model = BasicsTransformerLM(
        vocab_size=10_000, context_length=args.context_length, d_model=cfg.d_model,
        num_layers=cfg.num_layers, num_heads=cfg.num_heads, d_ff=cfg.d_ff,
    ).to(dev)
    wrapped = FSDP(model) if args.fsdp else DDPIndividual(model)
    torch.cuda.synchronize()
    mem_after_init = gb()

    opt = AdamW(wrapped.parameters(), lr=1e-3)
    x = torch.randint(0, 10_000, (args.batch_size, args.context_length), device=dev)
    y = torch.randint(0, 10_000, (args.batch_size, args.context_length), device=dev)

    try:
        for _ in range(args.iters):
            opt.zero_grad(set_to_none=True)
            loss = cross_entropy(wrapped(x), y)
            loss.backward()
            wrapped.finish_gradient_synchronization()
            opt.step()
        torch.cuda.synchronize()
        peak = gb()
        res = {"mode": "fsdp" if args.fsdp else "ddp", "size": args.size,
               "mem_after_init_gb": round(mem_after_init, 2), "peak_gb": round(peak, 2)}
    except torch.OutOfMemoryError:
        res = {"mode": "fsdp" if args.fsdp else "ddp", "size": args.size, "oom": True}

    if rank == 0:
        print(json.dumps(res), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
