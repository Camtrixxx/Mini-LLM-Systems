"""Problem (optimizer_state_sharding_accounting): peak memory and per-iteration
time for xl on 2 GPUs, comparing a sharded optimizer vs. a regular local AdamW.
Reports GPU memory after model init, after backward (before step), and after the
optimizer step. Launch with torchrun --nproc_per_node=2; select --sharded.
"""

from __future__ import annotations

import argparse
import json
import os
import timeit

import torch
import torch.distributed as dist

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.ddp import DDPIndividual
from cs336_systems.model_configs import MODEL_SIZES
from cs336_systems.sharded_optimizer import ShardedOptimizer


def gb() -> float:
    return torch.cuda.max_memory_allocated() / 1024**3


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sharded", action="store_true")
    p.add_argument("--size", default="xl")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--iters", type=int, default=8)
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
    ddp = DDPIndividual(model)
    torch.cuda.synchronize()
    mem_after_init = gb()

    if args.sharded:
        opt = ShardedOptimizer(ddp.parameters(), AdamW, lr=1e-3)
    else:
        opt = AdamW(ddp.parameters(), lr=1e-3)

    x = torch.randint(0, 10_000, (args.batch_size, args.context_length), device=dev)
    y = torch.randint(0, 10_000, (args.batch_size, args.context_length), device=dev)

    mem_before_step = mem_after_step = 0.0
    times = []
    for i in range(args.iters):
        torch.cuda.synchronize()
        t0 = timeit.default_timer()
        opt.zero_grad(set_to_none=True)
        loss = cross_entropy(ddp(x), y)
        loss.backward()
        ddp.finish_gradient_synchronization()
        torch.cuda.synchronize()
        if i == 1:
            mem_before_step = gb()
        opt.step()
        torch.cuda.synchronize()
        if i == 1:
            mem_after_step = gb()
        times.append(timeit.default_timer() - t0)

    if rank == 0:
        warm = times[2:] if len(times) > 2 else times
        print(json.dumps({
            "sharded": args.sharded, "size": args.size,
            "mem_after_init_gb": round(mem_after_init, 2),
            "mem_before_step_gb": round(mem_before_step, 2),
            "mem_after_step_gb": round(mem_after_step, 2),
            "iter_ms": round(sum(warm) / len(warm) * 1000, 1),
        }), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
