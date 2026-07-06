"""Problems (naive_ddp_benchmarking / minimal_ddp_flat_benchmarking /
ddp_overlap_benchmarking): time one DDP training step on the xl model and report
per-iteration time and the fraction spent communicating gradients.

Launch with torchrun --nproc_per_node=2. Select the variant with --ddp.
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
from cs336_systems.ddp import DDPFlat, DDPIndividual, DDPNaive
from cs336_systems.model_configs import MODEL_SIZES

VARIANTS = {"naive": DDPNaive, "flat": DDPFlat, "overlap": DDPIndividual}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ddp", choices=list(VARIANTS), default="naive")
    p.add_argument("--size", default="xl")
    p.add_argument("--batch-size", type=int, default=4, help="per-rank batch")
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    args = p.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = f"cuda:{local}"
    torch.manual_seed(0)

    cfg = MODEL_SIZES[args.size]
    model = BasicsTransformerLM(
        vocab_size=10_000, context_length=args.context_length, d_model=cfg.d_model,
        num_layers=cfg.num_layers, num_heads=cfg.num_heads, d_ff=cfg.d_ff,
    ).to(dev)
    ddp = VARIANTS[args.ddp](model)
    opt = AdamW(ddp.parameters())
    x = torch.randint(0, 10_000, (args.batch_size, args.context_length), device=dev)
    y = torch.randint(0, 10_000, (args.batch_size, args.context_length), device=dev)

    def step() -> float:
        opt.zero_grad(set_to_none=True)
        loss = cross_entropy(ddp(x), y)
        loss.backward()
        torch.cuda.synchronize()
        c0 = timeit.default_timer()
        ddp.finish_gradient_synchronization()
        torch.cuda.synchronize()
        comm = timeit.default_timer() - c0
        opt.step()
        torch.cuda.synchronize()
        return comm

    for _ in range(args.warmup):
        step()
    dist.barrier()

    total, comm_total = 0.0, 0.0
    for _ in range(args.iters):
        torch.cuda.synchronize()
        t0 = timeit.default_timer()
        comm = step()
        total += timeit.default_timer() - t0
        comm_total += comm

    if rank == 0:
        iter_ms = total / args.iters * 1000
        comm_ms = comm_total / args.iters * 1000
        print(json.dumps({
            "ddp": args.ddp, "size": args.size, "world_size": ws,
            "batch_per_rank": args.batch_size, "iter_ms": round(iter_ms, 2),
            "comm_ms": round(comm_ms, 2), "comm_pct": round(100 * comm_ms / iter_ms, 1),
        }), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
