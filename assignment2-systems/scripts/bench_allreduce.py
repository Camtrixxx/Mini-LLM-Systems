"""Problem (distributed_communication_single_node): benchmark all-reduce latency
over float32 tensors of {1,10,100,1000} MB. Launch with torchrun; world_size is
taken from the environment (run once each for --nproc_per_node 2, 4, 6).
"""

from __future__ import annotations

import json
import os
import timeit

import torch
import torch.distributed as dist

SIZES_MB = [1, 10, 100, 1000]


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = f"cuda:{local}"

    for mb in SIZES_MB:
        n = mb * 1024 * 1024 // 4  # float32 elements
        x = torch.randn(n, device=dev)
        for _ in range(5):  # warmup
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dist.barrier()
        iters = 30
        t0 = timeit.default_timer()
        for _ in range(iters):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dist.barrier()
        ms = (timeit.default_timer() - t0) / iters * 1000
        if rank == 0:
            print(json.dumps({"world_size": ws, "size_mb": mb, "ms": round(ms, 4)}), flush=True)
        del x
        torch.cuda.empty_cache()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
