"""Problem (pytorch_attention) + (torch_compile)(a): benchmark plain scaled
dot-product attention (no multihead) across d_model × seq_len, forward and
backward, compiled vs uncompiled, recording memory-before-backward and OOMs.

batch=8, d in {16,32,64,128}, seq in {256,1024,4096,8192,16384}.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json

import torch

from cs336_basics.model import scaled_dot_product_attention

BATCH = 8
DIMS = [16, 32, 64, 128]
SEQS = [256, 1024, 4096, 8192, 16384]


def bench_one(d: int, seq: int, compiled: bool, device: str, n: int = 100, warmup: int = 5) -> dict:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        q = torch.randn(BATCH, seq, d, device=device, requires_grad=True)
        k = torch.randn(BATCH, seq, d, device=device, requires_grad=True)
        v = torch.randn(BATCH, seq, d, device=device, requires_grad=True)
        attn = torch.compile(scaled_dot_product_attention) if compiled else scaled_dot_product_attention

        def fwd():
            return attn(q, k, v)

        for _ in range(warmup):
            fwd().sum().backward()
        torch.cuda.synchronize()

        # Forward timing (100 passes). Discard each output so we don't retain 100
        # autograd graphs (which would balloon memory and cause spurious OOMs).
        import timeit
        t0 = timeit.default_timer()
        for _ in range(n):
            o = fwd()
            del o
        torch.cuda.synchronize()
        fwd_ms = (timeit.default_timer() - t0) / n * 1000

        # Memory in use just before backward.
        o = fwd()
        loss = o.sum()
        torch.cuda.synchronize()
        mem_before_bwd = torch.cuda.memory_allocated() / 1024**3

        # Backward timing (100 passes); rebuild graph each time.
        t0 = timeit.default_timer()
        for _ in range(n):
            q.grad = k.grad = v.grad = None
            attn(q, k, v).sum().backward()
        torch.cuda.synchronize()
        bwd_ms = (timeit.default_timer() - t0) / n * 1000

        return {"d": d, "seq": seq, "compiled": compiled,
                "fwd_ms": round(fwd_ms, 3), "bwd_ms": round(bwd_ms, 3),
                "mem_before_bwd_gb": round(mem_before_bwd, 3)}
    except torch.OutOfMemoryError:
        return {"d": d, "seq": seq, "compiled": compiled, "oom": True}
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="out/pytorch_attention.jsonl")
    args = p.parse_args()
    with open(args.out, "w") as f:
        for compiled in (False, True):
            for d, seq in itertools.product(DIMS, SEQS):
                r = bench_one(d, seq, compiled, args.device)
                f.write(json.dumps(r) + "\n")
                f.flush()
                tag = "OOM" if r.get("oom") else f"fwd {r['fwd_ms']}ms bwd {r['bwd_ms']}ms mem {r['mem_before_bwd_gb']}GB"
                print(f"[{'compiled' if compiled else 'eager':>8}] d={d:>3} seq={seq:>5}: {tag}", flush=True)


if __name__ == "__main__":
    main()
