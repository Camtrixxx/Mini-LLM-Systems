"""Problem (flash_benchmarking): compare Triton FlashAttention-2 vs PyTorch
attention with triton.testing.do_bench. batch=1, causal, seq 128..65536 (pow2),
dim 16..128 (pow2), dtypes {bf16, fp32}. Reports forward / backward / end-to-end
latency (ms) for both. backward ~= e2e - forward.
"""

from __future__ import annotations

import argparse
import itertools
import json

import torch
import triton
from einops import einsum

from cs336_systems.flash_attention import FlashAttentionTriton

SEQS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
DIMS = [16, 32, 64, 128]
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def pytorch_attn_causal(q, k, v):
    d = q.shape[-1]
    s = einsum(q, k, "b q d, b k d -> b q k") / (d ** 0.5)
    nq, nk = q.shape[-2], k.shape[-2]
    mask = torch.arange(nq, device=q.device)[:, None] >= torch.arange(nk, device=q.device)[None, :]
    s = s.masked_fill(~mask, -1e6)
    p = torch.softmax(s, dim=-1)
    return einsum(p, v, "b q k, b k d -> b q d")


def timed(fn) -> float | None:
    try:
        return triton.testing.do_bench(fn, warmup=10, rep=50)
    except torch.OutOfMemoryError:
        return None
    except Exception:
        return None


def bench(impl, q, k, v, do) -> dict:
    def fwd():
        return impl(q, k, v)

    def e2e():
        q.grad = k.grad = v.grad = None
        impl(q, k, v).backward(do)

    fwd_ms = timed(fwd)
    e2e_ms = timed(e2e)
    torch.cuda.empty_cache()
    bwd_ms = (e2e_ms - fwd_ms) if (fwd_ms is not None and e2e_ms is not None) else None
    return {"fwd_ms": fwd_ms, "bwd_ms": bwd_ms, "e2e_ms": e2e_ms}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="out/flash_benchmarking.jsonl")
    args = p.parse_args()
    flash = lambda q, k, v: FlashAttentionTriton.apply(q, k, v, True)  # causal
    with open(args.out, "w") as f:
        for dtype_name, dtype in DTYPES.items():
            for d, seq in itertools.product(DIMS, SEQS):
                torch.cuda.empty_cache()
                try:
                    q = torch.randn(1, seq, d, device=args.device, dtype=dtype, requires_grad=True)
                    k = torch.randn(1, seq, d, device=args.device, dtype=dtype, requires_grad=True)
                    v = torch.randn(1, seq, d, device=args.device, dtype=dtype, requires_grad=True)
                    do = torch.randn(1, seq, d, device=args.device, dtype=dtype)
                except torch.OutOfMemoryError:
                    continue
                row = {"dtype": dtype_name, "d": d, "seq": seq}
                row["triton"] = bench(flash, q, k, v, do)
                row["pytorch"] = bench(pytorch_attn_causal, q, k, v, do)
                f.write(json.dumps(row) + "\n"); f.flush()
                t, pt = row["triton"], row["pytorch"]
                print(f"[{dtype_name}] d={d:>3} seq={seq:>5} | "
                      f"triton fwd={t['fwd_ms']} e2e={t['e2e_ms']} | "
                      f"pytorch fwd={pt['fwd_ms']} e2e={pt['e2e_ms']}", flush=True)


if __name__ == "__main__":
    main()
