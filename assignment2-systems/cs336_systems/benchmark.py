"""End-to-end benchmarking / profiling harness for the basics Transformer LM.

Supports (all toggled by CLI flags, so the same script serves every Section 2/3
problem):
  - forward-only / forward+backward / full training step (with AdamW)
  - warm-up steps then timed measurement steps (timeit, cuda.synchronize each step)
  - BF16 mixed precision via torch.autocast
  - torch.cuda memory profiling (dumps a pickle for pytorch.org/memory_viz) and
    peak-memory reporting
  - NVTX ranges around warmup/steps/phases for nsys attribution (+ annotated attn)
  - torch.compile
  - activation (gradient) checkpointing of TransformerBlocks

Examples:
  uv run python -m cs336_systems.benchmark --size small --mode full
  uv run nsys profile -- python -m cs336_systems.benchmark --size xl --context-length 512 --nvtx
  uv run python -m cs336_systems.benchmark --size xl --context-length 2048 --mode full --memory-profile snap.pickle
"""

from __future__ import annotations

import argparse
import contextlib
import statistics
import timeit

import torch
import torch.cuda.nvtx as nvtx

import cs336_basics.model as basics_model
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

from cs336_systems.model_configs import BATCH_SIZE, CONTEXT_LENGTH, MODEL_SIZES, VOCAB_SIZE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", choices=list(MODEL_SIZES), default="small")
    p.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--context-length", type=int, default=CONTEXT_LENGTH)
    p.add_argument("--rope-theta", type=float, default=10000.0)
    p.add_argument("--mode", choices=["forward", "forward_backward", "full"], default="forward_backward")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32", help="bf16 => autocast mixed precision")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--checkpoint", action="store_true", help="activation-checkpoint each TransformerBlock")
    p.add_argument("--nvtx", action="store_true", help="add NVTX ranges + annotated attention (for nsys)")
    p.add_argument("--inference", action="store_true",
                   help="forward mode only: run under no_grad (true inference, no saved activations)")
    p.add_argument("--memory-profile", default=None, metavar="PICKLE",
                   help="record CUDA memory history and dump a snapshot pickle")
    p.add_argument("--json", action="store_true", help="print result as a JSON line")
    return p.parse_args()


def build_model(args) -> BasicsTransformerLM:
    cfg = MODEL_SIZES[args.size]
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        rope_theta=args.rope_theta,
    ).to(args.device)
    if args.checkpoint:
        _wrap_blocks_with_checkpoint(model)
    if args.compile:
        model = torch.compile(model)
    return model


def _wrap_blocks_with_checkpoint(model: BasicsTransformerLM) -> None:
    """Run each TransformerBlock through torch.utils.checkpoint (recompute in backward)."""
    from torch.utils.checkpoint import checkpoint

    for block in model.layers:
        orig_forward = block.forward

        def make(fwd):
            def checkpointed(*a, **kw):
                return checkpoint(fwd, *a, use_reentrant=False, **kw)
            return checkpointed

        block.forward = make(orig_forward)


def run(args) -> dict:
    device = args.device
    if args.nvtx:
        basics_model.scaled_dot_product_attention = (
            __import__("cs336_systems.nvtx_attention", fromlist=["annotated_scaled_dot_product_attention"])
            .annotated_scaled_dot_product_attention
        )

    torch.manual_seed(0)
    model = build_model(args)
    optimizer = AdamW(model.parameters()) if args.mode == "full" else None

    # Random token batch + shifted targets (weights/data are random — we only measure speed/memory).
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    y = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)

    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.dtype == "bf16" else contextlib.nullcontext()
    )

    grad_ctx = torch.no_grad() if (args.inference and args.mode == "forward") else contextlib.nullcontext()
    rng = (lambda name: nvtx.range(name)) if args.nvtx else (lambda name: contextlib.nullcontext())

    def one_step():
        with rng("forward"), grad_ctx, autocast:
            logits = model(x)
            if args.mode == "forward":
                return
            loss = cross_entropy(logits, y)
        with rng("backward"):
            loss.backward()
        if args.mode == "full":
            with rng("optimizer"):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

    def sync():
        if "cuda" in device:
            torch.cuda.synchronize()

    # Warm-up.
    with nvtx.range("warmup") if args.nvtx else contextlib.nullcontext():
        for _ in range(args.warmup):
            one_step()
            sync()

    if args.memory_profile:
        torch.cuda.memory._record_memory_history(max_entries=1_000_000)

    # Timed measurement.
    per_step = []
    with nvtx.range("measure") if args.nvtx else contextlib.nullcontext():
        for i in range(args.steps):
            with nvtx.range(f"step_{i}") if args.nvtx else contextlib.nullcontext():
                t0 = timeit.default_timer()
                one_step()
                sync()
                per_step.append(timeit.default_timer() - t0)

    result = {
        "size": args.size,
        "mode": args.mode,
        "dtype": args.dtype,
        "context_length": args.context_length,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "warmup": args.warmup,
        "mean_s": statistics.mean(per_step),
        "std_s": statistics.pstdev(per_step) if len(per_step) > 1 else 0.0,
    }
    if "cuda" in device:
        result["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1024**3

    if args.memory_profile:
        torch.cuda.memory._dump_snapshot(args.memory_profile)
        torch.cuda.memory._record_memory_history(enabled=None)
        result["memory_snapshot"] = args.memory_profile

    return result


def main() -> None:
    args = parse_args()
    if "cuda" in args.device:
        torch.cuda.reset_peak_memory_stats()
    r = run(args)
    if args.json:
        import json
        print(json.dumps(r))
        return
    ms = r["mean_s"] * 1000
    std = r["std_s"] * 1000
    line = (f"[{r['size']:>6} {r['mode']:>16} {r['dtype']} ctx={r['context_length']:>5}] "
            f"{ms:8.2f} ms/step  (std {std:6.2f} ms)")
    if "peak_mem_gb" in r:
        line += f"  peak {r['peak_mem_gb']:6.2f} GB"
    print(line)


if __name__ == "__main__":
    main()
