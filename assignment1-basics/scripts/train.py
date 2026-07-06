"""Train a TransformerLM on a pre-tokenized (uint16 .npy) corpus.

Example (TinyStories base config):
    python scripts/train.py --run-name ts-base \
        --train-data out/data/tinystories_train.npy \
        --val-data out/data/tinystories_valid.npy \
        --vocab-size 10000 --lr 1e-3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy, get_batch, gradient_clipping, save_checkpoint
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule


def ddp_setup() -> tuple[int, int, int, bool]:
    """Init process group if launched under torchrun (WORLD_SIZE>1).

    Returns (rank, local_rank, world_size, is_ddp). Single-GPU runs (no
    torchrun) return (0, 0, 1, False) and behave exactly as before.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return 0, 0, 1, False
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size, True


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Strip torch.compile (_orig_mod) and DDP (module) wrappers for checkpointing."""
    model = getattr(model, "_orig_mod", model)  # torch.compile
    model = getattr(model, "module", model)  # DDP
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--out-dir", default="out/runs")
    p.add_argument("--train-data", required=True)
    p.add_argument("--val-data", required=True)
    # Model (defaults = assignment base config for TinyStories)
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--d-ff", type=int, default=1344)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--rope-theta", type=float, default=10000.0)
    # Ablations
    p.add_argument("--norm-position", choices=["pre", "post", "none"], default="pre")
    p.add_argument("--ffn-type", choices=["swiglu", "silu"], default="swiglu")
    p.add_argument("--no-rope", action="store_true")
    p.add_argument("--tie-embeddings", action="store_true")
    # Optimization
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-iters", type=int, default=10000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=None, help="default lr/10")
    p.add_argument("--warmup-iters", type=int, default=500)
    p.add_argument("--cosine-iters", type=int, default=None, help="default max-iters")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--grad-clip", type=float, default=1.0)
    # Runtime
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=32)
    p.add_argument("--checkpoint-interval", type=int, default=2500)
    p.add_argument("--max-runtime-minutes", type=float, default=None)
    return p.parse_args()


@torch.no_grad()
def evaluate(model, val_data, args, device) -> float:
    """Average loss over `eval_batches` fixed random batches (same every eval)."""
    model.eval()
    rng = np.random.default_rng(1234)
    losses = []
    for _ in range(args.eval_batches):
        starts = rng.integers(0, len(val_data) - args.context_length, args.batch_size)
        x = np.stack([val_data[i : i + args.context_length] for i in starts]).astype(np.int64)
        y = np.stack([val_data[i + 1 : i + 1 + args.context_length] for i in starts]).astype(np.int64)
        x = torch.from_numpy(x).to(device)
        y = torch.from_numpy(y).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled="cuda" in args.device):
            logits = model(x)
            losses.append(cross_entropy(logits, y).item())
    model.train()
    return float(np.mean(losses))


def main() -> None:
    args = parse_args()
    min_lr = args.min_lr if args.min_lr is not None else args.lr / 10
    cosine_iters = args.cosine_iters if args.cosine_iters is not None else args.max_iters

    rank, local_rank, world_size, is_ddp = ddp_setup()
    is_main = rank == 0
    if is_ddp:
        device = f"cuda:{local_rank}"

    run_dir = Path(args.out_dir) / args.run_name
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config.json", "w") as f:
            json.dump({**vars(args), "world_size": world_size}, f, indent=2)
        log_file = open(run_dir / "log.jsonl", "a")
    else:
        log_file = None

    def log(record: dict) -> None:
        if log_file is not None:
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()

    # Distinct seed per rank so each GPU samples different training batches
    # (get_batch draws random offsets from the global torch RNG).
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    if not is_ddp:
        device = args.device
    if "cuda" in device:
        torch.set_float32_matmul_precision("high")  # allow TF32

    train_data = np.load(args.train_data, mmap_mode="r")
    val_data = np.load(args.val_data, mmap_mode="r")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        norm_position=args.norm_position,
        ffn_type=args.ffn_type,
        use_rope=not args.no_rope,
        tie_embeddings=args.tie_embeddings,
        device=device,
    )
    n_params = sum(p.numel() for p in model.parameters())
    if is_main:
        eff_batch = args.batch_size * world_size
        print(f"[{args.run_name}] {n_params / 1e6:.1f}M params, device={device}, "
              f"world_size={world_size}, effective_batch={eff_batch}")
    if is_ddp:
        model = DDP(model, device_ids=[local_rank])
    if args.compile:
        model = torch.compile(model)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    t_start = time.perf_counter()
    running_loss, running_count = 0.0, 0
    best_val = float("inf")
    diverged = False

    for it in range(1, args.max_iters + 1):
        lr = get_lr_cosine_schedule(it, args.lr, min_lr, args.warmup_iters, cosine_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, args.batch_size, args.context_length, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled="cuda" in device):
            logits = model(x)
            loss = cross_entropy(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        running_loss += loss.item()
        running_count += 1

        if it % args.log_interval == 0:
            wall = time.perf_counter() - t_start
            train_loss = running_loss / running_count
            running_loss, running_count = 0.0, 0
            log({"type": "train", "step": it, "wall": round(wall, 1), "loss": round(train_loss, 4), "lr": lr})
            # Both stop conditions are decided here on a lockstep boundary so all
            # ranks break together — a rank that exits early would leave the rest
            # hanging at the next gradient all-reduce.
            bad = not math.isfinite(train_loss) or train_loss > 20
            out_of_time = bool(args.max_runtime_minutes) and wall > args.max_runtime_minutes * 60
            if is_ddp:
                flags = torch.tensor([1.0 if bad else 0.0, 1.0 if out_of_time else 0.0], device=device)
                dist.all_reduce(flags, op=dist.ReduceOp.MAX)
                bad, out_of_time = flags[0].item() > 0, flags[1].item() > 0
            if bad:
                if is_main:
                    print(f"[{args.run_name}] DIVERGED at step {it} (loss={train_loss:.2f})")
                    log({"type": "diverged", "step": it, "loss": train_loss})
                diverged = True
                break
            if out_of_time:
                if is_main:
                    print(f"[{args.run_name}] hit max runtime at step {it}")
                break

        if it % args.eval_interval == 0 or it == args.max_iters:
            # All ranks run eval in lockstep (model.eval()/train() and any DDP
            # buffer sync must stay symmetric); only rank 0 records the result.
            val_loss = evaluate(model, val_data, args, device)
            if is_main:
                wall = time.perf_counter() - t_start
                best_val = min(best_val, val_loss)
                print(f"[{args.run_name}] step {it}/{args.max_iters} wall {wall:.0f}s val_loss {val_loss:.4f}")
                log({"type": "eval", "step": it, "wall": round(wall, 1), "val_loss": round(val_loss, 4)})

        if it % args.checkpoint_interval == 0 or it == args.max_iters:
            if is_main:
                save_checkpoint(unwrap(model), optimizer, it, run_dir / "checkpoint.pt")
            if is_ddp:
                dist.barrier()  # others wait while rank 0 writes the checkpoint

    if is_main:
        if not diverged:
            save_checkpoint(unwrap(model), optimizer, it, run_dir / "checkpoint.pt")
        log({"type": "final", "step": it, "best_val": best_val if best_val < float("inf") else None})
        print(f"[{args.run_name}] done, best val_loss {best_val:.4f}")
        if log_file is not None:
            log_file.close()
    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
