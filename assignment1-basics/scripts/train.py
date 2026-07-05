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
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy, get_batch, gradient_clipping, save_checkpoint
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule


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

    run_dir = Path(args.out_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    log_file = open(run_dir / "log.jsonl", "a")

    def log(record: dict) -> None:
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
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
        device=device,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.run_name}] {n_params / 1e6:.1f}M params, device={device}")
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
            if not math.isfinite(train_loss) or train_loss > 20:
                print(f"[{args.run_name}] DIVERGED at step {it} (loss={train_loss:.2f})")
                log({"type": "diverged", "step": it, "loss": train_loss})
                diverged = True
                break

        if it % args.eval_interval == 0 or it == args.max_iters:
            val_loss = evaluate(model, val_data, args, device)
            wall = time.perf_counter() - t_start
            best_val = min(best_val, val_loss)
            print(f"[{args.run_name}] step {it}/{args.max_iters} wall {wall:.0f}s val_loss {val_loss:.4f}")
            log({"type": "eval", "step": it, "wall": round(wall, 1), "val_loss": round(val_loss, 4)})

        if it % args.checkpoint_interval == 0 or it == args.max_iters:
            raw = getattr(model, "_orig_mod", model)  # unwrap torch.compile
            save_checkpoint(raw, optimizer, it, run_dir / "checkpoint.pt")

        if args.max_runtime_minutes and (time.perf_counter() - t_start) > args.max_runtime_minutes * 60:
            print(f"[{args.run_name}] hit max runtime at step {it}")
            break

    if not diverged:
        raw = getattr(model, "_orig_mod", model)
        save_checkpoint(raw, optimizer, it, run_dir / "checkpoint.pt")
    log({"type": "final", "step": it, "best_val": best_val if best_val < float("inf") else None})
    print(f"[{args.run_name}] done, best val_loss {best_val:.4f}")
    log_file.close()


if __name__ == "__main__":
    main()
