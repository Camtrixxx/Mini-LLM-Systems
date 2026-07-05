"""Plot loss curves from one or more runs' log.jsonl files.

Usage:
    python scripts/plot_curves.py --runs out/runs/ts-lr-1e-3 out/runs/ts-lr-3e-3 \
        --output out/plots/lr_sweep.png [--x wall] [--train]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_log(run_dir: Path) -> tuple[list[dict], list[dict]]:
    train, evals = [], []
    with open(run_dir / "log.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            if rec["type"] == "train":
                train.append(rec)
            elif rec["type"] == "eval":
                evals.append(rec)
    return train, evals


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--x", choices=["step", "wall"], default="step")
    p.add_argument("--train", action="store_true", help="plot train loss instead of val")
    p.add_argument("--ymax", type=float, default=None)
    p.add_argument("--title", default=None)
    args = p.parse_args()

    plt.figure(figsize=(8, 5))
    for run in args.runs:
        run_dir = Path(run)
        train, evals = load_log(run_dir)
        recs = train if args.train else evals
        key = "loss" if args.train else "val_loss"
        xs = [r[args.x] for r in recs]
        ys = [r[key] for r in recs]
        plt.plot(xs, ys, label=run_dir.name, alpha=0.9)

    plt.xlabel("step" if args.x == "step" else "wall-clock (s)")
    plt.ylabel("train loss" if args.train else "validation loss")
    if args.ymax:
        plt.ylim(top=args.ymax)
    plt.legend()
    plt.grid(alpha=0.3)
    if args.title:
        plt.title(args.title)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
