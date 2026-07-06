"""Generate text from a trained checkpoint.

Usage:
    python scripts/generate.py --run-dir out/runs/ts-base \
        --tokenizer-dir out/tokenizers/tinystories-10k \
        --prompt "Once upon a time" --temperature 0.8 --top-p 0.9
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cs336_basics.decoding import generate
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "config.json").read_text())

    tokenizer = Tokenizer.from_files(
        f"{args.tokenizer_dir}/vocab.json",
        f"{args.tokenizer_dir}/merges.txt",
        special_tokens=["<|endoftext|>"],
    )
    eos_id = tokenizer._special_to_id["<|endoftext|>"]

    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
        norm_position=config.get("norm_position", "pre"),
        ffn_type=config.get("ffn_type", "swiglu"),
        use_rope=not config.get("no_rope", False),
        tie_embeddings=config.get("tie_embeddings", False),
        device=args.device,
    )
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location=args.device)
    model.load_state_dict(checkpoint["model"])

    torch.manual_seed(args.seed)
    prompt_ids = tokenizer.encode(args.prompt)
    for i in range(args.num_samples):
        out_ids = generate(
            model,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_id=eos_id,
            device=args.device,
        )
        print(f"--- sample {i} ({len(out_ids)} tokens) ---")
        print(args.prompt + tokenizer.decode(out_ids))


if __name__ == "__main__":
    main()
