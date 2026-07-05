"""Encode a text corpus into a uint16 numpy array of token IDs, in parallel.

Usage:
    python scripts/encode_dataset.py \
        --tokenizer-dir out/tokenizers/tinystories-10k \
        --input data/TinyStoriesV2-GPT4-train.txt \
        --output out/data/tinystories_train.npy
"""

from __future__ import annotations

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from cs336_basics.bpe import find_chunk_boundaries

_TOKENIZER = None
_ARGS = None


def _init_worker(tokenizer_dir: str, special_tokens: list[str]) -> None:
    global _TOKENIZER
    from cs336_basics.tokenizer import Tokenizer

    _TOKENIZER = Tokenizer.from_files(
        f"{tokenizer_dir}/vocab.json", f"{tokenizer_dir}/merges.txt", special_tokens
    )


def _encode_chunk(job: tuple[str, int, int]) -> np.ndarray:
    path, start, end = job
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    return np.array(_TOKENIZER.encode(text), dtype=np.uint16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"])
    parser.add_argument("--num-processes", type=int, default=64)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.input, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, args.num_processes * 4, args.special_tokens[0].encode("utf-8")
        )
    jobs = [(args.input, s, e) for s, e in zip(boundaries[:-1], boundaries[1:])]

    t0 = time.perf_counter()
    with Pool(
        processes=args.num_processes,
        initializer=_init_worker,
        initargs=(args.tokenizer_dir, args.special_tokens),
    ) as pool:
        parts = pool.map(_encode_chunk, jobs)  # order-preserving
    ids = np.concatenate(parts)
    elapsed = time.perf_counter() - t0

    assert ids.max() < 2**16, "vocab does not fit in uint16"
    np.save(args.output, ids)
    n_bytes = Path(args.input).stat().st_size
    print(
        f"{args.input}: {n_bytes:,} bytes -> {len(ids):,} tokens "
        f"(compression {n_bytes / len(ids):.2f} bytes/token) in {elapsed:.0f}s "
        f"({n_bytes / elapsed / 1e6:.1f} MB/s aggregate)"
    )


if __name__ == "__main__":
    main()
