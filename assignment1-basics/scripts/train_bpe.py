"""Train a BPE tokenizer on a corpus and serialize vocab/merges.

Usage:
    python scripts/train_bpe.py --input data/TinyStoriesV2-GPT4-train.txt \
        --vocab-size 10000 --output-dir out/tokenizers/tinystories-10k
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

from cs336_basics.bpe import _pretokenize_file, train_bpe_from_counts
from cs336_basics.tokenizer_io import save_vocab_merges


def peak_rss_gb() -> float:
    """Peak RSS of this process + all children, in GB (Linux reports KB)."""
    self_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    child_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return (self_kb + child_kb) / 1024**2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"])
    parser.add_argument("--num-processes", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    word_counts = _pretokenize_file(args.input, args.special_tokens, args.num_processes)
    t1 = time.perf_counter()
    print(f"[pretokenize] {t1 - t0:.1f}s, {len(word_counts):,} unique pre-tokens")

    vocab, merges = train_bpe_from_counts(word_counts, args.vocab_size, args.special_tokens)
    t2 = time.perf_counter()
    print(f"[merges]      {t2 - t1:.1f}s, {len(merges):,} merges")

    save_vocab_merges(vocab, merges, out_dir / "vocab.json", out_dir / "merges.txt")

    longest = max(vocab.values(), key=len)
    stats = {
        "input": args.input,
        "vocab_size": args.vocab_size,
        "special_tokens": args.special_tokens,
        "pretokenize_seconds": round(t1 - t0, 1),
        "merge_seconds": round(t2 - t1, 1),
        "total_seconds": round(t2 - t0, 1),
        "peak_rss_gb": round(peak_rss_gb(), 2),
        "unique_pretokens": len(word_counts),
        "longest_token_bytes": len(longest),
        "longest_token": longest.decode("utf-8", errors="replace"),
        "top_longest_tokens": [
            t.decode("utf-8", errors="replace")
            for t in sorted(vocab.values(), key=len, reverse=True)[:10]
        ],
    }
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
