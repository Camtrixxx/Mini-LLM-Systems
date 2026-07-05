"""Tokenizer experiments (assignment problem `tokenizer_experiments`).

Computes compression ratios on sampled documents and single-process throughput.
"""

from __future__ import annotations

import json
import time

from cs336_basics.tokenizer import Tokenizer

SPECIALS = ["<|endoftext|>"]


def sample_documents(path: str, n: int = 10, min_len: int = 200) -> list[str]:
    """Read documents from the head of the corpus (delimited by <|endoftext|>)."""
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read(20_000_000)
    docs = [d for d in text.split("<|endoftext|>") if len(d) >= min_len]
    return docs[:n]


def compression_ratio(tokenizer: Tokenizer, docs: list[str]) -> float:
    total_bytes = sum(len(d.encode("utf-8")) for d in docs)
    total_tokens = sum(len(tokenizer.encode(d)) for d in docs)
    return total_bytes / total_tokens


def main() -> None:
    ts_tok = Tokenizer.from_files(
        "out/tokenizers/tinystories-10k/vocab.json",
        "out/tokenizers/tinystories-10k/merges.txt",
        SPECIALS,
    )
    owt_tok = Tokenizer.from_files(
        "out/tokenizers/owt-32k/vocab.json",
        "out/tokenizers/owt-32k/merges.txt",
        SPECIALS,
    )

    ts_docs = sample_documents("data/TinyStoriesV2-GPT4-valid.txt")
    owt_docs = sample_documents("data/owt_valid.txt")

    results = {
        "ts_tok_on_ts": round(compression_ratio(ts_tok, ts_docs), 3),
        "owt_tok_on_owt": round(compression_ratio(owt_tok, owt_docs), 3),
        "ts_tok_on_owt": round(compression_ratio(ts_tok, owt_docs), 3),
        "owt_tok_on_ts": round(compression_ratio(owt_tok, ts_docs), 3),
    }

    # Single-process throughput on ~50MB of OWT text.
    with open("data/owt_valid.txt", encoding="utf-8", errors="ignore") as f:
        sample = f.read(50_000_000)
    t0 = time.perf_counter()
    owt_tok.encode(sample)
    elapsed = time.perf_counter() - t0
    mb_per_s = len(sample.encode("utf-8")) / elapsed / 1e6
    results["throughput_mb_per_s_single_proc"] = round(mb_per_s, 2)
    results["pile_825gb_hours_single_proc"] = round(825_000 / mb_per_s / 3600, 1)
    results["pile_825gb_hours_96_proc"] = round(825_000 / (mb_per_s * 96) / 3600, 2)

    print(json.dumps(results, indent=2))
    with open("out/tokenizer_experiments.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
