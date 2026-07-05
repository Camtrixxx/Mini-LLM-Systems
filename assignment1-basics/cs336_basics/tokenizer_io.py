"""Serialize/deserialize BPE vocab and merges in GPT-2 style.

Uses the GPT-2 printable-unicode remapping of bytes so the files are
human-inspectable text (same format as the course test fixtures).
"""

from __future__ import annotations

import json
import os


def gpt2_bytes_to_unicode() -> dict[int, str]:
    """GPT-2's invertible mapping from byte values to printable unicode chars."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


_B2U = gpt2_bytes_to_unicode()
_U2B = {u: b for b, u in _B2U.items()}


def bytes_to_str(data: bytes) -> str:
    return "".join(_B2U[b] for b in data)


def str_to_bytes(s: str) -> bytes:
    return bytes(_U2B[c] for c in s)


def save_vocab_merges(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
) -> None:
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(
            {bytes_to_str(token): idx for idx, token in vocab.items()},
            f,
            ensure_ascii=False,
            indent=0,
        )
    with open(merges_path, "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{bytes_to_str(a)} {bytes_to_str(b)}\n")


def load_vocab_merges(
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(vocab_path, encoding="utf-8") as f:
        raw = json.load(f)
    vocab = {idx: str_to_bytes(token) for token, idx in raw.items()}
    merges = []
    with open(merges_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line and len(line.split(" ")) == 2:
                a, b = line.split(" ")
                merges.append((str_to_bytes(a), str_to_bytes(b)))
    return vocab, merges
