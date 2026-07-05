"""BPE tokenizer training (CS336 Assignment 1, section 2)."""

from __future__ import annotations

import os
from collections import Counter
from multiprocessing import Pool
from typing import BinaryIO

import regex as re

# GPT-2 pre-tokenization pattern (Radford et al., 2019).
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_PAT_RE = re.compile(PAT)


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """Chunk the file into parts that can be counted independently.

    Boundaries are aligned to occurrences of `split_special_token` so no
    pre-token ever straddles two chunks.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def _pretokenize_and_count(text: str, special_tokens: list[str]) -> Counter[bytes]:
    """Count pre-token frequencies in `text`. Special tokens are stripped out
    first so that no merge can ever cross a document boundary."""
    if special_tokens:
        # Sort by length (longest first) so overlapping specials split greedily.
        specials = sorted(special_tokens, key=len, reverse=True)
        split_re = re.compile("|".join(re.escape(s) for s in specials))
        segments = split_re.split(text)
    else:
        segments = [text]

    counts: Counter[bytes] = Counter()
    for segment in segments:
        for match in _PAT_RE.finditer(segment):
            counts[match.group().encode("utf-8")] += 1
    return counts


def _count_chunk(args: tuple[str, int, int, list[str]]) -> Counter[bytes]:
    """Worker: read [start, end) of the file and count pre-tokens."""
    path, start, end, special_tokens = args
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    return _pretokenize_and_count(text, special_tokens)


def _pretokenize_file(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_processes: int | None = None,
) -> Counter[bytes]:
    """Pre-tokenize the whole file, in parallel for large files."""
    file_size = os.path.getsize(input_path)
    if num_processes is None:
        # Parallelism only pays off for large files.
        num_processes = min(os.cpu_count() or 1, 32) if file_size > 8 * 1024 * 1024 else 1

    if num_processes <= 1:
        with open(input_path, "rb") as f:
            text = f.read().decode("utf-8", errors="ignore")
        return _pretokenize_and_count(text, special_tokens)

    split_token = (special_tokens[0] if special_tokens else "\n").encode("utf-8")
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_token)

    jobs = [
        (str(input_path), start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    total: Counter[bytes] = Counter()
    with Pool(processes=len(jobs)) as pool:
        for counts in pool.imap_unordered(_count_chunk, jobs):
            total.update(counts)
    return total


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer.

    Returns:
        vocab: mapping from token ID to token bytes.
        merges: list of merges in creation order.
    """
    # --- 1. Pre-tokenize the corpus into word frequencies -------------------
    word_counts = _pretokenize_file(input_path, special_tokens, num_processes)

    # Each unique word is a list of symbols (bytes); merges rewrite these lists.
    words: list[list[bytes]] = []
    freqs: list[int] = []
    for word, freq in word_counts.items():
        words.append([bytes([b]) for b in word])
        freqs.append(freq)

    # --- 2. Build initial pair statistics -----------------------------------
    # pair_counts: total frequency of each adjacent symbol pair.
    # pair_to_words: which word indices currently contain the pair (superset;
    # stale entries are skipped during the merge).
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = {}
    for idx, (symbols, freq) in enumerate(zip(words, freqs)):
        for a, b in zip(symbols, symbols[1:]):
            pair = (a, b)
            pair_counts[pair] += freq
            pair_to_words.setdefault(pair, set()).add(idx)

    # --- 3. Iteratively merge the best pair ---------------------------------
    num_merges = vocab_size - 256 - len(special_tokens)
    merges: list[tuple[bytes, bytes]] = []

    for _ in range(max(num_merges, 0)):
        if not pair_counts:
            break
        # Most frequent pair; ties broken by lexicographically greater pair.
        best = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        if pair_counts[best] <= 0:
            break
        merges.append(best)
        new_symbol = best[0] + best[1]

        affected = pair_to_words.pop(best, set())
        del pair_counts[best]

        for idx in affected:
            symbols = words[idx]
            freq = freqs[idx]
            i = 0
            while i < len(symbols) - 1:
                if symbols[i] == best[0] and symbols[i + 1] == best[1]:
                    # Decrement counts of the neighbouring pairs we destroy.
                    if i > 0:
                        left = (symbols[i - 1], symbols[i])
                        pair_counts[left] -= freq
                        if pair_counts[left] <= 0:
                            del pair_counts[left]
                    if i + 2 < len(symbols):
                        right = (symbols[i + 1], symbols[i + 2])
                        pair_counts[right] -= freq
                        if pair_counts[right] <= 0:
                            del pair_counts[right]
                    # Perform the merge.
                    symbols[i : i + 2] = [new_symbol]
                    # Increment counts of the newly created neighbouring pairs.
                    if i > 0:
                        left = (symbols[i - 1], new_symbol)
                        pair_counts[left] += freq
                        pair_to_words.setdefault(left, set()).add(idx)
                    if i + 1 < len(symbols):
                        right = (new_symbol, symbols[i + 1])
                        pair_counts[right] += freq
                        pair_to_words.setdefault(right, set()).add(idx)
                else:
                    i += 1

    # --- 4. Assemble the vocabulary ------------------------------------------
    vocab: dict[int, bytes] = {}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    for b in range(256):
        vocab[len(vocab)] = bytes([b])
    for a, b in merges:
        vocab[len(vocab)] = a + b

    return vocab, merges
