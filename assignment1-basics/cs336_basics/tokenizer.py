"""BPE tokenizer for encoding/decoding text (CS336 Assignment 1, section 2)."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator

import regex as re

from cs336_basics.bpe import PAT

_PAT_RE = re.compile(PAT)


class Tokenizer:
    """Byte-level BPE tokenizer.

    Encodes text by (1) splitting out special tokens, (2) pre-tokenizing with
    the GPT-2 regex, and (3) applying BPE merges in creation order.
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.special_tokens = list(special_tokens) if special_tokens else []

        self._bytes_to_id: dict[bytes, int] = {b: i for i, b in self.vocab.items()}
        # Earlier merges have lower rank (higher priority).
        self._merge_ranks: dict[tuple[bytes, bytes], int] = {
            pair: rank for rank, pair in enumerate(merges)
        }

        # Register special tokens, appending any that are missing from vocab.
        self._special_to_id: dict[str, int] = {}
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self._bytes_to_id:
                new_id = max(self.vocab) + 1 if self.vocab else 0
                self.vocab[new_id] = token_bytes
                self._bytes_to_id[token_bytes] = new_id
            self._special_to_id[token] = self._bytes_to_id[token_bytes]

        # Longest-first alternation so overlapping specials match greedily.
        if self.special_tokens:
            specials = sorted(self.special_tokens, key=len, reverse=True)
            self._special_re = re.compile(
                "(" + "|".join(re.escape(s) for s in specials) + ")"
            )
        else:
            self._special_re = None

        # Cache of pre-token bytes -> token IDs (pre-tokens repeat a lot).
        self._encode_cache: dict[bytes, list[int]] = {}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        """Load a tokenizer from serialized vocab (JSON) and merges (text) files."""
        with open(vocab_filepath) as f:
            raw_vocab = json.load(f)
        vocab = {int(i): token.encode("utf-8") for token, i in raw_vocab.items()}
        merges = []
        with open(merges_filepath) as f:
            for line in f:
                line = line.rstrip("\n")
                if line and len(line.split(" ")) == 2:
                    a, b = line.split(" ")
                    merges.append((a.encode("utf-8"), b.encode("utf-8")))
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token IDs."""
        if self._special_re is not None:
            segments = self._special_re.split(text)
        else:
            segments = [text]

        ids: list[int] = []
        for i, segment in enumerate(segments):
            if not segment:
                continue
            # Odd indices are the captured special tokens.
            if i % 2 == 1:
                ids.append(self._special_to_id[segment])
            else:
                for match in _PAT_RE.finditer(segment):
                    ids.extend(self._encode_pretoken(match.group().encode("utf-8")))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode an iterable of strings (e.g. a file handle),
        yielding token IDs without materializing the whole input."""
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back into a string. Invalid UTF-8
        sequences are replaced with U+FFFD."""
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")

    def _encode_pretoken(self, token: bytes) -> list[int]:
        """Apply BPE merges to a single pre-token and return its token IDs."""
        cached = self._encode_cache.get(token)
        if cached is not None:
            return cached

        symbols = [bytes([b]) for b in token]
        while len(symbols) > 1:
            # Find the highest-priority (lowest-rank) applicable merge.
            best_rank = None
            best_idx = -1
            for i in range(len(symbols) - 1):
                rank = self._merge_ranks.get((symbols[i], symbols[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = i
            if best_rank is None:
                break
            symbols[best_idx : best_idx + 2] = [symbols[best_idx] + symbols[best_idx + 1]]

        ids = [self._bytes_to_id[s] for s in symbols]
        self._encode_cache[token] = ids
        return ids
