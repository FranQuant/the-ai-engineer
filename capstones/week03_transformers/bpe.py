"""Byte-pair encoding tokenizer for the BPE instrument (DESIGN.md v0.8, §5).

Ported from v1's SimpleBPE. Changes:
- fit on T bodies only, enforced: from_documents() is the only public way
  to fit and rejects any non-T document;
- the special tokens (<BOS>, <stmt>, <min>) are reserved IDs 0..2 that
  never appear in the merge input, so they are never merged;
- no <unk>: any character outside the base vocabulary fails closed, and
  decode(encode(body)) == body for every encodable body;
- merge ties break deterministically (highest count, then smallest pair),
  so a smaller vocabulary's merge list is a prefix of a larger one's, and
  truncate(k) gives exactly the tokenizer that fit(k) would;
- the end-of-word marker is a private-use character that bodies may not
  contain, so decoding is unambiguous;
- symbol IDs are kept apart from the special IDs (a body spelling "<BOS>"
  gets ordinary symbol IDs), and two merges that build the same string
  share one ID;
- encode applies merges by rank (lowest-rank adjacent pair first, as in
  GPT-2) instead of v1's full pass per merge, with a per-word cache.
"""

from __future__ import annotations

import collections
import re
from typing import Iterable, Sequence

from data import _PRIVATE, SPECIAL_TOKENS, Document, _require_t

END_OF_WORD = ""
WORD_RE = re.compile(r"\S+|\s+")


def _symbols(word: str) -> tuple[str, ...]:
    return (*word, END_OF_WORD)


def _merge(symbols: tuple[str, ...], pair: tuple[str, str]
           ) -> tuple[str, ...]:
    out, i = [], 0
    while i < len(symbols):
        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
            out.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


class SimpleBPE:
    """Build with from_documents(); _fit() and _build() are internal."""

    def __init__(self, base_chars: Iterable[str],
                 merges: Sequence[tuple[str, str]] = (), *,
                 _token: object = None):
        if _token is not _PRIVATE:
            raise TypeError("use SimpleBPE.from_documents()")
        base = sorted(set(base_chars))
        if END_OF_WORD in base or any(len(c) != 1 for c in base):
            raise ValueError("base vocabulary must be single characters "
                             "other than the end-of-word marker")
        self.base_chars = tuple(base)
        self.merges = tuple(tuple(m) for m in merges)
        symbols = dict.fromkeys([*base, END_OF_WORD])
        symbols.update(dict.fromkeys(a + b for a, b in self.merges))
        self.tokens = [*SPECIAL_TOKENS, *symbols]
        # Special tokens only: body symbols are looked up in _symbol_to_id.
        self.token_to_id = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        self._symbol_to_id = {t: i for i, t in enumerate(self.tokens)
                              if i >= len(SPECIAL_TOKENS)}
        self._ranks = {m: r for r, m in enumerate(self.merges)}
        self._char_set = frozenset(base)
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @classmethod
    def _build(cls, base_chars: Iterable[str],
               merges: Sequence[tuple[str, str]] = ()) -> SimpleBPE:
        return cls(base_chars, merges, _token=_PRIVATE)

    @classmethod
    def from_documents(cls, docs: Iterable[Document],
                       vocab_size: int) -> SimpleBPE:
        """Fit on the bodies of T documents; any other split fails."""
        docs = _require_t(docs, "BPE")
        return cls._fit([d.body for d in docs], vocab_size)

    @classmethod
    def _fit(cls, bodies: Iterable[str], vocab_size: int) -> SimpleBPE:
        """Internal: no provenance check. Learn merges until vocab_size
        tokens (specials included) or no pair occurs at least twice."""
        word_freq: collections.Counter[str] = collections.Counter()
        for body in bodies:
            word_freq.update(WORD_RE.findall(body))
        base = {c for w in word_freq for c in w}
        if END_OF_WORD in base:
            raise ValueError("body contains the end-of-word marker")
        n_tokens = len(SPECIAL_TOKENS) + len(base) + 1
        if vocab_size < n_tokens:
            raise ValueError(f"vocab_size {vocab_size} is below the base "
                             f"vocabulary ({n_tokens})")
        seen = set(base)

        splits = {w: _symbols(w) for w in word_freq}
        merges: list[tuple[str, str]] = []
        while n_tokens < vocab_size:
            pair_counts: collections.Counter[tuple[str, str]] = (
                collections.Counter())
            for w, freq in word_freq.items():
                s = splits[w]
                for i in range(len(s) - 1):
                    pair_counts[(s[i], s[i + 1])] += freq
            if not pair_counts:
                break
            best = min(pair_counts, key=lambda p: (-pair_counts[p], p))
            if pair_counts[best] < 2:
                break
            merges.append(best)
            if best[0] + best[1] not in seen:
                seen.add(best[0] + best[1])
                n_tokens += 1
            for w, s in splits.items():
                if best[0] in s:
                    splits[w] = _merge(s, best)
        return cls._build(base, merges)

    def truncate(self, vocab_size: int) -> SimpleBPE:
        """The tokenizer with the shortest merge prefix that reaches
        vocab_size tokens; its token IDs are a prefix of this one's."""
        n_tokens = self.vocab_size - len(set(a + b for a, b in self.merges))
        seen: set[str] = set()
        for n_merges, (a, b) in enumerate(self.merges):
            if n_tokens == vocab_size:
                return SimpleBPE._build(self.base_chars,
                                        self.merges[:n_merges])
            if a + b not in seen:
                seen.add(a + b)
                n_tokens += 1
        if n_tokens == vocab_size:
            return SimpleBPE._build(self.base_chars, self.merges)
        raise ValueError(f"vocab_size {vocab_size} out of range")

    def _encode_word(self, word: str) -> list[int]:
        symbols = _symbols(word)
        no_merge = len(self._ranks)
        while len(symbols) > 1:
            pairs = zip(symbols, symbols[1:])
            pair = min(pairs, key=lambda p: self._ranks.get(p, no_merge))
            if pair not in self._ranks:
                break
            symbols = _merge(symbols, pair)
        return [self._symbol_to_id[s] for s in symbols]

    def encode(self, body: str) -> list[int]:
        """Encode a body on its own; fails closed on unknown characters."""
        unknown = set(body) - self._char_set
        if unknown:
            raise ValueError("characters outside the vocabulary: "
                             f"{sorted(f'U+{ord(c):04X}' for c in unknown)}")
        ids = []
        for w in WORD_RE.findall(body):
            if w not in self._cache:
                self._cache[w] = self._encode_word(w)
            ids.extend(self._cache[w])
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        """Inverse of encode; special IDs render as their token names."""
        return "".join(self.tokens[i] for i in ids).replace(END_OF_WORD, "")
