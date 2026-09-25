"""Character n-gram baseline for E2 (DESIGN.md v0.8, §5).

Interpolated Witten–Bell, order 5, fixed a priori (no tuned parameters).
The order-0 fallback is uniform over the T character vocabulary, so no
probability is zero. Fit on T bodies only, enforced as for the tokenizers:
from_documents() is the only public way to fit.

Each body is counted and scored on its own, with the context reset at the
document start: position i sees at most body[i - order + 1:i], never text
from another document. There is no BOS or genre prefix; scores use the §4
denominator (body code points x ln 2), as evaluate.py does.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from data import _PRIVATE, Document, _require_t
from evaluate import DocumentScore, bits_per_character

ORDER = 5


class WittenBellNgram:
    """Build with from_documents(); _fit() is for tests and toy input."""

    def __init__(self, bodies: Sequence[str], order: int, *,
                 _token: object = None):
        if _token is not _PRIVATE:
            raise TypeError("use WittenBellNgram.from_documents()")
        if order < 1:
            raise ValueError("order must be at least 1")
        self.order = order
        self.vocab = sorted({c for b in bodies for c in b})
        if not self.vocab:
            raise ValueError("no characters to fit")
        self._vocab_set = set(self.vocab)
        # tables[k]: context of length k -> {next char: count}; k < order
        counts: list[dict[str, dict[str, int]]] = [{} for _ in range(order)]
        for body in bodies:
            for i, ch in enumerate(body):
                for k in range(min(order, i + 1)):
                    nxt = counts[k].setdefault(body[i - k:i], {})
                    nxt[ch] = nxt.get(ch, 0) + 1
        # per context: (next-char counts, total count, distinct next chars)
        self._tables = [{h: (n, sum(n.values()), len(n))
                         for h, n in t.items()} for t in counts]

    @classmethod
    def from_documents(cls, docs: Iterable[Document], order: int = ORDER
                       ) -> WittenBellNgram:
        docs = _require_t(docs, "the n-gram")
        return cls._fit([d.body for d in docs], order)

    @classmethod
    def _fit(cls, bodies: Iterable[str], order: int = ORDER
             ) -> WittenBellNgram:
        """Internal: no provenance check. Tests and toy input only."""
        return cls(list(bodies), order, _token=_PRIVATE)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def prob(self, context: str, ch: str) -> float:
        """P(ch | the last order - 1 characters of context)."""
        if ch not in self._vocab_set:
            raise ValueError(f"character outside the vocabulary: "
                             f"U+{ord(ch):04X}")
        context = context[-(self.order - 1):] if self.order > 1 else ""
        p = 1.0 / len(self.vocab)
        for k in range(len(context) + 1):
            entry = self._tables[k].get(context[len(context) - k:])
            if entry is None:
                break  # every longer context ends in this one: unseen too
            n, total, types = entry
            p = (n.get(ch, 0) + types * p) / (total + types)
        return p

    def body_nll(self, body: str) -> float:
        """Summed NLL in nats of every character of body, context reset at
        the document start."""
        unknown = {c for c in body if c not in self._vocab_set}
        if unknown:
            raise ValueError(f"characters outside the vocabulary: "
                             f"{sorted(f'U+{ord(c):04X}' for c in unknown)}")
        # Same recursion as prob(), inlined: this loop scores all of F.
        tables, uniform = self._tables, 1.0 / len(self.vocab)
        nll = 0.0
        for i, ch in enumerate(body):
            p = uniform
            for k in range(min(self.order, i + 1)):
                entry = tables[k].get(body[i - k:i])
                if entry is None:
                    break
                n, total, types = entry
                p = (n.get(ch, 0) + types * p) / (total + types)
            nll -= math.log(p)
        return nll

    def score_document(self, doc: Document) -> DocumentScore:
        nll = self.body_nll(doc.body)
        return DocumentScore(
            id=doc.id, genre=doc.genre, meeting=doc.meeting, split=doc.split,
            total_nll=nll, body_tokens=len(doc.body),
            body_code_points=len(doc.body),
            bpc=bits_per_character(nll, len(doc.body)))

    def score_documents(self, docs: Iterable[Document]
                        ) -> list[DocumentScore]:
        return [self.score_document(d) for d in docs]
