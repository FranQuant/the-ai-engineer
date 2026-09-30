"""Per-document scoring (DESIGN.md v0.8, §4).

Adapted from v1's evaluate_unsmoothed_nll. Each document is scored on its
own, starting from <BOS> + genre token, with deterministic sliding windows
at stride = block_size // 2. Only body targets are scored, each exactly
once, with unsmoothed next-token NLL. The §4 assertions run per document and
raise AssertionError (they do not depend on Python's -O flag).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Iterable

import torch
import torch.nn.functional as F

from data import PREFIX_LEN, Document, Tokenizer, serialize


@dataclasses.dataclass(frozen=True)
class DocumentScore:
    id: str
    genre: str
    meeting: str
    split: str
    total_nll: float  # nats, summed over body tokens
    body_tokens: int
    body_code_points: int
    bpc: float


def bits_per_character(total_nll: float, code_points: int) -> float:
    """§4: summed NLL in nats / (body code points x ln 2)."""
    if code_points <= 0:
        raise ValueError("code_points must be positive")
    return total_nll / (code_points * math.log(2))


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def window_plan(n_tokens: int, block_size: int, stride: int | None = None,
                first_target: int = PREFIX_LEN
                ) -> list[tuple[int, int, int]]:
    """Sliding windows over one serialized document.

    Returns (context_start, target_start, target_stop): the model reads
    ids[context_start:target_stop - 1] and scores targets
    ids[target_start:target_stop], at most block_size input tokens.
    Targets start at first_target, so BOS and the genre token are context
    only.
    """
    stride = max(1, block_size // 2) if stride is None else stride
    if not 1 <= stride <= block_size:
        raise ValueError("stride must be between 1 and block_size")
    if first_target < 1:
        raise ValueError("the first token has no context to be scored from")
    plan = []
    for target_start in range(first_target, n_tokens, stride):
        target_stop = min(target_start + stride, n_tokens)
        context_start = max(0, target_stop - 1 - block_size)
        plan.append((context_start, target_start, target_stop))
    return plan


@torch.no_grad()
def score_ids(model, ids: torch.Tensor, block_size: int,
              stride: int | None = None, first_target: int = PREFIX_LEN
              ) -> tuple[float, list[int]]:
    """Summed unsmoothed NLL of ids[first_target:] and the scored target
    positions, in order."""
    if ids.ndim != 1:
        raise ValueError("ids must be a 1-D tensor")
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    total = torch.zeros((), dtype=torch.float64, device=device)
    scored: list[int] = []
    try:
        for ctx, start, stop in window_plan(len(ids), block_size, stride,
                                            first_target):
            x = ids[ctx:stop - 1].unsqueeze(0).to(device)
            y = ids[start:stop].to(device)
            logits, _ = model(x)
            n = stop - start
            total += F.cross_entropy(logits[0, -n:, :].double(), y,
                                     reduction="sum")
            scored.extend(range(start, stop))
    finally:
        model.train(was_training)
    return total.item(), scored


def score_document(model, tokenizer: Tokenizer, doc: Document,
                   block_size: int, stride: int | None = None
                   ) -> DocumentScore:
    """Score one document independently with the §4 assertions."""
    body_ids = tokenizer.encode(doc.body)
    # (a) the body round-trips through the tokenizer on its own
    _check(tokenizer.decode(body_ids) == doc.body,
           f"{doc.id}: decode(encode(body)) != body")
    ids = serialize(tokenizer, doc.genre, doc.body)
    _check(ids[PREFIX_LEN:] == body_ids,
           f"{doc.id}: serialized body differs from the body encoding")
    total_nll, scored = score_ids(model, torch.tensor(ids), block_size,
                                  stride)
    # (b) one scored target per body token, each exactly once
    _check(len(scored) == len(body_ids),
           f"{doc.id}: scored {len(scored)} targets for "
           f"{len(body_ids)} body tokens")
    _check(scored == list(range(PREFIX_LEN, len(ids))),
           f"{doc.id}: body targets not scored exactly once")
    # (c) BOS and the genre token are never scored
    _check(min(scored, default=PREFIX_LEN) >= PREFIX_LEN,
           f"{doc.id}: a BOS or genre token was scored")
    return DocumentScore(
        id=doc.id, genre=doc.genre, meeting=doc.meeting, split=doc.split,
        total_nll=total_nll, body_tokens=len(body_ids),
        body_code_points=len(doc.body),
        bpc=bits_per_character(total_nll, len(doc.body)))


def score_documents(model, tokenizer: Tokenizer, docs: Iterable[Document],
                    block_size: int, stride: int | None = None
                    ) -> list[DocumentScore]:
    return [score_document(model, tokenizer, d, block_size, stride)
            for d in docs]
