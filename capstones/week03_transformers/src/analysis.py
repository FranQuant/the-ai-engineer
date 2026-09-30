"""Confirmatory and exploratory statistics (DESIGN.md v0.8, §6 and §7).

Inputs are per-document scores (evaluate.DocumentScore or anything with
id, genre, meeting, split and bpc). Uncertainty is a percentile bootstrap
over meetings, 2,000 resamples, with a fixed seed; each statistic draws
from its own generator, so results do not depend on call order.

Labels (§6): supported if the 95% CI excludes 0 in the predicted direction
(lower bound > 0); contradicted if the point estimate is <= 0, which takes
precedence; inconclusive otherwise. H3 is labelled by rho >= 0.7 alone;
its CI is reported but not used for the label.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Iterable, Protocol, Sequence

import numpy as np

RESAMPLES = 2000
SEED = 20260925  # fixed before any real score exists (§6)
CI_LEVEL = 0.95
H3_THRESHOLD = 0.7
E1_TOP = 5

SUPPORTED, CONTRADICTED, INCONCLUSIVE = (
    "supported", "contradicted", "inconclusive")


class Score(Protocol):
    id: str
    genre: str
    meeting: str
    split: str
    bpc: float


@dataclasses.dataclass(frozen=True)
class Estimate:
    point: float
    ci_low: float
    ci_high: float
    label: str | None = None  # None for exploratory results
    n_meetings: int = 0
    n_documents: int = 0
    excluded_meetings: int = 0  # H2: F meetings without both genres
    undefined_resamples: int = 0  # rho: constant resample, left out of CI

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def delta_label(point: float, ci_low: float) -> str:
    """§6 label for a difference predicted to be > 0."""
    if point <= 0:
        return CONTRADICTED
    if ci_low > 0:
        return SUPPORTED
    return INCONCLUSIVE


def rho_label(rho: float) -> str:
    """§6 H3 label: the threshold alone decides."""
    return SUPPORTED if rho >= H3_THRESHOLD else CONTRADICTED


def rankdata(x: Sequence[float]) -> np.ndarray:
    """Ranks 1..n, ties get their average rank."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rho: Pearson correlation of average ranks."""
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("need two equal-length samples of size >= 2")
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        raise ValueError("a sample is constant: rho is undefined")
    return float(np.corrcoef(rx, ry)[0, 1])


def _group(scores: Iterable[Score]) -> dict[str, list[Score]]:
    """Scores by meeting, meetings in sorted order."""
    groups: dict[str, list[Score]] = {}
    for s in scores:
        groups.setdefault(s.meeting, []).append(s)
    return dict(sorted(groups.items()))


def _in_split(scores: Iterable[Score], split: str) -> list[Score]:
    scores = [s for s in scores if s.split == split]
    if not scores:
        raise ValueError(f"no {split} documents")
    return scores


def _percentile_ci(values: Sequence[float]) -> tuple[float, float]:
    tail = (1 - CI_LEVEL) / 2 * 100
    lo, hi = np.percentile(np.asarray(values), [tail, 100 - tail])
    return float(lo), float(hi)


def _bootstrap(strata: Sequence[int],
               stat: Callable[[list[np.ndarray]], float],
               resamples: int, seed: int) -> list[float]:
    """Resample meeting indices with replacement within each stratum
    (stratum sizes in `strata`) and apply stat to the index arrays."""
    rng = np.random.default_rng(seed)
    return [stat([rng.integers(0, n, n) for n in strata])
            for _ in range(resamples)]


def h1_drift(char_scores: Iterable[Score], resamples: int = RESAMPLES,
             seed: int = SEED) -> Estimate:
    """Δ₁ = mean BPC(F) - mean BPC(N), document means; meetings resampled
    within N and within F separately."""
    char_scores = list(char_scores)
    groups = [list(_group(_in_split(char_scores, sp)).values())
              for sp in ("N", "F")]
    bpc = [[np.array([s.bpc for s in m]) for m in g] for g in groups]

    def mean_docs(meetings: list[np.ndarray], idx: np.ndarray) -> float:
        return float(np.concatenate([meetings[i] for i in idx]).mean())

    def stat(draw: list[np.ndarray]) -> float:
        return mean_docs(bpc[1], draw[1]) - mean_docs(bpc[0], draw[0])

    point = stat([np.arange(len(g)) for g in bpc])
    lo, hi = _percentile_ci(_bootstrap([len(g) for g in bpc], stat,
                                       resamples, seed))
    return Estimate(point, lo, hi, delta_label(point, lo),
                    n_meetings=sum(len(g) for g in bpc),
                    n_documents=sum(len(m) for g in groups for m in g))


def genre_pairs(char_scores: Iterable[Score]
                ) -> tuple[list[tuple[str, float, float]], int]:
    """Matched F meetings as (meeting, minutes BPC, statement BPC), and
    the number of F meetings excluded for lacking a genre."""
    pairs, excluded = [], 0
    for meeting, docs in _group(_in_split(char_scores, "F")).items():
        by_genre: dict[str, float] = {}
        for s in docs:
            if s.genre in by_genre:
                raise ValueError(f"meeting {meeting} has two {s.genre}")
            by_genre[s.genre] = s.bpc
        if set(by_genre) == {"minutes", "statement"}:
            pairs.append((meeting, by_genre["minutes"],
                          by_genre["statement"]))
        else:
            excluded += 1
    return pairs, excluded


def h2_genre(char_scores: Iterable[Score], resamples: int = RESAMPLES,
             seed: int = SEED) -> Estimate:
    """Δ₂ = mean over matched F meetings of BPC(minutes) - BPC(statement);
    matched meetings resampled."""
    pairs, excluded = genre_pairs(char_scores)
    if not pairs:
        raise ValueError("no matched F meetings")
    diffs = np.array([m - s for _, m, s in pairs])

    def stat(draw: list[np.ndarray]) -> float:
        return float(diffs[draw[0]].mean())

    point = float(diffs.mean())
    lo, hi = _percentile_ci(_bootstrap([len(diffs)], stat, resamples,
                                       seed))
    return Estimate(point, lo, hi, delta_label(point, lo),
                    n_meetings=len(pairs), n_documents=2 * len(pairs),
                    excluded_meetings=excluded)


def rho_by_meeting(a: Iterable[Score], b: Iterable[Score], split: str = "F",
                   resamples: int = RESAMPLES, seed: int = SEED
                   ) -> Estimate:
    """Spearman rho between two instruments' per-document BPC on split,
    with a meeting-bootstrap CI. Both must score the same documents."""
    a_by_id = {s.id: s for s in _in_split(a, split)}
    b_by_id = {s.id: s for s in _in_split(b, split)}
    if a_by_id.keys() != b_by_id.keys():
        raise ValueError("the two instruments scored different documents")
    meetings = list(_group(a_by_id.values()).values())
    pairs = [np.array([[s.bpc, b_by_id[s.id].bpc] for s in m])
             for m in meetings]

    def stat(draw: list[np.ndarray]) -> float:
        xy = np.concatenate([pairs[i] for i in draw[0]])
        return spearman(xy[:, 0], xy[:, 1])

    def stat_or_nan(draw: list[np.ndarray]) -> float:
        # A resample can make a sample constant (e.g. one meeting drawn
        # every time); rho is then undefined and the resample is counted
        # and left out of the percentile CI.
        try:
            return stat(draw)
        except ValueError:
            return float("nan")

    point = stat([np.arange(len(pairs))])
    boot = np.array(_bootstrap([len(pairs)], stat_or_nan, resamples, seed))
    finite = boot[np.isfinite(boot)]
    lo, hi = _percentile_ci(finite)
    return Estimate(point, lo, hi, n_meetings=len(pairs),
                    n_documents=len(a_by_id),
                    undefined_resamples=int(len(boot) - len(finite)))


def h3_instruments(char_scores: Iterable[Score], bpe_scores: Iterable[Score],
                   resamples: int = RESAMPLES, seed: int = SEED) -> Estimate:
    """Spearman rho between char and BPE per-document BPC on F."""
    est = rho_by_meeting(char_scores, bpe_scores, "F", resamples, seed)
    return dataclasses.replace(est, label=rho_label(est.point))


def e1_top(char_scores: Iterable[Score], k: int = E1_TOP) -> list[dict]:
    """The k F documents with the highest char BPC."""
    top = sorted(_in_split(char_scores, "F"), key=lambda s: -s.bpc)[:k]
    return [{"id": s.id, "genre": s.genre, "meeting": s.meeting,
             "bpc": s.bpc} for s in top]


def e2_ngram(char_scores: Iterable[Score], ngram_scores: Iterable[Score],
             resamples: int = RESAMPLES, seed: int = SEED) -> Estimate:
    """Spearman rho between char transformer and n-gram BPC on F; no
    label (exploratory)."""
    return rho_by_meeting(char_scores, ngram_scores, "F", resamples, seed)
