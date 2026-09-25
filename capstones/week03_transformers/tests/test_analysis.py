"""§6/§7 statistics on synthetic scores with known answers."""

import math

import numpy as np
import pytest

import analysis
from analysis import (CONTRADICTED, INCONCLUSIVE, SUPPORTED, delta_label,
                      e1_top, e2_ngram, genre_pairs, h1_drift, h2_genre,
                      h3_instruments, rankdata, rho_label, spearman)
from evaluate import DocumentScore


def score(split, meeting, genre, bpc, doc_id=None):
    doc_id = doc_id or f"{split}-{meeting}-{genre}"
    return DocumentScore(id=doc_id, genre=genre, meeting=meeting,
                         split=split, total_nll=0.0, body_tokens=1,
                         body_code_points=1, bpc=bpc)


def meetings(split, values, genres=("statement", "minutes")):
    """One meeting per entry of values; each entry gives one BPC per genre."""
    return [score(split, f"{split}{i:02d}", g, v)
            for i, vs in enumerate(values) for g, v in zip(genres, vs)]


# ---- labels ---------------------------------------------------------------

@pytest.mark.parametrize("point, lo, label", [
    (0.5, 0.1, SUPPORTED),
    (0.3, -0.1, INCONCLUSIVE),
    (0.3, 0.0, INCONCLUSIVE),   # CI touching 0 does not exclude it
    (0.0, -0.2, CONTRADICTED),  # point <= 0
    (-0.4, -0.9, CONTRADICTED),
    (0.0, 0.1, CONTRADICTED),   # point <= 0 takes precedence
])
def test_delta_label(point, lo, label):
    assert delta_label(point, lo) == label


def test_rho_label_threshold():
    assert rho_label(0.7) == SUPPORTED
    assert rho_label(0.95) == SUPPORTED
    assert rho_label(0.6999) == CONTRADICTED


# ---- ranks and Spearman ---------------------------------------------------

def test_rankdata_averages_ties():
    assert rankdata([10, 20, 20, 5]).tolist() == [2, 3.5, 3.5, 1]


def test_spearman_known_values():
    x = [1, 2, 3, 4, 5]
    assert spearman(x, [2, 1, 4, 3, 5]) == pytest.approx(0.8)  # 1-6*4/120
    assert spearman(x, [10, 20, 30, 40, 50]) == pytest.approx(1.0)
    assert spearman(x, [5, 4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman(x, [1, 4, 9, 16, 25]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        spearman([1, 1, 1], [1, 2, 3])


def test_spearman_matches_scipy_with_ties():
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    x, y = rng.integers(0, 5, 40), rng.integers(0, 5, 40)
    assert spearman(x, y) == pytest.approx(stats.spearmanr(x, y)[0])


# ---- H1 -------------------------------------------------------------------

def test_h1_point_is_document_mean_and_resampling_stays_in_split():
    # Constant BPC within each split: if a resample mixed N and F
    # meetings, the CI would not collapse to the point.
    n = meetings("N", [(1.0, 1.0)] * 4)
    f = meetings("F", [(1.5, 1.5)] * 6)
    est = h1_drift(n + f)
    assert est.point == pytest.approx(0.5)
    assert (est.ci_low, est.ci_high) == pytest.approx((0.5, 0.5))
    assert est.label == SUPPORTED
    assert (est.n_meetings, est.n_documents) == (10, 20)


def test_h1_weights_documents_not_meetings():
    n = [score("N", "N0", "statement", 1.0)]
    f = [score("F", "F0", "statement", 1.0),
         score("F", "F1", "statement", 4.0),
         score("F", "F1", "minutes", 4.0)]
    # document mean of F = 3.0 (meeting mean would be 2.5)
    assert h1_drift(n + f, resamples=10).point == pytest.approx(2.0)


def test_h1_contradicted():
    est = h1_drift(meetings("N", [(2.0, 2.2)] * 5)
                   + meetings("F", [(1.0, 1.1)] * 5))
    assert est.point == pytest.approx(-1.05) and est.label == CONTRADICTED


def test_h1_inconclusive():
    rng = np.random.default_rng(3)
    n = meetings("N", rng.normal(2.0, 0.5, (8, 2)))
    f = meetings("F", rng.normal(2.0, 0.5, (8, 2)))
    shift = 0.05 - h1_drift(n + f, resamples=10).point  # force point = 0.05
    f = [score("F", s.meeting, s.genre, s.bpc + shift) for s in f]
    est = h1_drift(n + f)
    assert est.point == pytest.approx(0.05)
    assert est.ci_low < 0 < est.ci_high and est.label == INCONCLUSIVE


def test_h1_bootstrap_is_seeded():
    rng = np.random.default_rng(4)
    data = (meetings("N", rng.normal(2, 0.3, (6, 2)))
            + meetings("F", rng.normal(2.2, 0.3, (6, 2))))
    assert h1_drift(data) == h1_drift(data)
    assert h1_drift(data).ci_low != h1_drift(data, seed=1).ci_low


# ---- H2 -------------------------------------------------------------------

def test_h2_pairs_within_meeting_and_counts_excluded():
    f = meetings("F", [(1.0, 1.3), (2.0, 2.1), (1.5, 1.7)])
    f.append(score("F", "F99", "statement", 9.0))  # unmatched: excluded
    n = meetings("N", [(5.0, 0.0)])  # N never enters H2
    pairs, excluded = genre_pairs(f + n)
    assert [m for m, _, _ in pairs] == ["F00", "F01", "F02"]
    assert excluded == 1
    est = h2_genre(f + n)
    assert est.point == pytest.approx((0.3 + 0.1 + 0.2) / 3)
    assert est.label == SUPPORTED  # every difference > 0
    assert (est.n_meetings, est.excluded_meetings) == (3, 1)


def test_h2_contradicted_and_inconclusive():
    est = h2_genre(meetings("F", [(1.0, 0.8), (1.0, 0.9)]))
    assert est.point < 0 and est.label == CONTRADICTED
    est = h2_genre(meetings("F", [(1.0, 1.5), (1.0, 0.6), (1.0, 1.2),
                                  (1.0, 0.8)]))
    assert est.point == pytest.approx(0.025)
    assert est.ci_low < 0 and est.label == INCONCLUSIVE


def test_h2_rejects_duplicate_genre():
    f = meetings("F", [(1.0, 1.2)]) + [score("F", "F00", "minutes", 1.0,
                                             doc_id="dup")]
    with pytest.raises(ValueError, match="two minutes"):
        h2_genre(f)


# ---- H3, E1, E2 -----------------------------------------------------------

def test_h3_supported_when_rankings_agree():
    rng = np.random.default_rng(5)
    char = meetings("F", rng.normal(2, 0.5, (15, 2)))
    bpe = [score("F", s.meeting, s.genre, 0.5 * s.bpc + 1, s.id)
           for s in char]  # monotone transform: rho = 1
    est = h3_instruments(char, bpe)
    assert est.point == pytest.approx(1.0) and est.label == SUPPORTED
    assert est.ci_low == pytest.approx(1.0) and est.n_documents == 30


def test_h3_contradicted_below_threshold():
    rng = np.random.default_rng(6)
    char = meetings("F", rng.normal(2, 0.5, (15, 2)))
    bpe = [score("F", s.meeting, s.genre, v, s.id)
           for s, v in zip(char, rng.normal(2, 0.5, 30))]  # unrelated
    est = h3_instruments(char, bpe)
    assert est.point < 0.7 and est.label == CONTRADICTED
    assert est.ci_low <= est.point <= est.ci_high


def test_h3_uses_f_only_and_matching_documents():
    char = meetings("F", [(1, 2), (3, 4), (5, 6)])
    bpe = [score("F", s.meeting, s.genre, s.bpc, s.id) for s in char]
    n_noise = meetings("N", [(9, 0)])
    assert h3_instruments(char + n_noise, bpe).point == pytest.approx(1.0)
    with pytest.raises(ValueError, match="different documents"):
        h3_instruments(char, bpe[:-1])


def test_e1_top_five_f_documents():
    f = meetings("F", [(1, 7), (3, 4), (8, 2), (5, 6)])
    n = meetings("N", [(99, 98)])
    top = e1_top(f + n)
    assert [d["bpc"] for d in top] == [8, 7, 6, 5, 4]
    assert all(d["id"].startswith("F-") for d in top)


def test_e2_is_exploratory():
    char = meetings("F", [(1, 2), (3, 4), (5, 6), (7, 8)])
    ngram = [score("F", s.meeting, s.genre, -s.bpc, s.id) for s in char]
    est = e2_ngram(char, ngram)
    assert est.point == pytest.approx(-1.0) and est.label is None


def test_frozen_constants():
    assert analysis.RESAMPLES == 2000 and analysis.H3_THRESHOLD == 0.7
    assert analysis.CI_LEVEL == 0.95 and analysis.E1_TOP == 5
    assert math.isfinite(analysis.SEED)
