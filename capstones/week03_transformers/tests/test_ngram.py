"""Witten–Bell character n-gram checks (DESIGN.md §5, E2)."""

import math

import pytest

from data import Document
from evaluate import bits_per_character
from ngram import WittenBellNgram

BODIES = ["the committee decided to maintain the target range",
          "inflation remains elevated; the committee is attentive",
          "xyz"]
CONTEXTS = ["", "t", "th", "the", "the ", "the committee", "zzzz", "q",
            "ee;x", "e ra"]


@pytest.fixture(scope="module")
def lm():
    return WittenBellNgram._fit(BODIES, order=5)


@pytest.mark.parametrize("context", CONTEXTS)
def test_probabilities_sum_to_one(lm, context):
    total = math.fsum(lm.prob(context, c) for c in lm.vocab)
    assert total == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("context", CONTEXTS)
def test_no_zero_probabilities(lm, context):
    assert min(lm.prob(context, c) for c in lm.vocab) > 0


def test_hand_computed_witten_bell():
    lm = WittenBellNgram._fit(["abab"], order=2)
    # order 0: uniform 1/2; order 1: a, b seen twice each, 2 types
    assert lm.prob("", "a") == pytest.approx((2 + 2 * 0.5) / (4 + 2))
    # after "a": b seen twice, 1 type; interpolate with P1(b) = 1/2
    assert lm.prob("a", "b") == pytest.approx((2 + 1 * 0.5) / (2 + 1))
    assert lm.prob("a", "a") == pytest.approx((0 + 1 * 0.5) / (2 + 1))
    # after "b": seen once (position 2), next a
    assert lm.prob("b", "a") == pytest.approx((1 + 1 * 0.5) / (1 + 1))
    # only the last order - 1 characters count
    assert lm.prob("bbbba", "b") == lm.prob("a", "b")


def test_context_resets_at_document_start(lm):
    body = "the committee"
    expected = -sum(math.log(lm.prob(body[max(0, i - 4):i], ch))
                    for i, ch in enumerate(body))
    assert lm.body_nll(body) == pytest.approx(expected, rel=1e-12)
    # The first character is scored from the empty context only.
    assert lm.body_nll("t") == pytest.approx(-math.log(lm.prob("", "t")))


def test_documents_scored_independently_with_section4_denominator(lm):
    docs = [Document("a", "statement", "2010-01-27", "F", BODIES[0]),
            Document("b", "minutes", "2010-01-27", "F", BODIES[1])]
    together = lm.score_documents(docs)
    alone = [lm.score_document(d) for d in docs]
    assert together == alone
    s = together[0]
    assert s.body_tokens == s.body_code_points == len(BODIES[0])
    assert s.bpc == bits_per_character(s.total_nll, len(BODIES[0]))
    assert s.bpc == pytest.approx(
        s.total_nll / (len(BODIES[0]) * math.log(2)))


def test_unknown_character_fails_closed(lm):
    with pytest.raises(ValueError, match=r"U\+2603"):
        lm.body_nll("the ☃")
    with pytest.raises(ValueError, match=r"U\+2603"):
        lm.prob("the", "☃")


def test_rejects_non_t_fitting(documents, t_documents):
    with pytest.raises(ValueError, match=r"T documents only.*\['F', 'N'"):
        WittenBellNgram.from_documents(documents)
    for split in ("N", "F", "excluded"):
        doc = Document("x", "statement", "2022-01-01", split, "abc")
        with pytest.raises(ValueError, match="T documents only"):
            WittenBellNgram.from_documents(t_documents[:2] + [doc])
    with pytest.raises(TypeError):
        WittenBellNgram(BODIES, 5)


def test_fits_on_t_documents(t_documents):
    lm = WittenBellNgram.from_documents(t_documents[:3])
    assert lm.order == 5
    assert lm.vocab == sorted({c for d in t_documents[:3] for c in d.body})
