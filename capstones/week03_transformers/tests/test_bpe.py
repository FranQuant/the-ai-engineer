"""BPE tokenizer checks (DESIGN.md §3-§5)."""

import pytest

from bpe import END_OF_WORD, SimpleBPE
from data import BOS, GENRE_TOKENS, SPECIAL_TOKENS, serialize


@pytest.fixture(scope="module")
def sample_bodies(t_documents):
    return [d.body for d in t_documents[:8]]


@pytest.fixture(scope="module")
def bpe(t_documents):
    return SimpleBPE.from_documents(t_documents[:8], vocab_size=260)


def test_fit_rejects_non_t_documents(documents):
    t = next(d for d in documents if d.split == "T")
    n = next(d for d in documents if d.split == "N")
    f = next(d for d in documents if d.split == "F")
    with pytest.raises(ValueError, match=r"T documents only.*\['F', 'N'\]"):
        SimpleBPE.from_documents([t, n, f], vocab_size=120)
    for leaked in (n, f):
        with pytest.raises(ValueError, match="T documents only"):
            SimpleBPE.from_documents([t, leaked], vocab_size=120)


def test_no_public_raw_fit():
    assert not hasattr(SimpleBPE, "fit")
    with pytest.raises(TypeError, match="from_documents"):
        SimpleBPE("abc")


def test_round_trip_is_lossless(bpe, sample_bodies):
    assert bpe.vocab_size == 260
    for body in sample_bodies:
        ids = bpe.encode(body)
        assert bpe.decode(ids) == body
        assert len(ids) < len(body)
    chars = "".join(bpe.base_chars)
    for text in (" \n\n  ", "\n" + chars + "  " + chars[::-1], "a"):
        text = "".join(c for c in text if c in bpe.base_chars)
        assert bpe.decode(bpe.encode(text)) == text


def test_unknown_characters_fail_closed(bpe):
    assert "<unk>" not in bpe.tokens
    with pytest.raises(ValueError, match="U\\+2603"):
        bpe.encode("rates ☃")
    with pytest.raises(ValueError):
        bpe.encode(END_OF_WORD)


def test_special_tokens_reserved_and_never_merged():
    # Bodies that spell the special tokens must still get ordinary IDs.
    body = "<BOS><stmt> <min> <BOS><BOS>\n" * 30
    tok = SimpleBPE._fit([body], vocab_size=60)
    assert tok.tokens[:3] == list(SPECIAL_TOKENS)
    assert tok.token_to_id == {t: i for i, t in enumerate(SPECIAL_TOKENS)}
    assert any(a + b in ("<BOS>", "<BOS>" + END_OF_WORD)
               for a, b in tok.merges)  # the literal string was merged ...
    ids = tok.encode(body)
    assert not set(ids) & {0, 1, 2}  # ... but never into a special ID
    assert tok.decode(ids) == body
    seq = serialize(tok, "minutes", body)
    assert seq[:2] == [tok.token_to_id[BOS],
                       tok.token_to_id[GENRE_TOKENS["minutes"]]]
    assert seq[2:] == ids  # body encoded alone: no merge across the genre


def test_smaller_vocabulary_is_a_prefix(sample_bodies, bpe):
    small = SimpleBPE._fit(sample_bodies, vocab_size=200)
    assert small.merges == bpe.merges[:len(small.merges)]
    assert small.tokens == bpe.tokens[:small.vocab_size]
    truncated = bpe.truncate(200)
    assert truncated.merges == small.merges
    assert truncated.tokens == small.tokens
    body = sample_bodies[0]
    assert truncated.encode(body) == small.encode(body)
    assert small.decode(small.encode(body)) == body


def test_committee_compresses():
    tok = SimpleBPE._fit(["the committee decided " * 20], vocab_size=60)
    ids = tok.encode("committee")
    assert len(ids) < len("committee")
    assert tok.decode(ids) == "committee"


def test_vocab_size_below_base_rejected():
    with pytest.raises(ValueError):
        SimpleBPE._fit(["abc"], vocab_size=5)
