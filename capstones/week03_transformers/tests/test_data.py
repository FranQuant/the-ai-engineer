"""Corpus handling checks (DESIGN.md §2-§3)."""

import json

import pytest
import torch

import data
from data import (BOS, GENRE_TOKENS, CharVocab, Document, NMonitorSampler,
                  WindowSampler, normalize, parse_documents, serialize)


@pytest.mark.parametrize("raw, expected", [
    ("café", "cafe"),              # (1)+(2) accent decomposed, dropped
    ("Vissing-Jørgensen", "Vissing-Jorgensen"),  # (4): NFD keeps ø
    ("Ǿ", "O"),              # (2) before (4): mark, then map
    ("co­op‍eration‎", "cooperation"),  # (3) Cf dropped
    ("Fedwire® ™©", "Fedwire "),        # (4) deletions
    ("한국", "한국"),   # (5) NFC recomposes Hangul jamo
    ("Å", "A"),                   # combining ring on A
])
def test_normalization_order(raw, expected):
    assert normalize(raw) == expected


def test_normalization_steps_are_ordered():
    # NFD alone would leave the Hangul syllable decomposed (Lo, not Mn), so
    # the result is only correct if NFC runs last.
    import unicodedata
    assert unicodedata.normalize("NFD", "한") != "한"
    assert normalize("한") == "한"


def test_normalization_idempotent():
    corpus = data.CORPUS.read_text(encoding="utf-8")
    once = normalize(corpus)
    assert normalize(once) == once
    for s in ("Ǿ®", "é‍", "ﬁ Ω"):
        assert normalize(normalize(s)) == normalize(s)


def test_body_parser_on_real_documents(documents):
    text, manifest, _ = data.load_corpus(data.CORPUS, data.MANIFEST)
    raw = {d["document_id"]: d for d in parse_documents(text)}
    assert len(raw) == manifest["document_count"] == len(documents)
    for entry in manifest["documents"]:
        assert (len(raw[entry["document_id"]]["body"])
                == entry["normalized_characters"])
    split = data.load_split_manifest()
    lengths = {e["document_id"]: e["body_code_points"]
               for m in split["meetings"] for e in m["documents"]}
    for d in documents:
        assert len(d.body) == lengths[d.id]
        assert not d.body.startswith(("date:", "document_id:", "\n"))
        assert "<|" not in d.body and not d.body.endswith("\n")


def test_body_parser_rejects_malformed_header():
    doc = ("<|fomc_statement|>\ndate: 2020-01-01\ndocument_id: x\n"
           "body without blank line\n<|end_fomc_statement|>\n")
    with pytest.raises(ValueError, match="malformed header"):
        parse_documents(doc)


def test_documents_loaded_with_split_counts(documents):
    split = data.load_split_manifest()
    for s in data.SPLITS:
        n = sum(1 for d in documents if d.split == s)
        assert n == split["counts"][s]["documents"]


def test_frozen_hash_mismatch_fails_closed(tmp_path):
    bad = tmp_path / "split_manifest.json"
    obj = json.loads(data.SPLIT_MANIFEST.read_text())
    obj["design_version"] = "tampered"
    bad.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="SHA-256"):
        data.load_documents(split_path=bad)


def test_serialization(t_documents):
    vocab = CharVocab.from_documents(t_documents)
    for d in t_documents[:5]:
        ids = serialize(vocab, d.genre, d.body)
        assert ids[0] == vocab.token_to_id[BOS] == 0
        assert ids[1] == vocab.token_to_id[GENRE_TOKENS[d.genre]]
        assert vocab.decode(ids[2:]) == d.body
        assert len(ids) == len(d.body) + data.PREFIX_LEN
        assert not set(ids[2:]) & {0, 1, 2}


def test_char_vocabulary(documents, t_documents):
    vocab = CharVocab.from_documents(t_documents)
    split = data.load_split_manifest()
    assert vocab.vocab_size == len(split["t_character_vocab"]) + 3 == 86
    assert data.check_characters(documents) == {}
    for d in documents:
        if d.split in ("N", "F"):
            assert vocab.decode(vocab.encode(d.body)) == d.body


def test_char_vocab_rejects_non_t_documents(documents):
    t = next(d for d in documents if d.split == "T")
    n = next(d for d in documents if d.split == "N")
    f = next(d for d in documents if d.split == "F")
    with pytest.raises(ValueError, match=r"T documents only.*\['F', 'N'\]"):
        CharVocab.from_documents([t, n, f])
    for leaked in (n, f):
        with pytest.raises(ValueError, match="T documents only"):
            CharVocab.from_documents([t, leaked])
    with pytest.raises(TypeError, match="from_documents"):
        CharVocab("abc")


def test_unknown_characters_fail_closed(t_documents):
    vocab = CharVocab.from_documents(t_documents)
    with pytest.raises(ValueError, match="U\\+2603"):
        vocab.encode("rates ☃")
    with pytest.raises(ValueError):
        vocab.encode("ø")  # mapped away by §3, so absent from T
    with pytest.raises(ValueError):
        CharVocab.from_documents(
            [Document("x", "statement", "2022-01-01", "F", "abc")])
    docs = [Document("a", "statement", "2019-01-01", "T", "ab"),
            Document("b", "statement", "2022-01-01", "F", "ab☃")]
    assert data.check_characters(docs) == {"F": {"U+2603": 1}}


def test_training_windows_never_cross_documents():
    # Document k holds tokens 1000k, 1000k + 1, ...: a window lies inside
    # one document iff its tokens are consecutive within one range.
    lengths = [9, 40, 17, 120]
    seqs = [list(range(1000 * k, 1000 * k + n))
            for k, n in enumerate(lengths)]
    sampler = WindowSampler(seqs, block_size=8)
    g = torch.Generator().manual_seed(0)
    x, y, doc = sampler.sample(5000, generator=g)
    window = torch.cat([x, y[:, -1:]], dim=1)
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert torch.all(window[:, 1:] - window[:, :-1] == 1)
    assert torch.equal(window[:, 0] // 1000, doc)
    assert torch.equal(window[:, -1] // 1000, doc)
    last = torch.tensor([1000 * k + n - 1 for k, n in enumerate(lengths)])
    assert torch.all(window[:, -1] <= last[doc])
    # The 9-token document has exactly one window; all positions reachable.
    assert set(window[doc == 0, 0].tolist()) == {0}
    assert set(window[doc == 3, 0].tolist()) == set(range(3000, 3112))


def test_training_windows_proportional_to_length():
    sampler = WindowSampler([[0] * 100, [1] * 300], block_size=4)
    g = torch.Generator().manual_seed(1)
    _, _, doc = sampler.sample(40000, generator=g)
    assert doc.double().mean().item() == pytest.approx(0.75, abs=0.01)


def test_training_windows_from_real_t_documents(documents, t_documents):
    vocab = CharVocab.from_documents(t_documents)
    sampler = WindowSampler.from_documents(t_documents, vocab, 256)
    x, y, _ = sampler.sample(64, generator=torch.Generator().manual_seed(2))
    assert x.shape == y.shape == (64, 256)
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert not torch.any(y == vocab.token_to_id[BOS])  # BOS is input only
    with pytest.raises(ValueError, match="T documents only"):
        WindowSampler.from_documents(documents, vocab, 256)


def test_short_documents_fail_closed():
    with pytest.raises(ValueError, match="shorter than block_size"):
        WindowSampler([[1, 2, 3]], block_size=3)


def _toy_docs(split):
    return [Document(f"{split}{i}", g, f"2020-0{i + 1}-01", split,
                     "rates were held steady " * 3)
            for i, g in enumerate(("statement", "minutes"))]


def test_monitor_sampler_takes_n_documents_only():
    vocab = CharVocab._from_chars("rates wheldy")
    sampler = NMonitorSampler.from_documents(_toy_docs("N"), vocab, 16)
    x, y, doc = sampler.sample(8, generator=torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (8, 16) and set(doc.tolist()) <= {0, 1}
    for split in ("T", "F", "excluded"):
        with pytest.raises(ValueError, match="N documents only"):
            NMonitorSampler.from_documents(
                _toy_docs("N") + _toy_docs(split), vocab, 16)
    with pytest.raises(TypeError):
        NMonitorSampler(WindowSampler([[0] * 20], block_size=4))
    # The training sampler's T-only guard is unchanged.
    with pytest.raises(ValueError, match="T documents only"):
        WindowSampler.from_documents(_toy_docs("N"), vocab, 16)
