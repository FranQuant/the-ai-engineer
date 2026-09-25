"""Per-document scoring checks (DESIGN.md §4)."""

import math

import pytest
import torch
import torch.nn.functional as F

from data import PREFIX_LEN, CharVocab, Document, serialize
from evaluate import (bits_per_character, score_document, score_ids,
                      window_plan)
from model import ModelConfig, TinyTransformerLM

DOC = Document("fomc-2023-01-31", "statement", "2023-01-31", "F",
               "The Committee decided to maintain the target range.\n" * 3)


def _tiny_lm(vocab, block_size, seed=0):
    torch.manual_seed(seed)
    return TinyTransformerLM(ModelConfig(
        vocab_size=vocab, d_model=16, num_heads=2, num_layers=2, d_ff=32,
        block_size=block_size, dropout=0.0)).eval()


@pytest.mark.parametrize("n_tokens", [3, 4, 17, 64, 65, 200])
@pytest.mark.parametrize("block_size", [1, 2, 7, 16, 64])
def test_every_body_target_scored_once(n_tokens, block_size):
    plan = window_plan(n_tokens, block_size)
    targets = [t for _, start, stop in plan for t in range(start, stop)]
    assert targets == list(range(PREFIX_LEN, n_tokens))
    for ctx, start, stop in plan:
        assert 0 <= ctx < start and stop - 1 - ctx <= block_size
        assert stop - start <= max(1, block_size // 2)


def test_bits_per_character():
    assert bits_per_character(math.log(2) * 10, 10) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        bits_per_character(1.0, 0)


def test_uniform_model_bpc():
    vocab = CharVocab._from_chars(set(DOC.body))
    lm = _tiny_lm(vocab.vocab_size, 16)
    with torch.no_grad():
        lm.tok_emb.weight.zero_()
    score = score_document(lm, vocab, DOC, block_size=16)
    assert score.body_tokens == score.body_code_points == len(DOC.body)
    assert score.bpc == pytest.approx(math.log2(vocab.vocab_size))


def test_windows_match_full_context_when_document_fits():
    vocab = CharVocab._from_chars(set(DOC.body))
    ids = torch.tensor(serialize(vocab, DOC.genre, DOC.body))
    lm = _tiny_lm(vocab.vocab_size, len(ids))
    nll, scored = score_ids(lm, ids, block_size=len(ids))
    with torch.no_grad():
        logits, _ = lm(ids[:-1].unsqueeze(0))
        full = F.cross_entropy(logits[0, PREFIX_LEN - 1:].double(),
                               ids[PREFIX_LEN:], reduction="sum")
    assert scored == list(range(PREFIX_LEN, len(ids)))
    assert nll == pytest.approx(full.item(), rel=1e-6)


def test_windowed_scores_match_bruteforce():
    vocab = CharVocab._from_chars(set(DOC.body))
    ids = torch.tensor(serialize(vocab, DOC.genre, DOC.body))
    block = 8
    lm = _tiny_lm(vocab.vocab_size, block)
    nll, _ = score_ids(lm, ids, block_size=block)
    brute = 0.0
    with torch.no_grad():
        for plan_ctx, start, stop in window_plan(len(ids), block):
            logits, _ = lm(ids[plan_ctx:stop - 1].unsqueeze(0))
            for t in range(start, stop):
                p = logits[0, t - 1 - plan_ctx].double().log_softmax(-1)
                brute -= p[ids[t]].item()
    assert nll == pytest.approx(brute, rel=1e-6)


def test_lossy_tokenizer_fails_assertion():
    class Lossy(CharVocab):
        def decode(self, ids):
            return super().decode(ids).upper()

    vocab = Lossy._from_chars(set(DOC.body))
    lm = _tiny_lm(vocab.vocab_size, 16)
    with pytest.raises(AssertionError, match="decode"):
        score_document(lm, vocab, DOC, block_size=16)
