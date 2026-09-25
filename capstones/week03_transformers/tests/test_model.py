"""Model checks from DESIGN.md §11 (handout coverage)."""

import math

import pytest
import torch
import torch.nn.functional as F

from model import (ModelConfig, MultiHeadAttention, SelfAttention,
                   TinyTransformerLM, make_causal_mask,
                   scaled_dot_product_attention)

# §4.3 worked example
Q = torch.tensor([[1., 0], [0, 1], [1, 1]])
K = torch.tensor([[1., 0], [1, 1], [0, 1]])
V = torch.tensor([[1., 0], [0, 2], [3, 1]])
Y_EXPECTED = torch.tensor(
    [[0.994440, 1.0], [1.401112, 1.203336], [0.993020, 1.255235]])


def test_worked_example():
    Y, S, A = scaled_dot_product_attention(Q, K, V)

    qkt = torch.tensor([[1., 1, 0], [0, 1, 1], [1, 2, 1]])
    assert torch.equal(Q @ K.T, qkt)
    r = 1 / math.sqrt(2)
    # S is returned shifted by its row maximum (softmax-invariant).
    s_expected = torch.tensor([[0., 0, -r], [-r, 0, 0], [-r, 0, -r]])
    assert torch.allclose(S, s_expected, atol=1e-6)
    a = math.exp(-r)
    a_expected = torch.tensor([[1, 1, a], [a, 1, 1], [a, 1, a]])
    a_expected = a_expected / a_expected.sum(dim=-1, keepdim=True)
    assert torch.allclose(A, a_expected, atol=1e-6)
    assert torch.allclose(A.sum(dim=-1), torch.ones(3))
    assert torch.allclose(Y, Y_EXPECTED, atol=1e-5)


def test_worked_example_causal():
    _, _, A = scaled_dot_product_attention(Q, K, V,
                                           mask=make_causal_mask(3))
    assert torch.equal(A[0], torch.tensor([1., 0., 0.]))
    assert torch.count_nonzero(torch.triu(A, diagonal=1)) == 0
    assert torch.allclose(A.sum(dim=-1), torch.ones(3))


def test_boolean_and_float_masks_identical():
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(2, 3, 6, 4, generator=g) for _ in range(3))
    bool_mask = make_causal_mask(6)
    float_mask = make_causal_mask(6, additive=True)
    assert bool_mask.dtype == torch.bool and float_mask.is_floating_point()
    assert torch.equal(float_mask == 0, bool_mask)
    out_b = scaled_dot_product_attention(q, k, v, mask=bool_mask)
    out_f = scaled_dot_product_attention(q, k, v, mask=float_mask)
    for tb, tf in zip(out_b, out_f):
        assert torch.equal(tb, tf)


def test_integer_mask_rejected():
    with pytest.raises(TypeError):
        scaled_dot_product_attention(Q, K, V, mask=torch.ones(3, 3).long())


@pytest.mark.parametrize("causal", [False, True])
def test_fused_matches_manual(causal):
    g = torch.Generator().manual_seed(1)
    q, k, v = (torch.randn(2, 3, 5, 4, generator=g) for _ in range(3))
    mask = make_causal_mask(5) if causal else None
    manual, _, _ = scaled_dot_product_attention(q, k, v, mask=mask)
    fused = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0,
                                           is_causal=causal)
    assert torch.allclose(manual, fused, atol=1e-6, rtol=1e-5)

    torch.manual_seed(2)
    mha = MultiHeadAttention(8, 2, causal=causal, use_fused=False)
    x = torch.randn(2, 5, 8)
    y_manual = mha(x)
    mha.use_fused = True
    assert torch.allclose(mha(x), y_manual, atol=1e-6, rtol=1e-5)


def test_mha_single_head_equals_self_attention():
    torch.manual_seed(3)
    sa = SelfAttention(d_model=4, causal=True)
    mha = MultiHeadAttention(d_model=4, num_heads=1, causal=True)
    with torch.no_grad():
        mha.proj_qkv.weight.copy_(torch.cat(
            [sa.W_Q.weight, sa.W_K.weight, sa.W_V.weight], dim=0))
        mha.proj_out.weight.copy_(torch.eye(4))
    x = torch.randn(1, 3, 4)
    assert torch.allclose(mha(x), sa(x), atol=1e-6)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("use_fused", [False, True])
def test_mha_matches_per_head_reference(causal, use_fused):
    """H = 2: each head computed by hand from sliced projection weights.

    proj_qkv rows are [Q | K | V], each split into H contiguous heads of
    d_head rows; the head outputs are concatenated in head order.
    """
    d_model, H, T = 8, 2, 5
    d_head = d_model // H
    mha = MultiHeadAttention(d_model, H, causal=causal, use_fused=use_fused)
    w_qkv = torch.sin(torch.arange(3 * d_model * d_model, dtype=torch.float)
                      ).reshape(3 * d_model, d_model)
    w_out = torch.cos(0.7 * torch.arange(d_model * d_model,
                                         dtype=torch.float)
                      ).reshape(d_model, d_model)
    with torch.no_grad():
        mha.proj_qkv.weight.copy_(w_qkv)
        mha.proj_out.weight.copy_(w_out)
    x = torch.cos(1.3 * torch.arange(2 * T * d_model, dtype=torch.float)
                  ).reshape(2, T, d_model)

    mask = torch.ones(T, T).tril().bool() if causal else None
    heads = []
    for h in range(H):
        rows = slice(h * d_head, (h + 1) * d_head)
        w_q = w_qkv[0 * d_model:1 * d_model][rows]
        w_k = w_qkv[1 * d_model:2 * d_model][rows]
        w_v = w_qkv[2 * d_model:3 * d_model][rows]
        y_h, _, _ = scaled_dot_product_attention(
            x @ w_q.T, x @ w_k.T, x @ w_v.T, mask=mask)
        heads.append(y_h)
    expected = torch.cat(heads, dim=-1) @ w_out.T

    assert torch.allclose(mha(x), expected, atol=1e-5, rtol=1e-5)
    # The two heads must differ, or a head swap would go unnoticed.
    assert not torch.allclose(heads[0], heads[1], atol=1e-2)


def test_uniform_logits_loss_is_log_vocab():
    vocab = 50
    lm = TinyTransformerLM(ModelConfig(
        vocab_size=vocab, d_model=16, num_heads=2, num_layers=1, d_ff=32,
        block_size=8, dropout=0.0))
    with torch.no_grad():
        lm.tok_emb.weight.zero_()  # tied head -> all logits zero
    idx = torch.randint(0, vocab, (4, 8))
    logits, loss = lm(idx, idx)
    assert torch.count_nonzero(logits) == 0
    assert loss.item() == pytest.approx(math.log(vocab), abs=1e-6)


def test_loss_accepts_non_contiguous_targets():
    # WindowSampler returns y = window[:, 1:], a strided view.
    torch.manual_seed(3)
    lm = TinyTransformerLM(ModelConfig(
        vocab_size=11, d_model=16, num_heads=2, num_layers=1, d_ff=32,
        block_size=8, dropout=0.0))
    window = torch.randint(0, 11, (4, 9))
    x, y = window[:, :-1], window[:, 1:]
    assert not y.is_contiguous()
    _, loss = lm(x, y)
    _, loss_ref = lm(x.contiguous(), y.contiguous())
    assert torch.equal(loss, loss_ref)


def test_overfits_ab_pattern():
    torch.manual_seed(1)
    data = torch.tensor([0, 1] * 200)
    cfg = ModelConfig(vocab_size=2, d_model=16, num_heads=2, num_layers=2,
                      d_ff=32, block_size=8, dropout=0.0)
    lm = TinyTransformerLM(cfg)
    opt = torch.optim.Adam(lm.parameters(), lr=3e-3)
    for _ in range(300):
        ix = torch.randint(0, len(data) - cfg.block_size - 1, (16,))
        xb = torch.stack([data[i:i + cfg.block_size] for i in ix])
        yb = torch.stack([data[i + 1:i + 1 + cfg.block_size] for i in ix])
        _, loss = lm(xb, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05, loss.item()


def test_v1_parameter_count():
    lm = TinyTransformerLM(ModelConfig(vocab_size=102))
    assert sum(p.numel() for p in lm.parameters()) == 4_759_040


def test_rejects_sequence_longer_than_block():
    lm = TinyTransformerLM(ModelConfig(
        vocab_size=5, d_model=8, num_heads=2, num_layers=1, d_ff=16,
        block_size=4))
    with pytest.raises(ValueError):
        lm(torch.zeros(1, 5, dtype=torch.long))


def test_sampling_modes():
    torch.manual_seed(4)
    lm = TinyTransformerLM(ModelConfig(
        vocab_size=7, d_model=8, num_heads=2, num_layers=1, d_ff=16,
        block_size=4, dropout=0.0)).eval()
    idx = torch.zeros(1, 1, dtype=torch.long)
    greedy = lm.generate(idx, 10, greedy=True)
    assert greedy.shape == (1, 11)
    assert torch.equal(greedy, lm.generate(idx, 10, greedy=True))
    g = torch.Generator().manual_seed(0)
    top1 = lm.generate(idx, 10, temperature=0.7, top_k=1, generator=g)
    assert torch.equal(top1, greedy)  # top-1 sampling is greedy
    sampled = lm.generate(idx, 10, temperature=1.5, top_k=3, generator=g)
    assert sampled.shape == (1, 11) and int(sampled.max()) < 7


@pytest.mark.parametrize("vocab", [86, 2000])
def test_initial_loss_near_log_vocab_at_c1(vocab):
    # §8 (v0.7): tied embedding N(0, d_model^-1/2). With N(0, 1) the C1
    # char model started at ~186 nats/token.
    torch.manual_seed(1)
    lm = TinyTransformerLM(ModelConfig(vocab_size=vocab)).eval()
    assert lm.tok_emb.weight.std().item() == pytest.approx(256 ** -0.5,
                                                           rel=0.05)
    idx = torch.randint(0, vocab, (8, 256))
    with torch.no_grad():
        _, loss = lm(idx, torch.randint(0, vocab, (8, 256)))
    print(f"V={vocab}: initial loss {loss.item():.4f}, "
          f"ln V {math.log(vocab):.4f}")
    assert abs(loss.item() - math.log(vocab)) < 0.75
