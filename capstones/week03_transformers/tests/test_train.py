"""Frozen training recipe checks (DESIGN.md §8)."""

import pytest
import torch

from data import CharVocab, Document, NMonitorSampler, WindowSampler
from train import (ARCH, TrainConfig, _step_and_update, lr_at, make_model,
                   monitor_steps, train)

FROZEN = TrainConfig()


def test_frozen_settings():
    assert (FROZEN.steps, FROZEN.batch_size, FROZEN.seed) == (3696, 64, 1)
    assert (FROZEN.weight_decay, FROZEN.grad_clip) == (0.1, 1.0)
    assert (FROZEN.monitor_evals, FROZEN.monitor_batches,
            FROZEN.monitor_batch_size) == (10, 20, 64)
    assert ARCH == {"d_model": 256, "num_layers": 6, "num_heads": 8,
                    "d_ff": 1024}
    lm = make_model(vocab_size=50)
    assert lm.block_size == 256
    assert lm.blocks[0].ff.net[-1].p == 0.1  # dropout


def test_lr_schedule_values():
    assert lr_at(0, FROZEN) == pytest.approx(3e-4 / 200)
    assert lr_at(199, FROZEN) == pytest.approx(3e-4)
    assert lr_at(200, FROZEN) == pytest.approx(3e-4)
    assert lr_at(FROZEN.steps - 1, FROZEN) == pytest.approx(3e-5)
    warm = [lr_at(s, FROZEN) for s in range(200)]
    decay = [lr_at(s, FROZEN) for s in range(200, FROZEN.steps)]
    assert all(a < b for a, b in zip(warm, warm[1:]))
    assert all(a > b for a, b in zip(decay, decay[1:]))
    for bad in (-1, FROZEN.steps):
        with pytest.raises(ValueError):
            lr_at(bad, FROZEN)


def test_monitor_steps_evenly_spaced():
    steps = monitor_steps(FROZEN)
    assert len(steps) == 10 and steps[-1] == FROZEN.steps
    gaps = {b - a for a, b in zip([0] + steps, steps)}
    assert gaps <= {369, 370}


def test_config_rejects_steps_within_warmup():
    for steps in (200, 201):
        with pytest.raises(ValueError):
            TrainConfig(steps=steps)
    assert lr_at(201, TrainConfig(steps=202)) == pytest.approx(3e-5)


def _toy(split, n=3):
    return [Document(f"{split}{i}", "statement", f"2015-0{i + 1}-01", split,
                     "abcabcabd" * 30) for i in range(n)]


@pytest.fixture(scope="module")
def toy():
    vocab = CharVocab._from_chars("abcd")
    t = WindowSampler.from_documents(_toy("T"), vocab, 16)
    n = NMonitorSampler.from_documents(_toy("N", 2), vocab, 16)
    return vocab, t, n


def _cfg():
    return TrainConfig(steps=120, batch_size=16, warmup_steps=10,
                       monitor_evals=4, monitor_batches=2,
                       monitor_batch_size=8)


def _tiny(vocab):
    return make_model(vocab.vocab_size, block_size=16, d_model=32,
                      num_layers=2, num_heads=2, d_ff=64)


def test_tiny_cpu_training_reduces_loss(toy):
    vocab, t, n = toy
    cfg = _cfg()
    result = train(_tiny(vocab), t, n, cfg, torch.device("cpu"))
    assert result.precision == "fp32" and result.steps == cfg.steps
    assert result.skipped_updates == 0  # no GradScaler in fp32
    assert len(result.train_loss) == len(result.lr) == cfg.steps
    first = sum(result.train_loss[:10]) / 10
    last = sum(result.train_loss[-10:]) / 10
    assert last < 0.5 * first, (first, last)
    assert [m["step"] for m in result.monitor] == [30, 60, 90, 120]
    assert result.monitor[-1]["loss"] < result.monitor[0]["loss"]
    # The optimizer applied exactly the schedule.
    assert result.lr == pytest.approx([lr_at(s, cfg)
                                       for s in range(cfg.steps)])


def test_training_is_deterministic(toy):
    vocab, t, n = toy
    cfg = TrainConfig(steps=15, batch_size=4, warmup_steps=5,
                      monitor_evals=1, monitor_batches=1,
                      monitor_batch_size=4)
    a = train(_tiny(vocab), t, n, cfg, torch.device("cpu"))
    b = train(_tiny(vocab), t, n, cfg, torch.device("cpu"))
    assert a.train_loss == b.train_loss and a.monitor == b.monitor


def test_samplers_cannot_be_swapped(toy):
    vocab, t, n = toy
    cpu = torch.device("cpu")
    with pytest.raises(TypeError, match="WindowSampler"):
        train(_tiny(vocab), n, None, _cfg(), cpu)
    with pytest.raises(TypeError, match="NMonitorSampler"):
        train(_tiny(vocab), t, t, _cfg(), cpu)


@pytest.mark.parametrize("bad", [False, True])
def test_skipped_update_detected(bad):
    w = torch.nn.Parameter(torch.ones(3))
    opt = torch.optim.SGD([w], lr=0.1)
    scaler = torch.amp.GradScaler("cpu", enabled=True)
    loss = (w * (float("inf") if bad else 1.0)).sum()
    scaler.scale(loss).backward()
    assert _step_and_update(scaler, opt) is bad
    assert torch.equal(w.detach(), torch.ones(3)) is bad  # skipped: no update


def test_disabled_scaler_never_skips():
    w = torch.nn.Parameter(torch.ones(3))
    opt = torch.optim.SGD([w], lr=0.1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    w.sum().backward()
    assert _step_and_update(scaler, opt) is False
