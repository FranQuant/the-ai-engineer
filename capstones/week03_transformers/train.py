"""Training recipe frozen in DESIGN.md v0.8 §8.

AdamW, lr 3e-4 with 200 linear warm-up steps, then cosine decay to 3e-5 at
the final step; weight decay 0.1; dropout 0.1 (ModelConfig default);
gradient-norm clip 1.0; seed 1; a fixed step count, and the final weights
are used. fp16 autocast with a GradScaler on CUDA, fp32 on CPU. A step is
one optimizer iteration: updates the GradScaler skips are counted and
reported, not replaced.

N monitoring: 10 evenly spaced evaluations, each on 20 batches of 64
windows from an NMonitorSampler, the last one after the final step. The
losses are logged for plotting only; nothing reads them back (§2, §8).
"""

from __future__ import annotations

import dataclasses
import math
import time
import warnings
from typing import Any

import torch

from data import NMonitorSampler, WindowSampler
from model import ModelConfig, TinyTransformerLM

# §8 frozen compute settings (Phase 3 pilot, pilot/phase3_timing.json)
ARCH = {"d_model": 256, "num_layers": 6, "num_heads": 8, "d_ff": 1024}
BLOCK_SIZE = 256
BPE_VOCAB = 2000


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    steps: int = 3696
    batch_size: int = 64
    lr: float = 3e-4
    warmup_steps: int = 200
    final_lr: float = 3e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 1
    monitor_evals: int = 10
    monitor_batches: int = 20
    monitor_batch_size: int = 64

    def __post_init__(self):
        if self.steps < self.warmup_steps + 2:
            raise ValueError("need at least one cosine step after warm-up")
        if not 1 <= self.monitor_evals <= self.steps:
            raise ValueError("monitor_evals must be in [1, steps]")


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Learning rate used for optimizer step `step` (0-based).

    Linear warm-up: steps 0..warmup-1 use lr x (step + 1) / warmup, so
    step warmup - 1 reaches the peak. Cosine decay from the peak at step
    warmup to final_lr at the final step, steps - 1.
    """
    if not 0 <= step < cfg.steps:
        raise ValueError(f"step {step} outside [0, {cfg.steps})")
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    decay_steps = cfg.steps - 1 - cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / decay_steps
    return cfg.final_lr + (cfg.lr - cfg.final_lr) * 0.5 * (
        1 + math.cos(math.pi * progress))


def monitor_steps(cfg: TrainConfig) -> list[int]:
    """Completed-step counts after which N is evaluated; the last is
    cfg.steps."""
    return [round(cfg.steps * (i + 1) / cfg.monitor_evals)
            for i in range(cfg.monitor_evals)]


def make_model(vocab_size: int, block_size: int = BLOCK_SIZE,
               seed: int = TrainConfig.seed, **arch: Any
               ) -> TinyTransformerLM:
    """A freshly initialized model; the seed fixes the initial weights."""
    torch.manual_seed(seed)
    return TinyTransformerLM(ModelConfig(
        vocab_size=vocab_size, block_size=block_size, **(arch or ARCH)))


@dataclasses.dataclass
class TrainResult:
    steps: int
    precision: str
    seconds: float
    train_loss: list[float]  # per step, nats/token
    lr: list[float]  # per step, as applied by the optimizer
    monitor: list[dict[str, float]]  # {"step", "loss"}: N, plot only
    skipped_updates: int  # fp16 overflow: counted, not replaced (§8)


def _step_and_update(scaler: torch.amp.GradScaler, opt) -> bool:
    """scaler.step + scaler.update; True if the scaler skipped the
    optimizer update (inf/NaN gradients), which lowers its scale."""
    before = scaler.get_scale()
    scaler.step(opt)
    scaler.update()
    return scaler.is_enabled() and scaler.get_scale() < before


@torch.no_grad()
def _monitor_loss(model, sampler: NMonitorSampler, cfg: TrainConfig,
                  device: torch.device, amp: bool) -> float:
    # Same windows at every evaluation, so the curve compares like with like.
    gen = torch.Generator().manual_seed(cfg.seed + 1)
    was_training = model.training
    model.eval()
    total = torch.zeros((), device=device)
    for _ in range(cfg.monitor_batches):
        x, y, _ = sampler.sample(cfg.monitor_batch_size, generator=gen)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            _, loss = model(x.to(device), y.to(device))
        total += loss.float()
    model.train(was_training)
    return total.item() / cfg.monitor_batches


def train(model: TinyTransformerLM, sampler: WindowSampler,
          monitor: NMonitorSampler | None, cfg: TrainConfig,
          device: torch.device, log_every: int = 0) -> TrainResult:
    """Train model in place for exactly cfg.steps steps (§8)."""
    if not isinstance(sampler, WindowSampler):
        raise TypeError("training windows must come from a (T-only) "
                        "WindowSampler")
    if monitor is not None and not isinstance(monitor, NMonitorSampler):
        raise TypeError("monitoring windows must come from an "
                        "NMonitorSampler")
    if sampler.block_size != model.block_size:
        raise ValueError("sampler and model block sizes differ")
    amp = device.type == "cuda"
    torch.manual_seed(cfg.seed)  # dropout
    gen = torch.Generator().manual_seed(cfg.seed)  # window sampling
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_at(min(s, cfg.steps - 1), cfg) / cfg.lr)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    losses = torch.empty(cfg.steps, device=device)  # no per-step sync
    lrs: list[float] = []
    skipped = 0
    evals = set(monitor_steps(cfg)) if monitor is not None else set()
    curve: list[dict[str, float]] = []
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        # A GradScaler skips optimizer steps whose fp16 gradients overflow
        # (usually the first); PyTorch then warns that the scheduler stepped
        # first. The call order below is optimizer, then scheduler.
        warnings.filterwarnings(
            "ignore", message="Detected call of `lr_scheduler.step",
            category=UserWarning)
        for step in range(cfg.steps):
            x, y, _ = sampler.sample(cfg.batch_size, generator=gen)
            x, y = x.to(device), y.to(device)
            with torch.autocast(device.type, dtype=torch.float16,
                                enabled=amp):
                _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            lrs.append(opt.param_groups[0]["lr"])
            skipped += _step_and_update(scaler, opt)
            sched.step()
            losses[step] = loss.detach().float()
            if step + 1 in evals:
                curve.append({"step": step + 1, "loss": _monitor_loss(
                    model, monitor, cfg, device, amp)})
            if log_every and (step + 1) % log_every == 0:
                print(f"step {step + 1:5d}/{cfg.steps}  "
                      f"loss {losses[step].item():.4f}  lr {lrs[-1]:.2e}"
                      + (f"  N {curve[-1]['loss']:.4f}"
                         if curve and curve[-1]["step"] == step + 1
                         else ""))
    if device.type == "cuda":
        torch.cuda.synchronize()
    return TrainResult(
        steps=cfg.steps, precision="fp16" if amp else "fp32",
        seconds=time.perf_counter() - t0, train_loss=losses.tolist(),
        lr=lrs, monitor=curve, skipped_updates=skipped)
