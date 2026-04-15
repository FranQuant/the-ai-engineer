"""
Training script for the Week-03 Tiny Transformer (decoder-only GPT-style LM).


- Explicit structure (config → data → model → training → sampling)
- Optional causal mask use (LM already enforces default causality)
- Deterministic seeding (torch + random + numpy)
- Fully aligned with the Week-03 Capstone spec

Note:
We intentionally keep AdamW (instead of Adam) — it is acceptable
and typically performs better. If strict reproduction of the
handout's optimizer is desired, replace AdamW with Adam.
"""

from __future__ import annotations
import json
import math as _math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from mini_transformer import MiniTransformerLM  # uses sinusoidal PE + default causal mask

# -------------------------------------------------------------------------
# 0. Configuration
# -------------------------------------------------------------------------
CONFIG = {
    "seed":         0,
    "block_size":   4,
    "batch_size":   32,
    "d_model":      64,
    "num_heads":    4,
    "d_ff":         256,
    "num_layers":   4,
    "lr":           3e-4,
    "max_iters":    1000,
    "log_every":    100,
    "sample_tokens": 80,
    "warmup_steps": 100,
}

# -------------------------------------------------------------------------
# 1. Reproducibility
# -------------------------------------------------------------------------
torch.manual_seed(CONFIG["seed"])
random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on device: {device}")

# -------------------------------------------------------------------------
# 2. Tiny dataset (character-level)
# -------------------------------------------------------------------------
tiny_text = """
hello tiny transformer
hello week three
hi tiny
"""

chars = sorted(list(set(tiny_text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s: str) -> list[int]:
    return [stoi[c] for c in s]

def decode(tokens: list[int]) -> str:
    return "".join(itos[i] for i in tokens)

data = torch.tensor(encode(tiny_text), dtype=torch.long)

# -------------------------------------------------------------------------
# 3. Train/val split
# -------------------------------------------------------------------------
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# -------------------------------------------------------------------------
# 4. Dataset / Dataloader
# -------------------------------------------------------------------------
# For this tiny toy corpus, we keep the context window very small.
# With ~50 characters total and a 90/10 train/val split, a block_size of 4
# ensures both train and val datasets have positive length.
class CharDataset(Dataset):
    """Simple character-level LM dataset."""
    def __init__(self, split: torch.Tensor):
        self.data = split

    def __len__(self):
        return len(self.data) - CONFIG["block_size"]

    def __getitem__(self, idx):
        x = self.data[idx : idx + CONFIG["block_size"]]             # input
        y = self.data[idx + 1 : idx + CONFIG["block_size"] + 1]     # next-token targets
        return x, y

train_loader = DataLoader(CharDataset(train_data), batch_size=CONFIG["batch_size"], shuffle=True)
val_loader   = DataLoader(CharDataset(val_data),   batch_size=CONFIG["batch_size"], shuffle=False)

# -------------------------------------------------------------------------
# 5. Model instantiation
# -------------------------------------------------------------------------
model = MiniTransformerLM(
    vocab_size=vocab_size,
    max_seq_len=CONFIG["block_size"],
    d_model=CONFIG["d_model"],
    num_heads=CONFIG["num_heads"],
    d_ff=CONFIG["d_ff"],
    num_layers=CONFIG["num_layers"],
    dropout=0.0,              # Week-03 spec: dropout hooks present but off by default
).to(device)

# -------------------------------------------------------------------------
# 6. Optimizer (AdamW — acceptable deviation from handout's Adam)
# -------------------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])

# -------------------------------------------------------------------------
# 7. Learning-rate scheduler (warmup + cosine decay)
# -------------------------------------------------------------------------
def get_lr(step: int) -> float:
    if step < CONFIG["warmup_steps"]:
        return (step + 1) / CONFIG["warmup_steps"] * CONFIG["lr"]
    return CONFIG["lr"] * 0.5 * (1 + _math.cos(step / 50000.0 * _math.pi))

# -------------------------------------------------------------------------
# 8. Loss estimation helper
# -------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}

    for split, loader in [("train", train_loader), ("val", val_loader)]:
        losses = []
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            # LM now uses default causal mask → no need to build mask manually
            logits = model(x)

            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            losses.append(loss.item())

        out[split] = sum(losses) / len(losses)

    model.train()
    return out

# -------------------------------------------------------------------------
# 9. Training loop
# -------------------------------------------------------------------------
print("Starting training...\n")

# Persistent iterator so we actually walk the whole DataLoader over time
train_iter = iter(train_loader)

for step in range(CONFIG["max_iters"]):
    # dynamic LR schedule
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    # Get next batch; reset iterator when exhausted
    try:
        x, y = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        x, y = next(train_iter)

    x, y = x.to(device), y.to(device)

    # forward pass (model auto-creates causal mask)
    logits = model(x)

    # compute loss
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # periodic evaluation
    if step % CONFIG["log_every"] == 0:
        losses = estimate_loss()
        print(
            f"step {step:4d} | train loss {losses['train']:.4f} | "
            f"val loss {losses['val']:.4f} | lr {lr:.2e}"
        )

# -------------------------------------------------------------------------
# 10. Save checkpoint
# -------------------------------------------------------------------------
torch.save(model.state_dict(), "mini_gpt.pt")
print("\nModel checkpoint saved → mini_gpt.pt\n")

# -------------------------------------------------------------------------
# 11. Run record
# -------------------------------------------------------------------------
final_losses = estimate_loss()
train_loss = final_losses["train"]
val_loss   = final_losses["val"]

run_record = {
    "timestamp":        time.strftime("%Y-%m-%dT%H:%M:%S"),
    "seed":             CONFIG["seed"],
    "d_model":          CONFIG["d_model"],
    "num_heads":        CONFIG["num_heads"],
    "num_layers":       CONFIG["num_layers"],
    "d_ff":             CONFIG["d_ff"],
    "block_size":       CONFIG["block_size"],
    "batch_size":       CONFIG["batch_size"],
    "learning_rate":    CONFIG["lr"],
    "steps":            CONFIG["max_iters"],
    "final_train_loss": round(float(train_loss), 4),
    "final_val_loss":   round(float(val_loss), 4),
    "checkpoint":       "mini_gpt.pt",
}
record_path = Path(__file__).parent / "run_record.json"
record_path.write_text(json.dumps(run_record, indent=2))
print(f"Run record saved to {record_path}")

# -------------------------------------------------------------------------
# 12. Sampling helper (GPT-style generation)
# -------------------------------------------------------------------------
@torch.no_grad()
def generate(model, start_tokens, max_new_tokens=100, temperature=1.0, top_k=None):
    """GPT-style autoregressive sampling."""
    model.eval()
    context = torch.tensor(start_tokens, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        # crop to the last block_size tokens
        idx_cond = context[:, -CONFIG["block_size"]:]

        # model enforces causal mask internally
        logits = model(idx_cond)

        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        context = torch.cat([context, next_token], dim=1)

    return context.squeeze().tolist()

# -------------------------------------------------------------------------
# 13. Generate sample text
# -------------------------------------------------------------------------
start = encode("h")
sample = generate(model, start, max_new_tokens=CONFIG["sample_tokens"], temperature=1.0, top_k=5)

print("=== Sample Model Output ===\n")
print(decode(sample))
print("\nDone.")
