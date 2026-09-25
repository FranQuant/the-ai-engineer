"""Tiny decoder-only transformer (DESIGN.md v0.4, §5 and §11).

Ported from the v1 notebook. Changes: scaled_dot_product_attention accepts a
boolean mask (True = keep) or an additive float mask (0 / -inf); label
smoothing removed (plain cross-entropy, §5); top-p sampling dropped (greedy,
temperature and top-k kept).
"""

from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclasses.dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 6
    d_ff: int = 1024
    block_size: int = 256
    dropout: float = 0.1
    # Training speed; verified equivalent to the hand-written attention.
    use_fused_attention: bool = True


def make_causal_mask(T: int, device: torch.device | str | None = None,
                     additive: bool = False) -> torch.Tensor:
    """(T, T) causal mask: boolean (True = keep) or additive (0 / -inf)."""
    keep = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
    if not additive:
        return keep
    return torch.zeros(T, T, device=device).masked_fill(~keep, float("-inf"))


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Return (Y, S, A): output, max-shifted masked scores, weights.

    mask is either boolean (True = keep) or an additive float mask
    (0 = keep, -inf = drop), broadcastable to the scores.
    """
    d_k = Q.shape[-1]
    S = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        if mask.dtype == torch.bool:
            S = S.masked_fill(~mask, float("-inf"))
        elif mask.is_floating_point():
            S = S + mask
        else:
            raise TypeError(f"mask must be bool or float, got {mask.dtype}")
    S = S - S.max(dim=-1, keepdim=True).values
    A = torch.softmax(S, dim=-1)
    return A @ V, S, A


class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k=None, d_v=None, causal=True):
        super().__init__()
        d_k, d_v = d_k or d_model, d_v or d_model
        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_v, bias=False)
        self.causal = causal

    def forward(self, x):
        T = x.shape[1]
        Q, K, V = self.W_Q(x), self.W_K(x), self.W_V(x)
        mask = make_causal_mask(T, x.device) if self.causal else None
        y, _, _ = scaled_dot_product_attention(Q, K, V, mask=mask)
        return y


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0, causal=True,
                 use_fused=False):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads, self.d_head = num_heads, d_model // num_heads
        self.causal = causal
        # Default False: unit tests use the hand-written path.
        self.use_fused = use_fused
        self.proj_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj_out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.proj_qkv(x).view(B, T, 3, self.num_heads, self.d_head)
        Q, K, V = (t.transpose(1, 2) for t in qkv.unbind(dim=2))
        if self.use_fused:
            Y = F.scaled_dot_product_attention(Q, K, V, is_causal=self.causal)
        else:
            mask = make_causal_mask(T, x.device) if self.causal else None
            Y, _, _ = scaled_dot_product_attention(Q, K, V, mask=mask)
        Y = Y.transpose(1, 2).contiguous().view(B, T, D)
        return self.dropout(self.proj_out(Y))


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                 nn.Linear(d_ff, d_model), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LN block: x + MHA(LN(x)), then x + FFN(LN(x))."""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.0, causal=True,
                 use_fused=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout=dropout,
                                       causal=causal, use_fused=use_fused)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.shape[1], :]


class TinyTransformerLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.block_size = cfg.block_size
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_enc = PositionalEncoding(cfg.d_model, max_len=cfg.block_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.num_heads, cfg.d_ff,
                             cfg.dropout, causal=True,
                             use_fused=cfg.use_fused_attention)
            for _ in range(cfg.num_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

    def forward(self, idx, targets=None):
        if idx.shape[1] > self.block_size:
            raise ValueError(
                f"sequence length {idx.shape[1]} > block_size "
                f"{self.block_size}")
        z = self.pos_enc(self.tok_emb(idx))
        for blk in self.blocks:
            z = blk(z)
        logits = self.head(self.ln_f(z))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, greedy=False,
                 top_k=None, generator=None):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.block_size:])
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if greedy:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    kth = torch.topk(logits, k, dim=-1).values[:, -1, None]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                next_id = torch.multinomial(F.softmax(logits, dim=-1), 1,
                                            generator=generator)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
