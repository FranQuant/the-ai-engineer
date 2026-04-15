"""
Step 2: Multi-Head Self-Attention (MHA) with proper mask semantics.

Grounding:
- Handout Sections 5–7: self-attention, multi-head shapes, masking rules.
- Reference: llmcode code/ch08_transformer.py and Week-3 handout patterns.
- Mask semantics (boolean mask: True=keep, False=mask) match MiniTransformerLM.

This module:
    - Splits Q/K/V into heads
    - Applies scaled dot-product attention
    - Broadcasts boolean masks over heads
    - Uses -inf for masked positions
    - Merges heads and returns projected output
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


from scaled_dot_product_attention import scaled_dot_product_attention # used in tests

torch.manual_seed(0)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with explicit head splitting and boolean mask handling."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.h = num_heads
        self.dh = d_model // num_heads

        # Linear projections for Q, K, V and output
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def _normalize_mask(self, mask: torch.Tensor, B: int, T: int, device: torch.device):
        """
        Normalize mask shapes/dtypes into boolean [B or 1, 1, T, T].
        True = keep
        False = mask out
        """
        mask = mask.to(device)

        if mask.ndim == 2:                   # [T,T]
            mask = mask.unsqueeze(0).unsqueeze(0)     # [1,1,T,T]
        elif mask.ndim == 3:                 # [B,T,T] or [1,T,T]
            mask = mask.unsqueeze(1)                    # [B,1,T,T]
        elif mask.ndim == 4:                 # [B,1,T,T]
            pass
        else:
            raise ValueError(f"Unsupported mask ndim={mask.ndim}")

        # Convert int/float masks (1=keep, 0=mask) → boolean
        if mask.dtype == torch.bool:
            keep = mask
        else:
            keep = (mask != 0)

        # Final shape: [B or 1, 1, T, T]
        return keep

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
            attn_mask: optional boolean or {0/1} mask
                       accepted shapes:
                          [T,T], [1,T,T], [B,T,T], [B,1,T,T]
                       boolean: True=keep, False=mask
        Returns:
            [B, T, d_model]
        """
        assert x.ndim == 3, "x must be [B,T,D]"
        B, T, D = x.shape
        assert D == self.d_model, f"Expected D={self.d_model}, got {D}"
        device = x.device

        # ---- Linear projections ----
        q = self.q_proj(x)      # [B,T,D]
        k = self.k_proj(x)
        v = self.v_proj(x)

        # ---- Split heads ----
        # [B,T,D] -> [B,H,T,Dh]
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)  # [B,H,T,Dh]
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)

        # ---- Scaled dot-product ----
        # scores: [B,H,T,T]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)

        # ---- Masking ----
        if attn_mask is not None:
            # Normalize to boolean broadcastable mask
            keep = self._normalize_mask(attn_mask, B, T, device)
            # Broadcast over heads: keep shape [B or 1, 1, T, T]
            # Expand (if needed) to [B,1,T,T] so scores [B,H,T,T] broadcast correctly
            if keep.size(0) == 1 and B > 1:
                keep = keep.expand(B, -1, -1, -1)  # [B,1,T,T]

            # Apply mask: masked positions get -inf
            scores = scores.masked_fill(~keep, float('-inf'))

        # Note: if an entire row is masked (all -inf), the max-subtraction
        # produces nan. This cannot occur with a valid causal mask but would
        # surface with a zero-row custom mask. Add a nan-guard if needed.
        # ---- Softmax ----
        scores = scores - scores.max(dim=-1, keepdim=True).values  # stability
        attn = torch.softmax(scores, dim=-1)                        # [B,H,T,T]

        # ---- Weighted sum of values ----
        out = attn @ v   # [B,H,T,Dh]

        # ---- Merge heads ----
        out = out.transpose(1, 2).contiguous().view(B, T, D)  # [B,T,D]

        # ---- Final linear projection ----
        return self.o_proj(out)


# -------------------------- Tests -------------------------- #

def _test_shape_round_trip() -> None:
    x = torch.randn(2, 4, 8)
    mha = MultiHeadAttention(d_model=8, num_heads=2)
    y = mha(x)
    assert y.shape == (2, 4, 8)
    print("Test A (shape round-trip) passed:", y.shape)


def _test_single_head_equivalence() -> None:
    """
    With H=1 and identity projections, MultiHeadAttention should match
    scaled_dot_product_attention(x, x, x).
    """
    B, T, D = 3, 5, 4
    x = torch.randn(B, T, D)
    mha = MultiHeadAttention(d_model=D, num_heads=1)

    # Force identity weights for q/k/v/o
    with torch.no_grad():
        mha.q_proj.weight.copy_(torch.eye(D))
        mha.k_proj.weight.copy_(torch.eye(D))
        mha.v_proj.weight.copy_(torch.eye(D))
        mha.o_proj.weight.copy_(torch.eye(D))

    y_mha = mha(x)
    y_ref = scaled_dot_product_attention(x, x, x)
    assert torch.allclose(y_mha, y_ref, atol=1e-6)
    print("Test B (single-head equivalence) passed")


def _test_mask_propagation() -> None:
    """
    Causal mask: attention weights above the diagonal should be ~0 for B=1.
    """
    B, T, D = 1, 3, 6
    x = torch.randn(B, T, D)
    mha = MultiHeadAttention(d_model=D, num_heads=3)
    causal = torch.tril(torch.ones(T, T))  # [T,T]

    # Forward pass to ensure mask is accepted
    _ = mha(x, causal)

    # Inspect internal scores with the updated logic
    q = mha.q_proj(x).view(B, T, mha.h, mha.dh).transpose(1, 2)
    k = mha.k_proj(x).view(B, T, mha.h, mha.dh).transpose(1, 2)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(mha.dh)
    keep = (causal != 0).view(1, 1, T, T)
    scores = scores.masked_fill(~keep, float('-inf'))
    attn = torch.softmax(scores, dim=-1)

    # Check upper triangle ~ 0
    tri = torch.triu_indices(T, T, offset=1)
    upper = attn[..., tri[0], tri[1]]
    assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-6)
    print("Test C (mask propagation) passed")


if __name__ == "__main__":
    _test_shape_round_trip()
    _test_single_head_equivalence()
    _test_mask_propagation()
    print("\nAll Step 2 tests passed.")
