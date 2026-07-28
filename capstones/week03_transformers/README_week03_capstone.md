<img src="https://theaiengineer.dev/tae_logo_gw_flatter.png" width="35%" align="right">

# Week 3 Capstone — A Tiny Transformer, From Scratch, on FOMC Language

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/the-ai-engineer/blob/main/capstones/week03_transformers/week03_tiny_transformer.ipynb)

Attention, causal masking, multi-head attention, and training — all implemented
from scratch in PyTorch, no transformer libraries. Trained on FOMC statements
and minutes through Chair Powell's final meeting (Apr 29, 2026), just before
the incoming chair's shift to shorter, no-forward-guidance statements.

## What's inside

| Section | Contents |
|---|---|
| Attention & blocks | SDPA, causal mask, self-attn, multi-head attn — each numerically verified |
| Model | 4.76M-param decoder-only LM, weight-tied, cosine LR + resume phase |
| Training | Guarded (`TRAIN` flag), best-checkpoint tracking, lr/grad-norm logging |
| Sampling | Greedy + temperature, `top_k=40` |
| Extension | From-scratch BPE tokenizer (5.76M-param matched model), chunk-split, fair comparison |

## Results

| | Character-level | BPE |
|---|---|---|
| Params | 4.76M | 5.76M |
| Val loss (resumed) | 1.239 | 2.509 |
| **Bits/char** | **1.79** | **1.20** |

**BPE wins by ~33%, reproduced on two GPUs (T4 and L4)** — reverses two
earlier rounds where char-level won 2-3x. Likely driver:
`weight_decay` 0.01→0.08 for BPE (sparse 4k-vocab embeddings were
under-regularized). L4 also ran ~1.9x faster (34 vs 63 min) — T4 stays the
default for free-tier reproducibility. Full reasoning in the notebook's
Extension Conclusion.

Neither model writes fully coherent prose at this scale — expected. Samples
do use real FOMC names and correct procedural/policy language.

## Pipeline

<img src="week03_pipeline_diagram.png" alt="Week 3 pipeline: shared attention/transformer code, branching into character-level and BPE paths, merging at Fair Comparison" width="700">

Every stage is verified against a hand-computable example before the next is
built on top of it. The BPE path reuses the identical model code, trained on
the same corpus, for a direct comparison rather than a separate model.

## Corpus

`fomc_training_corpus.txt` — 210 docs (93 statements + 117 minutes,
2010–2026), 6.73M chars. Built by `build_fomc_corpus.py` /
`build_fomc_minutes_corpus.py` / `merge_fomc_corpus.py`; provenance in
`fomc_training_corpus_manifest.json`.

Pre-2010 minutes aren't included — they live under a legacy URL scheme on the
Fed's site that this corpus's builder doesn't crawl (disclosed in the
manifest's `known_coverage_gap`). Because of that, BPE's larger vocabulary
gets much less data per entry than char-level's, which is likely why it
needed stronger regularization to train well.

## Run it

Colab badge above → `TRAIN = True` → Run All. Self-fetches the corpus, no
upload needed. ~1 hr on T4, ~35 min on L4.

## Notes

- Checkpoints not committed (regenerated per run, same as Week 2)
- BPE is a labeled extension, not a replacement — both models kept and
  reported, including the finding that reversed which one wins
- Metrics are from a small number of seeded runs, not averaged

<img src="https://theaiengineer.dev/tae_logo_gw_flatter.png" width="35%" align="right">
