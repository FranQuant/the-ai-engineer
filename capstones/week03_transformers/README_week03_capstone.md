<img src="https://theaiengineer.dev/tae_logo_gw_flatter.png" width="35%" align="right">

# Week 3 Capstone — A Tiny Transformer, From Scratch, on FOMC Language

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/the-ai-engineer/blob/main/capstones/week03_transformers/week03_tiny_transformer.ipynb)

**TAE Program — Core Track** · Implements scaled dot-product attention, causal
masking, multi-head attention, and a tiny decoder-only transformer language
model, all from scratch in PyTorch (no transformer libraries). Trained on a
real, provenance-tracked corpus of FOMC statements and minutes through Chair
Powell's final meeting (April 29, 2026).

## What it does

Builds every component of the transformer architecture step by step:

- **Scaled dot-product attention** with a hand-computable three-token example
  verified numerically, and the causal mask explicitly checked (the first
  query's attention row is forced to `[1, 0, 0]` — a degeneracy, not an
  equality to the unmasked case)
- **Self-attention** and **multi-head attention** with structural sanity checks
- **Position-wise FFN**, **sinusoidal positional encoding**, and **pre-LN
  residual** wiring assembled into a `TransformerBlock`
- A **tiny decoder-only LM** (4.76M parameters, weight-tied embedding/output,
  d_model=256, 8 heads, 6 layers) trained with Adam + cosine LR schedule,
  warmup, weight decay, and gradient clipping
- **Guarded training**: default loads the committed checkpoint for validation;
  set `TRAIN = True` to retrain from scratch
- **Sampling gallery**: greedy decoding and temperature sampling (τ = 0.7,
  1.0, 1.5) on FOMC-flavored prompts
- **Run record** (JSON) capturing config, seed, final metrics, and artifact paths

## Results

Trained for 6,000 steps on the FOMC corpus (6.73M characters, ~5.4M train /
~0.6M val):

| Metric | Value |
|---|---|
| Final train loss | 0.786 |
| Final val loss | 0.837 |
| Train/val gap | 0.05 (no overfitting) |
| Wall-clock time | ~70 min on Apple M4 MPS |

Samples at τ = 0.7 show recognizable FOMC structure: correct document
markers, real committee vocabulary, and plausible meeting-minutes phrasing.
At this loss level some short phrases are likely near-memorized; see
Limitations in the notebook.

## Corpus

`fomc_training_corpus.txt` — 210 documents (93 FOMC statements 2015–2026,
117 FOMC minutes 2010–2026), 6.73 MB, frozen at the last Powell-era meeting
for stylistic homogeneity. Built by `build_fomc_corpus.py`,
`build_fomc_minutes_corpus.py`, and `merge_fomc_corpus.py` in this directory;
full provenance in `fomc_training_corpus_manifest.json`.

## How to run

Single click — the Colab badge above. The notebook installs `tqdm` in its
first cell and runs end-to-end without edits in under 5 minutes on a Colab
GPU (checkpoint already committed; full retraining requires setting
`TRAIN = True` and takes ~15–20 min on a T4).

Locally: `jupyter lab week03_tiny_transformer.ipynb` → Run All (with
`TRAIN = False`, validates the committed checkpoint in seconds).

## Deliverables

- Notebook: [`week03_tiny_transformer.ipynb`](https://github.com/FranQuant/the-ai-engineer/blob/main/capstones/week03_transformers/week03_tiny_transformer.ipynb)
- Corpus builders: `build_fomc_corpus.py`, `build_fomc_minutes_corpus.py`, `merge_fomc_corpus.py`
- Corpus + manifest: `fomc_training_corpus.txt`, `fomc_training_corpus_manifest.json`
- Repo: https://github.com/FranQuant/the-ai-engineer

## Honest notes

- Checkpoint binary not committed (regenerated on Colab run, same policy as
  Week 2; only Week 4's checkpoint is force-committed per the repo's gitignore
  policy)
- Pre-2010 FOMC minutes are absent from the corpus (legacy URL scheme on the
  Fed's site, documented in the manifest's `known_coverage_gap` field)
- Character-level tokenization; a lightweight tokenizer would likely improve
  sample quality — noted as future work, not attempted here
