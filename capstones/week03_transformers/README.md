<img src="https://theaiengineer.dev/tae_logo_gw_flatter.png" width="35%" align="right">

# Week 3 Capstone — How Surprising Is the Fed?

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/the-ai-engineer/blob/main/capstones/week03_transformers/week03_fomc_surprise.ipynb)

A tiny decoder-only transformer, built from scratch (attention, multi-head attention, sinusoidal positions, Pre-LN blocks), trained only on FOMC statements and minutes public by the end of 2019. It then scores every later FOMC document in bits per character (BPC): how surprising its wording is, given the Committee's earlier language. Three hypotheses were pre-registered in [DESIGN.md](DESIGN.md) before any training on the real split.

## Results (pre-registered confirmatory run, [runs/confirmatory_results.json](runs/confirmatory_results.json))

- **H1, drift (far − near BPC > 0): contradicted.** Δ₁ = −0.0141 (95% CI −0.0639 to +0.0521).
- **H2, minutes more surprising than statements: supported.** Δ₂ = +0.1233 (95% CI +0.1008 to +0.1438), 35 matched meetings.
- **H3, ranking stable across tokenizers: supported.** ρ(char, BPE) = 0.936 (95% CI 0.877 to 0.964).

The submission run (tag `week03-v3`, [runs/week03-v3/](runs/week03-v3/)) gives the same three labels: Δ₁ = −0.0234, Δ₂ = +0.1357, ρ = 0.923.

## What the notebook shows

The handout's attention example with and without the causal mask; checks of every module; char and BPE transformers trained with a fixed recipe and logged train/held-out loss; a saved and reloaded checkpoint; a sampling gallery (greedy and temperature); per-document scoring against a 5-gram baseline; bootstrap tests; and a `results.json` run record.

## Run

**Colab:** badge → *Runtime → Change runtime type → T4 GPU* → *Run all* (about 13 minutes). The notebook clones this repository at tag `week03-v3` and verifies the data by SHA-256.

**Locally (CPU):** set `MODE = "smoke"` in the first code cell: the whole pipeline in under a minute on a fake split (not a result). Tests: `cd capstones/week03_transformers && python -m pytest tests`.

## Layout

| Path | Contents |
|---|---|
| `week03_fomc_surprise.ipynb` | The notebook (saved without outputs) |
| `DESIGN.md` | Pre-registration: split, scoring, statistics, settings |
| `src/` | `model.py` (attention → TinyTransformerLM), `train.py`, `data.py`, `bpe.py`, `evaluate.py`, `ngram.py`, `analysis.py` |
| `corpus/` | Frozen FOMC corpus, provenance manifest, meeting-level split; `builders/` rebuilds them |
| `tests/` | pytest suite |
| `runs/` | confirmatory run (`confirmatory_results.json`, executed `confirmatory_run.ipynb`, `figures/`) and the submission run (`week03-v3/`: executed notebook, results, checkpoint) |

## Data

210 FOMC documents (93 statements, 117 minutes) from federalreserve.gov. Split by meeting into train (to 2019, 104 documents), near (2020–21, 32) and far (2022 on, 70); 4 documents from meetings within 30 days of a split boundary are excluded.

## Limitations

Textual surprise is not market surprise; one cutoff; one seed; small models (absolute BPC not comparable to published LMs). Full list in DESIGN.md §14.
