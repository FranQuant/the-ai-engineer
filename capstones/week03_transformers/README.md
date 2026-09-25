<img src="https://theaiengineer.dev/tae_logo_gw_flatter.png" width="35%" align="right">

# Week 3 Capstone: How Surprising Is the Fed?

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/the-ai-engineer/blob/main/capstones/week03_transformers/week03_fomc_surprise.ipynb)

**Per-document bits per character from a from-scratch tiny transformer on FOMC text.**
J. Francisco Salazar (FranQuant) · The AI Engineer, Week 3 · 2026-09-25

## Purpose

A decoder-only transformer, built from scratch to the TAE handout's specification, is used as a measurement instrument rather than as the contribution. It is trained only on FOMC statements and minutes that were public before a cutoff and never updated after it. Each later document is scored by its bits per character (BPC): how surprising its wording is, given the Committee's earlier language.

## Research question

Trained only on FOMC text available before a cutoff, how surprising is each later FOMC document, measured as per-document BPC? Is that surprise measure stable across tokenizers?

Three hypotheses were pre-registered in [`DESIGN.md`](DESIGN.md) before any model was trained on the real split. The split is by meeting, so a meeting's statement and minutes always fall in the same part: train T (available by 2019-12-31), near N (2020–2021), far F (from 2022).

- **H1 (drift):** mean BPC(F) − mean BPC(N) > 0.
- **H2 (genre):** within matched F meetings, minutes are more surprising than statements.
- **H3 (instrument robustness):** Spearman ρ between char- and BPE-transformer BPC on F ≥ 0.7.

## Results (confirmatory run)

The confirmatory result is the first real run, from tag `week03-v2-run1` on a Colab Tesla T4, archived as [`runs/confirmatory_results.json`](runs/confirmatory_results.json); all numbers below come from that file. **H1 is contradicted.** Δ₁ = -0.0141 (95% CI -0.0639 to +0.0521). The point estimate is ≤ 0, and the interval includes 0, so there is no evidence that later documents are more surprising. **H2 is supported.** Minutes exceed their meeting's statement by Δ₂ = +0.1233 (95% CI +0.1008 to +0.1438) bits per character, over 35 matched F meetings. **H3 is supported.** ρ(char, BPE) = 0.936 (95% CI 0.877 to 0.964). Mean char BPC is 1.0939 on N and 1.0797 on F; BPE scores 0.9486 on F and the 5-gram baseline 1.4341. Exploratory E2, ρ(char, 5-gram) on F, is 0.882 (95% CI 0.787 to 0.935). A clearly labelled post-hoc section of the notebook shows that the 2020 minutes are the most surprising year × genre group. §14 anticipated this pandemic-era effect on N; it does not change the H1 label. The confirmatory Run all took 747.7 s (12.5 min): over the 12-minute design target, under the 15-minute CPF limit.

## How to run

**Colab (the submission path).** Open the badge above and select *Runtime → Change runtime type → T4 GPU*, then *Run all*. The first cell clones this repository and checks the corpus SHA-256 values; the whole run takes about 13 minutes on a T4. The notebook stops without a CUDA GPU. Later runs are reproductions: fp16 GPU training is not bit-reproducible, so their values differ slightly from the confirmatory run, which each results section prints alongside.

**Local tests** (Python 3.11+, `torch`, `numpy`, `matplotlib`, `pytest`, `git`):

```bash
cd capstones/week03_transformers
python -m pytest tests
```

**Local smoke run.** Set `MODE = "smoke"` in the setup cell to run the whole notebook on CPU in under a minute, with tiny settings. The fake N/F split is cut from T meetings only, and its numbers are not results.

## Files

| Path | Contents |
|---|---|
| `week03_fomc_surprise.ipynb` | The paper: experiment, results, post-hoc exploration, limitations, references, AI-use statement. Saved without outputs. |
| `DESIGN.md` | Pre-registration (frozen; amendments made before the real run are in its change log) |
| `data.py` | SHA-256 checks, normalization, body parser, split loading, serialization, char vocabulary, window samplers |
| `bpe.py` | BPE tokenizer, fit on T bodies only |
| `model.py` | Attention, masks, multi-head attention, positional encoding, Pre-LN block, `TinyTransformerLM` |
| `train.py` | Frozen training recipe (AdamW, warm-up + cosine, fp16 autocast, fixed steps) |
| `evaluate.py` | Per-document sliding-window scoring and BPC, with the §4 assertions |
| `ngram.py` | Interpolated Witten–Bell character 5-gram baseline (E2) |
| `analysis.py` | H1–H3 and E1–E2: meeting bootstrap, labels |
| `make_split.py` | Builds `split_manifest.json` from the corpus (deterministic) |
| `split_manifest.json` | Committed meeting-level split: T/N/F/excluded, availability dates |
| `fomc_training_corpus.txt`, `fomc_training_corpus_manifest.json` | Frozen corpus snapshot and its provenance manifest |
| `fomc_statements_2015_2025.txt`, `fomc_statements_2015_2025_manifest.json` | Statements-only source corpus, input to the merge |
| `build_fomc_corpus.py`, `build_fomc_minutes_corpus.py`, `merge_fomc_corpus.py` | Corpus builders (statements, minutes, merge); each has `--self-test` |
| `tests/` | pytest suite for every module and for the notebooks |
| `pilot/` | Archival: the Phase 3 timing pilot (`phase3_timing.ipynb`, `.json`, tag `week03-v2-pilot3`) and the Phase 4 rehearsal notebook; neither needs re-running. |
| `runs/` | `confirmatory_results.json`, the executed confirmatory notebook `confirmatory_run.ipynb`, and its `figures/` |

## Data

FOMC post-meeting statements (2015–2026) and minutes (2010–2026) from federalreserve.gov: 210 documents (93 statements, 117 minutes), 6,731,426 characters. After the §2 boundary exclusion the split has 104 T, 32 N and 70 F documents; 4 are excluded because their meetings fall within 30 days of a split boundary. Minutes before 2010 use a legacy URL scheme that the builder does not crawl; the manifest records this gap.

## Pre-registration and reproducibility

- `DESIGN.md` fixed the split, normalization, scoring protocol, statistics, labels, compute settings and hyperparameters before any training on the real split. Amendments made before the real run (v0.4–v0.8) are logged with their reasons and state that no outcome had been observed.
- No early stopping or model selection: each transformer trains for a fixed step count, and the final weights are used.
- The corpus, its manifest and the split manifest are verified by SHA-256 at load time; a mismatch stops the notebook.
- The confirmatory run was executed once from a tag (`week03-v2-run1`) in a fresh clone; the notebook refuses a real run on uncloned or off-tag code. Its results are archived unchanged.
- Colab's preinstalled libraries are used without pins; versions are printed and saved with each run.

## Use of AI tools

- **Claude** (claude.ai chat, Anthropic): co-developed the research direction and the pre-registration (`DESIGN.md`) with the author, reviewed audit and review findings against the handout and the CPF rules, and drafted prompts.
- **Claude Code** (Anthropic, Claude Opus 5.5): wrote the Python modules, tests, notebooks and the prose under the author's direction, and ran the local tests and smoke runs.
- **Codex** (OpenAI, via the herdr terminal multiplexer): read-only adversarial reviews of the design and the code. Every finding was triaged and re-verified before any change.
- **The author** chose the direction and made all design decisions and amendments, approved or vetoed every change, ran all Colab executions (pilot, rehearsal, confirmatory run), and made every git commit.
- Earlier v1 commits in this folder carry `Co-Authored-By` trailers for Claude Sonnet 4.6.

The author is responsible for all content.

## References

- Y. Hilpisch, *Attention Mechanisms and Tiny Transformers*, The AI Engineer (TAE) handout. The attention, multi-head attention, positional encoding and block code in `model.py` follows its skeletons.
- yhilpisch/yoctoGPT, https://github.com/yhilpisch/yoctoGPT: reference implementation consulted for the model structure and the training loop.
- Vaswani, A., et al. (2017). Attention is all you need. *Advances in Neural Information Processing Systems 30*.
- Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine translation of rare words with subword units. *Proceedings of ACL 2016*, 1715–1725.
- Witten, I. H., & Bell, T. C. (1991). The zero-frequency problem: Estimating the probabilities of novel events in adaptive text compression. *IEEE Transactions on Information Theory*, 37(4), 1085–1094.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Board of Governors of the Federal Reserve System, FOMC statements and minutes, https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm.

Week 3 v1 (character vs BPE tiny transformer) is preserved at tag `week03-v1`.
