# Week 3 Capstone v2 — Design (pre-registration)

Status: FROZEN v0.8 (Phase 4 amendment from Codex review, logged below). Revised after Codex review rounds 1–2. Freeze before any model is trained on the real split. Changes after freeze go in the change log.

## 1. Research question

Trained only on FOMC text available before a cutoff, how surprising is each later FOMC document, measured as per-document bits per character (BPC)? Is that surprise measure stable across tokenizers?

The tiny decoder-only transformer (handout spec, built from scratch) is the measurement instrument. The contribution is the measurement and its robustness, not the architecture.

## 2. Data and split

- Source: existing FOMC corpus (statements + minutes), static snapshot. At load time, SHA-256 of corpus and manifest are checked against values hardcoded in the notebook; mismatch stops the notebook.
- **Unit of assignment is the meeting.** A meeting's statement and minutes always go to the same split.
- **Availability rule:** a meeting is assigned by the date its last document became public. Minutes are released 21 days after the meeting (use the manifest's release date if present; Phase 1 checks this). So a meeting counts as "available" at meeting date + 21 days.
- **Boundary exclusion:** any meeting whose availability date falls within 30 days of a split boundary (2019-12-31 or 2021-12-31) is excluded from all splits and counted. This makes the assignment robust to the exact release date.
- Clarifications (v0.4): "within 30 days" means |availability − boundary| ≤ 30 days; a meeting with no minutes (e.g. unscheduled 2020 meetings) has availability = meeting date; per-year tables group by meeting year. Unmatched meetings in N are allowed (H2 uses F only).
- Splits by availability date:

| Split | Availability date | Role |
|---|---|---|
| Train (T) | ≤ 2019-12-31 | fit tokenizers, n-gram, transformers |
| Near (N) | 2020-01-01 → 2021-12-31 | H1 reference; plotted during training, **no decision depends on it** |
| Far (F) | ≥ 2022-01-01 | main evaluation |

- No early stopping and no model selection: each transformer trains for a fixed number of steps; the final weights are used. So no split is used for selection.
- Split is written to a committed static manifest (meeting_id → split, document_ids, genre, availability date).
- **Feasibility thresholds (exact):** after boundary exclusion, F must contain ≥ 40 documents and ≥ 15 matched statement/minutes pairs; N ≥ 16 documents. If any fails, the study is declared infeasible (§10).

## 3. Text normalization and document format

- Normalization, applied identically to every document before anything else, in this order: (1) Unicode NFD; (2) remove combining marks (category Mn); (3) remove format characters (category Cf, e.g. zero-width joiner); (4) fixed map: ø→o, Ø→O, and delete ®, ™, ©; (5) NFC. No other edits.
- After normalization, every character in N and F must appear in T. If not, the notebook stops and lists the offending characters (fail closed). Phase 1 checks this on the snapshot.
- **Body** = the document text after its metadata header (header lines: `date:`, `document_id:`, `meeting_type:`, special tokens). The parser is one function, used for training and scoring alike.
- Each document is serialized as: `<BOS>` + genre token (`<stmt>` or `<min>`) + body. No date, ID or meeting type anywhere in model input. Estimand: body surprise conditional on genre.

## 4. Scoring protocol

- Per-document BPC = Σ (unsmoothed next-token NLL over the tokens that encode the body) / (number of body code points × ln 2).
- Each document is scored independently, starting from `<BOS>` + genre; context never crosses documents.
- Deterministic sliding windows, stride = block_size / 2; every body target scored exactly once.
- Assertions per document: (a) decode(encode(body)) == body; (b) number of scored targets == number of body tokens; (c) the genre and BOS tokens are never scored.
- BPE body tokens are encoded from the body alone, so token boundaries never cross the genre token.

## 5. Instruments

| Instrument | Tokenizer | Role |
|---|---|---|
| Char transformer | characters of normalized T text + BOS + 2 genre tokens | **primary** |
| BPE transformer | BPE fit on T bodies only; vocab size fixed in Phase 3 | second instrument (H3) |
| n-gram baseline | characters | exploratory (E2) |

- Transformers: same depth, width, block_size (in tokens). Plain cross-entropy (no label smoothing), AdamW, cosine LR with warmup, gradient clipping, fixed step count. No resume phase.
- n-gram: interpolated Witten–Bell, order 5, fixed a priori (no tuned parameters). Order-0 fallback is uniform over the T character vocabulary, so no probability is zero.
- Known confounds, stated not controlled: BPE sees more raw characters per block and has more vocabulary-dependent parameters. Both are reported.

## 6. Hypotheses (confirmatory)

All use the char transformer unless stated. Uncertainty: percentile bootstrap, 2,000 resamples of **meetings**, seed fixed. Result labels: **supported** if the 95% CI excludes 0 in the predicted direction; **contradicted** if the point estimate is ≤ 0; **inconclusive** otherwise. Bootstrap seed 20260925; a resample with an undefined ρ (constant input) is excluded from the CI and counted; a point estimate ≤ 0 is "contradicted" regardless of the CI.

- **H1 (drift):** Δ₁ = mean BPC(F) − mean BPC(N) > 0. Surprise grows with distance from the cutoff.
- **H2 (genre):** Δ₂ = mean over matched F meetings of [BPC(minutes) − BPC(statement)] > 0. Statements are more formulaic. Unmatched meetings are excluded and counted.
- **H3 (instrument robustness):** Spearman ρ between char- and BPE-transformer per-document BPC on F ≥ 0.7. Supported if ρ ≥ 0.7, contradicted if ρ < 0.7 (CI reported, not used for the label). The two instruments differ in tokenizer, raw-character context span and vocabulary-dependent parameters together (§5); H3 tests whether the surprise ranking survives that change of instrument, not the causal effect of tokenization alone.

A contradicted or inconclusive hypothesis is reported as such, never tuned away.

## 7. Exploratory analyses (no pass/fail)

- **E1:** the 5 highest-BPC F documents, listed with dates. Discussed in context; no causal or market claim.
- **E2:** Spearman ρ between char transformer and n-gram per-document BPC on F. Interpretation only: high ρ means the transformer adds little ranking information beyond local statistics. The n-gram has no BOS or genre context (context resets at the document start).

## 8. Compute budget and pilot

- Whole notebook ≤ 12 min on a Colab T4 (CPF limit 15), measured by a cold "Run all", including data fetch, checks, both transformers, n-gram, scoring and figures.
- **Phase 3 timing pilot:** trains on T data only, scores nothing on N or F, and its models are discarded. It may change only compute settings: d_model, layers, heads, d_ff, block_size, batch size, step count, BPE vocab size, numeric precision (fp16 autocast or fp32). Once set, these are logged here and frozen before the real run.
- **Selection rule (pre-registered):**
  - Budget 720 s.
  - Subtract, all measured on T only: overheads (imports and CUDA init, clone, load and hash checks, checks, tokenization, figures), BPE fit, n-gram fit, char scoring projected to N+F, BPE and n-gram scoring projected to F, bootstrap, and N monitoring.
  - Split the remainder equally between the char and BPE seed-1 runs.
  - Candidates in the order C1 (256/6/8/1024), C2 (192/4/6/768), C3 (128/4/4/512) (d_model/layers/heads/d_ff), fp16 before fp32 within each. The first where both runs get ≥ 1,500 steps wins, with steps = floor(per-run budget / median s/step), the minimum of the two.
  - BPE vocab = the largest of {4000, 2000, 1000} with shortest T document ≥ block_size + 1 tokens and fit ≤ 90 s.
  - Seed 2 (char only) is added only if one more char run at the same config and steps, plus its monitoring and F scoring, fits in the budget left after the selection. It never changes config or steps.
  - If nothing qualifies: NO CONFIG FITS, and the study is infeasible under §10.
- If budget allows (rule above), a second seed for the char transformer; its ρ with seed 1 on F is reported as model uncertainty. Otherwise all claims are limited to seed 1.
- **Frozen compute settings (Phase 3 pilot, Colab Tesla T4, torch 2.11.0+cu128, commit 1f0cd9d, `pilot/phase3_timing.json`):**
  - C1: d_model 256, 6 layers, 8 heads, d_ff 1024.
  - fp16 autocast with GradScaler for training; fp32 scoring (`evaluate.py`).
  - block_size 256, batch 64, 3,696 steps for each of the char and BPE seed-1 runs.
  - BPE vocab 2000 (vocab 4000 fit 95.4 s > 90 s).
  - No second char seed (needs 268.0 s, 24.1 s left).
  - Projected total 695.9 s.
- **Frozen training hyperparameters:** AdamW, lr 3e-4 with 200 linear warm-up steps, then cosine decay to 3e-5 at the final step; weight decay 0.1 (both instruments); dropout 0.1; gradient-norm clip 1.0; seed 1; token embedding (tied with the output layer) initialized N(0, d_model^-1/2) (std 0.0625 at C1); all other layers PyTorch defaults; weight decay 0.1 applied to all parameters. N monitoring at 10 evenly spaced evaluations, each on 20 batches of 64 windows, plotted only.
- A step is one optimizer iteration; fp16 GradScaler-skipped updates are counted and reported, not replaced.
- **Runtime contingency:** if the Phase 6 cold T4 Run all exceeds 15 min, multiply both runs' step count by the same factor so the projected total is ≤ 13.5 min. Nothing else changes. The new step count is logged here.

## 9. Reproducibility and Colab

- First code cell clones the repository at the fixed tag `week03-v2` (depth 1) when the package is not present, then verifies the corpus and manifest SHA-256 and stops on mismatch. The tag is created on the submitted commit.
- The confirmatory result is the first real run, executed from tag `week03-v2-run1`; its results.json is committed as `runs/confirmatory_results.json`. Later runs (Phase 6, graders) are reproductions; fp16 GPU training is not bit-reproducible, and differences from the confirmatory run are reported, not substituted.
- A rehearsal (full frozen settings on a fake split built from T meetings only, never touching N/F) may run on the T4 to measure runtime.
- Model code lives in `model.py`, `bpe.py`, `data.py`, `evaluate.py`; the notebook imports them after the clone.
- Library versions (Python, torch, CUDA, GPU name) are printed. Colab's preinstalled torch is used without pins: pinning would add install time and risk CUDA mismatches. Accepted limitation (§14).
- Notebook metadata requests a GPU (T4). The first code cell **stops** if CUDA is unavailable. Any CUDA GPU is accepted; the budget in §8 is set on a T4, the slowest standard Colab GPU.
- Every number in prose is printed from variables.

## 10. Infeasibility rule (fixed now)

If the §2 thresholds fail after boundary exclusion, or the §3 character check fails after normalization, this study is declared infeasible and stops before any model is trained on the real split. No substitute hypotheses are tested under this pre-registration; any alternative study needs its own design document.

## 11. Handout coverage (kept, condensed)

From-scratch: scaled dot-product attention (boolean and additive float masks), causal mask, self-attention, multi-head attention, FFN, Pre-LN block, sinusoidal positional encoding, TinyTransformerLM, greedy + temperature sampling (one short demo).
Checks: §4.3 worked example (print + assert QKᵀ, S, A, Y; causal rerun), fused vs manual SDPA parity, MHA = single-head at H = 1, uniform logits ≈ log V, trivial-pattern overfit.

## 12. Dropped from v1

Resume phases, label smoothing, early stopping, Char-KL, attention-map figure, 3-prompt sampling gallery, contiguous 90/10 split, date/ID headers in model input.

## 13. Notebook outline

1. Title, author, date, abstract
2. Introduction and hypotheses
3. Data, normalization, split
4. Model (imported) and verification checks
5. Training
6. Scoring protocol
7. Results: H1, H2, H3; exploratory E1, E2
8. Limitations and future work
9. References and reused code
10. Use of AI tools
11. Runtime

## 14. Known limitations (stated up front)

Textual surprise is not market surprise. One cutoff, not a rolling backtest. N includes 2020 pandemic-era language, which may raise N and make H1 harder to support. Small models; absolute BPC is not comparable to published LMs. Tokenizer confounds in §5. Seed limits in §8. Unpinned Colab libraries (§9): results are reproducible to the printed library versions, not bit-identical across Colab updates. BPE training sees about 43 passes over its T tokens and char about 16; with fixed steps and no early stopping, BPE may overfit, visible in the N curve. H3 compares rankings, not absolute BPC.

## Change log

- v0.2 (Phase 0, Codex round 1): meeting-level split with availability rule (#1); removed validation-based selection, fixed step count, N/F split (#2); former H2 made exploratory E1 (#3); n-gram agreement made exploratory E2, char–BPE ρ is confirmatory H3 (#4); primary instrument, statistics, bootstrap and labels defined (#5); n-gram fixed as Witten–Bell order 5 (#6); normalization + fail-closed replaces `<unk>` (#7); runtime gate kept in §8 (#8); date/ID removed from input, BOS + genre only (#9); H2 paired within meeting (#10); tokenizer confounds stated (#11); body and scoring assertions defined (#12); pilot and exact thresholds and fallback defined (#13); Colab bootstrap with hash check (#14); seed handling (#15).
- v0.3 (Phase 0, Codex round 2, frozen): 30-day boundary exclusion (#1); H3 reworded as instrument robustness with confounds explicit (#11); fallback replaced by an infeasibility rule (#13, new #16); clone pinned to tag `week03-v2`, unpinned libraries stated as limitation (#14); CPU now stops the notebook (new #17). #8 stays open by design until the Phase 3 T4 measurement.
- v0.4 (Phase 1 amendment, 2026-09-24, before any model training; no BPC or other outcome observed): Phase 1 found 3 N/F characters absent from T after v0.3 normalization: ø (21 in F, 2 in N; one staff name, "Vissing-Jørgensen"), U+200D zero-width joiner (2 in F, 1 in N), ® (1 in N, "Fedwire®"). NFD does not decompose ø, and format/symbol characters carry no language content. §3 normalization extended with steps (3)–(4). Fail-closed check unchanged. Also ratified three Phase 1 interpretations of §2 (see clarifications). Split counts at amendment time: T 104 docs, N 32, F 70 (35 matched pairs), 2 boundary meetings excluded. Observed effect of step (3) on the full corpus: removes 221 soft hyphens (U+00AD), 3 zero-width joiners and 1 left-to-right mark; T character vocabulary 85 → 83.
- v0.5 (Phase 3 amendment, before any T4 measurement; only CPU smoke-run numbers (not used) had been seen): Precision added as a pilot setting; selection rule and seed-2 rule moved from the pilot notebook into §8 (Codex Phase 3 #2, #4).
- v0.6 (Phase 3 closed): compute settings chosen by the §8 rule on a Colab T4 (`pilot/phase3_timing.json`); training hyperparameters and the runtime contingency frozen before any training on the real split.
- v0.7 (Phase 4 amendment, before any training on the real split and before any N/F scoring): initial loss at C1 was ~186 nats/token vs ln V ≈ 4.45 because the tied embedding was initialized N(0, 1); token embedding init changed from N(0, 1) to N(0, d_model^-1/2) (0.0625 at C1). Chosen among N(0, 0.02), N(0, d^-1/2) and d^-1/2 init with √d input scaling, using a 300-step training-loss trial on T only (no N/F data): N(0, 0.02) left the model at the T unigram entropy after 300 steps because the token signal was small next to the sinusoidal positions; N(0, d^-1/2) keeps the handout's forward pass unchanged. Weight-decay scope, bootstrap seed and label edge cases made explicit. Phase 3 timings unaffected (init does not change step cost).
- v0.8 (Phase 4 amendment from Codex review, before any real run): skipped-update accounting, confirmatory-run definition and tag pinning, rehearsal mode.

## Run log

- Confirmatory run: tag `week03-v2-run1` (d3abf6c), Colab Tesla T4, 2026-09-25, 747.7 s, results in `runs/confirmatory_results.json`.
- Submission tag: `week03-v2`.
