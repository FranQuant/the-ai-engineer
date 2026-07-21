<table width="100%">
<tr>
<td style="vertical-align: top;">

<h1>Week 03 Capstone — Finance MiniGPT</h1>

<p><strong>Primary grading artifact:</strong> <code>week03_master_capstone.ipynb</code></p>

</td>
<td align="right" width="200">
<img src="../../assets/tae_logo.png" alt="TAE Banner" width="160">
</td>
</tr>
</table>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/the-ai-engineer/blob/capstone%2Fweek03-fomc/capstones/week03_transformers/week03_master_capstone.ipynb)

## Research question and scope

Can a small, transparent, character-level decoder-only Transformer built from first principles learn measurable language regularities from official FOMC policy-decision statements while preserving chronological validation integrity?

This is an educational language-model experiment. It is not a forecasting model, backtest, trading system, investment recommendation, or factual monetary-policy oracle.

## Primary submission

The unexecuted source notebook is the primary grading artifact. It contains:

- frozen-corpus and chronological-split verification;
- scaled dot-product attention and causal masking implemented from scratch;
- multi-head attention, sinusoidal positions, Pre-LN blocks, and the complete Finance MiniGPT;
- bounded numerical, gradient, batching, checkpoint, generation, and guard checks;
- guarded smoke and canonical-training interfaces;
- fail-closed verification and inline presentation of the adopted canonical results.

The notebook keeps stored outputs empty. Running it in default `validation-only` mode performs no training and renders the committed canonical results after verifying them.

## Frozen FOMC corpus

| Split | Documents | Dates | Body characters | Overlapping windows at context 128 |
|---|---:|---|---:|---:|
| Train | 82 | 2015-01-28 to 2024-12-18 | 237,840 | 227,344 |
| Validation | 8 | 2025-01-29 to 2025-12-10 | 17,474 | 16,450 |

- Corpus ID: `fomc-statements-2015-2025-v2`
- Corpus SHA-256: `c632b6a2e7bcfc0360e9fe18113a2ec1c9edce2d42f83b4ff96f5ce4d74ea125`
- Manifest SHA-256: `3fc37f4a8ea0cc358269e7f8e37bf310cc96da4ba3981d361038a30a4511c2f2`

The 2025 split is chronological but repeatedly evaluated for checkpoint selection, so it is validation—not a pristine untouched test set. Overlapping character windows are not independent observations.

## Adopted canonical run

- Run ID: `phase4-canonical-20260720T141501Z-532748ca`
- Source Git commit: `b0c3903135779ec7ba78d43f89b859454458a4da`
- Mode/status: `phase4-canonical` / complete canonical candidate adopted after review
- Device/precision: CUDA / FP32, AMP disabled
- Architecture: context 128, width 256, 8 heads, 4 layers, feed-forward width 1024, dropout 0.1
- Parameters: 3,178,240
- Optimization: AdamW, batch size 32, 1,221 steps, 5,001,216 tokens
- Best/final step: 1220
- Exhaustive 2025 validation CE: `1.8916519146487343`
- Character perplexity: `6.630312349647294`
- Reload maximum absolute logits difference: `0.0`
- Canonical reconciliation: 13/13 acceptance checks true

Cross-entropy and character perplexity improved substantially, and checkpoint reload reproduced the recorded probe exactly. The generated samples nevertheless remain predominantly malformed: greedy decoding collapses into repetition, while temperature/top-k decoding is more diverse but not reliably grammatical or semantically coherent. This is the principal educational finding—successful implementation and optimization do not by themselves produce sentence-level or policy-level competence.

## Promoted canonical artifacts

Only the reviewed inference-and-presentation set is committed under `results/canonical/`:

| File | Bytes | SHA-256 |
|---|---:|---|
| `finance_minigpt_best.pt` | 13,074,792 | `c2b7d44b655d667382e7f318bf9f7eeabbdf25215adad22bd5449ff8292dfb49` |
| `finance_minigpt_metrics.csv` | 135,567 | `3f93d83aae4718f2e96a206601494814e54b028728cfb58d3a73c1272a03346e` |
| `finance_minigpt_samples.json` | 64,372 | `9196aa6e39284207dc7ea1d5aaf7ea8742235a6a6f0f19bca2b789437a42f37a` |
| `finance_minigpt_training.png` | 89,714 | `e2635e0d6bce985d16e2504a2d023fc9d2f7ac2fa5e4e92e84dc939e7a7e59c3` |
| `finance_minigpt_attention.png` | 67,659 | `ee6c9d8c2c2a3740bacf79e985c096f97056c73f5f0f1d5d3077d22a5f56230d` |
| `finance_minigpt_run.json` | 6,368 | `b8de4e910222b1eb8bc9a5e4af82d4d10aec8320b43194fa811e85a347b300ae` |

The 38.5 MB final checkpoint is intentionally external because it carries optimizer state. The raw evidence ZIP is also external. Neither is required for default validation or presentation.

## Execution modes

### Default: validation-only

From the repository root, open the notebook and run all cells with no Week 3 mode variables set. The default:

1. reads and verifies the committed corpus and manifest;
2. runs all bounded from-scratch implementation checks;
3. performs no training;
4. verifies the exact canonical filename allowlist, sizes, hashes, schemas, run ID, Git/corpus identities, checkpoint policy, metrics, samples, PNGs, reload result, and acceptance checks;
5. renders the canonical tables, plots, all ten samples, scorecard, attention diagnostic, and limitations.

### Smoke diagnostic

Set `WEEK03_RUN_MODE=phase3b-smoke`. Smoke artifacts remain explicitly non-canonical and isolated beneath `/tmp`; they cannot overwrite `results/canonical/`.

### Guarded canonical retraining

Canonical retraining is optional and is not needed to grade or inspect the adopted run. In a clean CUDA Colab clone, set:

```bash
WEEK03_REPO_ROOT=/content/the-ai-engineer
WEEK03_RUN_MODE=phase4-canonical
WEEK03_CONFIRM_CANONICAL=YES
WEEK03_CANONICAL_OUTPUT_DIR=/content/week03_finance_minigpt_canonical_candidate
```

The output directory must not already exist, must be beneath `/content`, and must be outside the repository. A new run remains an external candidate and is never auto-promoted.

## Colab instructions

1. Open the branch-specific [Colab notebook](https://colab.research.google.com/github/FranQuant/the-ai-engineer/blob/capstone%2Fweek03-fomc/capstones/week03_transformers/week03_master_capstone.ipynb).
2. Use a standard runtime for validation-only; select a GPU runtime only for an explicitly authorized canonical candidate.
3. Ensure the repository is available at `/content/the-ai-engineer` and set `WEEK03_REPO_ROOT` if Colab does not start inside it.
4. Choose **Runtime → Run all**. With no mode override, training is disabled.

## Interpretation and limitations

- CE and character perplexity measure next-character prediction, not syntax, factuality, policy understanding, forecasting, or economic value.
- Sample scorecard metrics are distributional and lexical proxies; they do not establish semantic quality.
- Generated text has high hallucination risk and must not be represented as Federal Reserve content.
- The attention plot shows the masked probability pattern for one selected head and prompt. Attention probabilities are not explanations or causal importance.
- Stochastic samples can differ across PyTorch/device backends despite correct checkpoint reconstruction.

## Educational reference

`week03_yocto_comparison.ipynb` remains an optional historical learning appendix about implementation and training-loop choices. It uses the earlier arXiv experiment and is not the primary result, canonical evidence, or evaluation basis for this FOMC capstone.
