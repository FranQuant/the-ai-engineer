<table width="100%">
<tr><td><h1>Week 03 Capstone — Finance MiniGPT</h1><p><strong>Primary artifact:</strong> <code>week03_master_capstone.ipynb</code></p></td><td align="right" width="200"><img src="../../assets/tae_logo.png" alt="TAE Banner" width="160"></td></tr>
</table>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/the-ai-engineer/blob/week03-capstone-v1/capstones/week03_transformers/week03_master_capstone.ipynb)

## Objective and responsible-use boundary

Build a small, transparent, character-level decoder-only Transformer from first principles and test whether it learns measurable language regularities from official FOMC policy-decision statements under a chronological validation design.

This is an educational language-model experiment—not a forecasting model, backtest, trading system, investment recommendation, Federal Reserve simulator, or factual monetary-policy oracle. Generated text is synthetic, frequently malformed, and must not be represented as Federal Reserve communication or used for decisions.

## Learning outcomes

Readers can follow how to:

- implement scaled causal attention and multi-head self-attention from tensor operations;
- assemble sinusoidal positions, Pre-LN blocks, and a decoder-only Transformer;
- keep document windows inside a frozen chronological split;
- trace one AdamW optimization step from batch construction through gradient clipping and scheduling;
- distinguish numerical optimization from semantic generation quality;
- reconcile immutable artifacts and reconstruct a checkpoint without retraining.

The unexecuted notebook is the primary source artifact. Stored outputs remain empty under the repository policy. Default `validation-only` execution performs no training and renders the verified canonical results.

## Data provenance

The frozen corpus contains 90 normalized public FOMC policy-decision statements sourced from Federal Reserve pages. Source URLs, dates, extraction metadata, failures, and per-document hashes are recorded in the committed manifest.

| Split | Documents | Dates | Body characters | Overlapping windows at context 128 |
|---|---:|---|---:|---:|
| Train | 82 | 2015-01-28 to 2024-12-18 | 237,840 | 227,344 |
| Validation | 8 | 2025-01-29 to 2025-12-10 | 17,474 | 16,450 |

- Corpus ID: `fomc-statements-2015-2025-v2`
- Corpus SHA-256: `c632b6a2e7bcfc0360e9fe18113a2ec1c9edce2d42f83b4ff96f5ce4d74ea125`
- Manifest SHA-256: `3fc37f4a8ea0cc358269e7f8e37bf310cc96da4ba3981d361038a30a4511c2f2`

The 2025 documents are repeatedly evaluated for checkpoint selection, so this is validation—not a pristine test set. Overlapping character windows are not independent observations. Additional training or tuning against this frozen validation period is not part of the release.

## Architecture

The sorted training-only character vocabulary feeds a context-128 decoder with width 256, eight heads, four Pre-LN blocks, 1,024-unit GELU feed-forward layers, sinusoidal positions, dropout 0.1, causal masking, and tied input/output weights. The essential implementation stays visible; it does not substitute a high-level attention API.

## Canonical run and honest result

- Run: `phase4-canonical-20260720T141501Z-532748ca`
- Device/precision: CUDA / FP32; AMP disabled
- Parameters: 3,178,240
- Optimization: AdamW, batch size 32, 1,221 steps, 5,001,216 tokens
- Best/final step: 1220
- Exhaustive 2025 validation CE: `1.8916519146487343`
- Character perplexity: `6.630312349647294`
- Reload maximum absolute logits difference: `0.0`
- Canonical reconciliation: 13/13 acceptance checks true

Validation CE improved from 4.3814 to 1.8917. Greedy samples nevertheless collapse into repetition, while temperature/top-k samples are more diverse but remain predominantly malformed and semantically incoherent. All ten generations are shown without cherry-picking.

- Transformer implementation: successful
- numerical optimization: successful
- reproducibility and artifact reconciliation: successful
- coherent semantic FOMC generation: unsuccessful

The attention image verifies causal masking for one selected head and prompt. It is not an explanation of model reasoning, feature importance, policy reasoning, or semantic understanding.

## Canonical artifact inventory

Only the reviewed inference-and-presentation set is committed under `results/canonical/`:

| File | Bytes | SHA-256 |
|---|---:|---|
| `finance_minigpt_best.pt` | 13,074,792 | `c2b7d44b655d667382e7f318bf9f7eeabbdf25215adad22bd5449ff8292dfb49` |
| `finance_minigpt_metrics.csv` | 135,567 | `3f93d83aae4718f2e96a206601494814e54b028728cfb58d3a73c1272a03346e` |
| `finance_minigpt_samples.json` | 64,372 | `9196aa6e39284207dc7ea1d5aaf7ea8742235a6a6f0f19bca2b789437a42f37a` |
| `finance_minigpt_training.png` | 89,714 | `e2635e0d6bce985d16e2504a2d023fc9d2f7ac2fa5e4e92e84dc939e7a7e59c3` |
| `finance_minigpt_attention.png` | 67,659 | `ee6c9d8c2c2a3740bacf79e985c096f97056c73f5f0f1d5d3077d22a5f56230d` |
| `finance_minigpt_run.json` | 6,368 | `b8de4e910222b1eb8bc9a5e4af82d4d10aec8320b43194fa811e85a347b300ae` |

The optimizer-bearing final checkpoint and raw evidence archive are intentionally external and are not needed for validation or presentation.

## Default validation-only path

From a local checkout, open the notebook and run all cells without setting Week 3 mode variables. The opening runtime summary reports Python, PyTorch, device, Colab status, repository root, and run mode. The notebook then verifies the corpus and manifest, runs 48 bounded implementation checks, verifies all six canonical artifacts and 13 canonical acceptance checks, reloads the checkpoint, and renders the trajectory, all ten samples, scorecard, training plot, and masking diagnostic. Training remains disabled.

## Automatic Colab behavior

1. Open the [`week03-capstone-v1` notebook in Colab](https://colab.research.google.com/github/FranQuant/the-ai-engineer/blob/week03-capstone-v1/capstones/week03_transformers/week03_master_capstone.ipynb).
2. Choose **Runtime → Run all** on a standard CPU runtime.
3. The bootstrap detects Colab, reuses a valid checkout or clones the public repository at immutable revision `week03-capstone-v1`, sets `WEEK03_REPO_ROOT`, and installs only a genuinely missing dependency.
4. Validation-only mode discovers the frozen corpus and six canonical artifacts, validates them, and displays results.

No ZIP upload, manual clone, credential, or private filesystem path is required. `WEEK03_REVISION` is only a temporary override for private/local validation before the public tag exists; the released default stays pinned to `week03-capstone-v1`.

## Limitations

- Character CE and perplexity do not measure syntax, factuality, policy understanding, forecasting, or economic value.
- Scorecard metrics are distributional and lexical proxies, not semantic evaluation.
- The frozen 2025 period is validation already used for model selection.
- Generated text has high hallucination risk.
- Seeded stochastic output may vary across PyTorch/device backends.
