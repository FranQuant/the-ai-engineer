<table width="100%">
<tr>
<td style="vertical-align: top;">

<h1>Week 03 Capstone — Mini GPT Transformer</h1>

<p>
<strong>Primary submission notebook:</strong> <code>week03_master_capstone.ipynb</code><br>
<strong>Supporting notebook:</strong> <code>mini_gpt_diagnostics.ipynb</code>
</p>

<p>
This folder contains a from-scratch decoder-only Transformer capstone with a clear submission boundary:
the master notebook is the graded artifact, and the diagnostics notebook is secondary analysis only.
</p>

</td>
<td align="right" width="200">
<img src="../../assets/tae_logo.png" alt="TAE Banner" width="160">
</td>
</tr>
</table>

[![Open Primary In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/FranQuant/the_ai_engineer_capstones/blob/main/capstones/week03_transformers/week03_master_capstone.ipynb
)
[![Open Diagnostics In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/FranQuant/the_ai_engineer_capstones/blob/main/capstones/week03_transformers/mini_gpt_diagnostics.ipynb
)

## Submission Boundary

- `week03_master_capstone.ipynb` is the primary Week 03 submission notebook.
- `mini_gpt_diagnostics.ipynb` is supporting analysis only.
- The modular Python files remain part of the implementation and are reused by the notebook.
- `mini_gpt.pt` and `run_record.json` are the saved training artifacts.

## What The Master Notebook Covers

- Tiny character corpus and tokenizer
- Scaled dot-product attention with causal masking
- Multi-head self-attention
- Pre-layernorm transformer block
- Tiny decoder-only language model
- Train/val split and real optimization loop
- Checkpoint save and run record save
- Greedy and temperature-based sampling

## Folder Contents

```text
week03_transformers/
├── week03_master_capstone.ipynb       # Primary submission notebook
├── mini_gpt_diagnostics.ipynb         # Supporting analysis notebook
├── mini_gpt.pt                        # Saved model checkpoint
├── mini_transformer.py
├── multihead_attention.py
├── README_week03_capstone.md
├── run_record.json                    # Saved training metadata
├── scaled_dot_product_attention.py
├── train_mini_gpt.py
└── transformer_block.py
```

## How To Run

1. Open `week03_master_capstone.ipynb` in Colab or run it locally from the repository checkout.
2. Execute the notebook top-to-bottom. It will train the model, save `mini_gpt.pt`, and write `run_record.json`.
3. Use `mini_gpt_diagnostics.ipynb` only for secondary inspection and analysis.

## Notes

- The notebook uses the same tiny corpus as `train_mini_gpt.py`.
- The notebook is self-contained and does not depend on hidden local state.
- If you want to regenerate the checkpoint outside the notebook, `train_mini_gpt.py` mirrors the same core architecture and artifact names.
