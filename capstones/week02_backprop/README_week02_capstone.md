<table width="100%">
<tr>

<td style="vertical-align: top;">

<h1>Week 2 Capstone — From Chain Rule to Backpropagation and <code>nn.Module</code></h1>


<p>
This folder packages the Week 2 backprop submission following the TAE program structure.<br>

</p>

</td>

<td align="right" width="200">
<img src="../../assets/tae_logo.png" alt="TAE Banner" width="160">
</td>

</tr>
</table>

## Primary submission notebook

The main Week 2 submission artifact is [week02_master_capstone.ipynb](week02_master_capstone.ipynb).

<a target="_blank" href="https://colab.research.google.com/github/FranQuant/the-ai-engineer/blob/main/capstones/week02_backprop/week02_master_capstone.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/>
</a>

This notebook consolidates the verified Week 2 evidence into one compact submission.


## Run locally

From the repo root, install the notebook dependencies from `requirements.txt` and a platform-appropriate PyTorch wheel, then open the notebook:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab capstones/week02_backprop/week02_master_capstone.ipynb
```

For a non-interactive check, run `jupyter nbconvert --execute --to notebook --inplace capstones/week02_backprop/week02_master_capstone.ipynb`.


## Evidence summary

In the verified master notebook:

- Finite-difference gradient check: max abs diff `9.79e-12`
- Torch forward parity: abs diff `1.09e-10`
- Torch manual backward parity vs NumPy: max abs diff `3.66e-08`
- Fixed-batch manual vs autograd agreement: max abs diff `0.0`
- `nn.Sequential` parity check: max abs diff `0.00e+00`
- Best validation loss: `0.072691`
- Best validation accuracy: `0.9300`
- Checkpoint smoke test: the best checkpoint reloaded and reproduced the saved validation metrics
- Diagnostics: 4-panel figure with train/val loss, validation accuracy, mean gradient norm, and mean hidden ReLU activity


## Developmental progression

The Week 2 solution builds from manual NumPy backprop through PyTorch forward parity, PyTorch autograd, and `nn.Module` training. The equations below capture the shared model structure used across that progression:

$$
a_1 = W_1 x + b_1,\qquad
h_1 = \mathrm{ReLU}(a_1),\qquad
f = W_2 h_1 + b_2
$$

Loss (per sample):

$$
L = \tfrac{1}{2}(f - y)^2
$$

PyTorch's built-in `MSELoss(reduction="mean")` computes the batch mean of
$(f - y)^2$. This changes gradient scaling, but not the underlying fixed points
of the optimization problem.

