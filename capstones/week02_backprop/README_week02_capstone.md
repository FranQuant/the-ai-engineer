<table width="100%">
<tr>

<td style="vertical-align: top;">

<h1>Week 2 Capstone — From Chain Rule to Backpropagation and <code>nn.Module</code></h1>


<p>
This folder packages the Week 2 backprop submission following the TAE program structure.<br>
<code>week02_master_capstone.ipynb</code> is the primary submission artifact; the other notebooks are supporting progression notebooks that document the build-up from manual NumPy backprop to the PyTorch <code>nn.Module</code> training pipeline.
</p>

</td>

<td align="right" width="200">
<img src="../../assets/tae_logo.png" alt="TAE Banner" width="160">
</td>

</tr>
</table>

## Primary submission notebook

The main Week 2 submission artifact is [week02_master_capstone.ipynb](week02_master_capstone.ipynb).

<a target="_blank" href="https://colab.research.google.com/github/FranQuant/the_ai_engineer_capstones/blob/main/capstones/week02_backprop/week02_master_capstone.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/>
</a>

This notebook consolidates the verified Week 2 evidence into one compact submission.


## Evidence summary

In the verified master notebook:

- Finite-difference gradient check: max abs diff `9.79e-12`
- Torch forward parity: abs diff `1.09e-10`
- Torch manual backward parity vs NumPy: max abs diff `3.66e-08`
- Fixed-batch manual vs autograd agreement: max abs diff `0.00e+00`
- `nn.Sequential` parity check: max abs diff `0.00e+00`
- Best validation loss: `0.072691`
- Best validation accuracy: `0.9300`
- Checkpoint smoke test: the best checkpoint reloaded and reproduced the saved validation metrics
- Diagnostics: 4-panel figure with train/val loss, validation accuracy, mean gradient norm, and mean hidden ReLU activity


## Supporting / developmental progression notebooks

The original notebooks remain as the stepwise build-up behind the master submission. They use the same XOR-style synthetic data-generating process, the same `d -> h -> 1` MLP, and the same ReLU hidden-layer equations:

$$
a_1 = W_1 x + b_1,\qquad
h_1 = \mathrm{ReLU}(a_1),\qquad
f = W_2 h_1 + b_2
$$

Loss (per sample in Notebooks 01-03):

$$
L = \tfrac{1}{2}(f - y)^2
$$

In Notebook 04, PyTorch's built-in `MSELoss(reduction="mean")` is used, which
computes the batch mean of $(f - y)^2$. This changes gradient scaling, but not
the underlying fixed points of the optimization problem.


<table>
<tr>

<td width="50%" valign="top">

<h3>Notebook 01 — <code>01_numpy_manual.ipynb</code></h3>

<a target="_blank" href="https://colab.research.google.com/github/FranQuant/the_ai_engineer_capstones/blob/main/capstones/week02_backprop/01_numpy_manual.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/>
</a>

<b>Goal:</b> Manual forward + backward pass in NumPy.<br>
<b>Features:</b><br>
– Manual ReLU + derivative<br>
– Full chain-rule backprop<br>
– Finite-difference gradient checks (NumPy)<br>
– Source-of-truth implementation

</td>

<td width="50%" valign="top">

<h3>Notebook 02 — <code>02_pytorch_no_autograd.ipynb</code></h3>

<a target="_blank" href="https://colab.research.google.com/github/FranQuant/the_ai_engineer_capstones/blob/main/capstones/week02_backprop/02_pytorch_no_autograd.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/>
</a>

<b>Goal:</b> Reproduce NumPy forward pass in PyTorch without autograd.<br>
<b>Features:</b><br>
– <code>requires_grad = False</code><br>
– Forward consistency vs NumPy<br>
– Ensures algebraic and numerical alignment before enabling autograd<br>
– Same seeds + dataset

</td>

</tr>

<tr>

<td width="50%" valign="top">

<h3>Notebook 03 — <code>03_pytorch_autograd.ipynb</code></h3>

<a target="_blank" href="https://colab.research.google.com/github/FranQuant/the_ai_engineer_capstones/blob/main/capstones/week02_backprop/03_pytorch_autograd.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

<b>Goal:</b> Use PyTorch autograd and compare with manual gradients.<br>
<b>Features:</b><br>
– <code>loss.backward()</code> gradient flow<br>
– Manual vs autograd gradient match<br>
– Optional finite differences<br>
– Prepares for <code>nn.Module</code>

</td>

<td width="50%" valign="top">

<h3>Notebook 04 — <code>04_pytorch_nn_module.ipynb</code></h3>

<a target="_blank" href="https://colab.research.google.com/github/FranQuant/the_ai_engineer_capstones/blob/main/capstones/week02_backprop/04_pytorch_nn_module.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

<b>Goal:</b> Wrap the model in <code>nn.Module</code> and train with mini-batch SGD.<br>
<b>Features:</b><br>
– Custom <code>TwoLayerXOR</code> with <code>last_h1</code> ReLU activity tracking<br>
– 80/20 stratified train–validation split (<code>sklearn.model_selection</code>)<br>
– Separate <code>train_loader</code> / <code>val_loader</code> (<code>BATCH_SIZE=16</code>)<br>
– SGD training loop (200 epochs) with held-out validation tracking<br>
– Validation diagnostics and checkpoint round-trip<br>
– <code>nn.Sequential</code> weight-parity check (diff &lt; 1e-5)<br>
– Checkpoint save/load round-trip (<code>two_layer_xor.pt</code>, verified &lt; 1e-6)

</td>

</tr>
</table>


## Summary

The four supporting notebooks document the progression that leads to the
master submission: manual gradients, Torch forward parity, Torch autograd, and
`nn.Module` training.

They provide the step-by-step derivations and implementation details, while
`week02_master_capstone.ipynb` is the compact primary submission artifact.

---

## Dependencies (Minimal)

Minimal packages used in Week 02:

```text
numpy
matplotlib
torch
scikit-learn   # train_test_split used in Notebook 04
```

For the full environment used during development, see the root-level `requirements.txt`.
