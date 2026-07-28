<table width="100%">
<tr>

<td style="vertical-align: top;">

<h1>Week 2 Capstone — From Chain Rule to Backpropagation and <code>nn.Module</code></h1>


<p>
This folder contains the full Week-02 Capstone completed following the TAE Program structure.<br>
The goal is to implement a tiny <strong>1-hidden-layer MLP</strong>, step-by-step, moving from fully manual NumPy backprop to PyTorch’s <code>nn.Module</code> API.
</p>

</td>

<td align="right" width="200">
<img src="../../assets/tae_logo.png" alt="TAE Banner" width="160">
</td>

</tr>
</table>


All notebooks use:

- Deterministic seeds  
- Same XOR-style synthetic dataset  
- Same MLP architecture  

Forward pass:

$$
a_1 = W_1 x + b_1,\qquad
h_1 = \mathrm{ReLU}(a_1),\qquad
f = W_2 h_1 + b_2
$$

Loss:

$$
L = \frac{1}{2}(f - y)^2
$$


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
– Gradient checks<br>
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
– Ensures math alignment before autograd<br>
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
– Custom <code>TwoLayerXOR</code><br>
– <code>DataLoader</code> shuffling<br>
– SGD training loop (~200 epochs)<br>
– Loss + gradient-norm diagnostics

</td>

</tr>
</table>


## Summary

This 4-notebook progression builds the full intuition and engineering workflow:

1. Manual gradients  
2. Torch forward  
3. Torch autograd  
4. `nn.Module` + training loop  

It prepares the foundation for future Capstones involving:

- Deep networks  
- Optimizers  
- Regularization  
- Vision/sequence models  
- Reinforcement learning  
- Agentic training workflows  

---

## Dependencies (Minimal)

Minimal packages used in Week 02:

```text
numpy
matplotlib
torch
```

For the full environment used during development, see the root-level `requirements.txt`.

