<img src="https://theaiengineer.dev/tae_logo_gw_flatter.png" width="35%" align="right">

# Week 2 Capstone — From Chain Rule to Backpropagation and `nn.Module`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/the-ai-engineer/blob/main/capstones/week02_backprop/week02_master_capstone.ipynb)

Notebook: [week02_master_capstone.ipynb](week02_master_capstone.ipynb) · Repository: https://github.com/FranQuant/the-ai-engineer

A one-hidden-layer ReLU network (2 → 4 → 1) on synthetic XOR data, built four ways with a check at each step: manual NumPy backprop verified by finite differences; PyTorch tensors without autograd; manual batch gradients vs autograd; and `nn.Module`/`nn.Sequential` trained with SGD, with a validation split, best-on-validation checkpointing, and diagnostics (loss, accuracy, gradient norm, ReLU activity).

The manual derivation uses $L=\tfrac12(f-y)^2$; training uses mean-reduced MSE.

## Run

**Colab:** badge → Runtime → Run all (CPU, under a minute).

**Locally**, from the repository root: `pip install -r requirements.txt`, then `jupyter lab capstones/week02_backprop/week02_master_capstone.ipynb`.

Seeds are fixed; reruns reproduce every number. The checkpoint `week02_best_two_layer_xor.pt` is written at run time and not committed; `week02_diagnostics.png` is the saved diagnostics figure.
