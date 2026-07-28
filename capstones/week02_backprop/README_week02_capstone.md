<img src="https://theaiengineer.dev/tae_logo_gw_flatter.png" width="35%" align="right">

# Week 2 Capstone — From Chain Rule to Backpropagation and `nn.Module`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/the-ai-engineer/blob/main/capstones/week02_backprop/week02_master_capstone.ipynb)

Implements a tiny 1-hidden-layer MLP on an XOR-style synthetic dataset,
progressing from fully manual NumPy backpropagation to PyTorch's `nn.Module`
API — all four stages consolidated into one notebook with a shared seed and
dataset throughout, so every implementation is checked against the same
ground truth.

## What's inside

| Section | Contents |
|---|---|
| Manual gradients | Forward pass, ReLU + derivative, full chain-rule backprop by hand |
| Verification | Finite-difference gradient checks against the manual implementation |
| PyTorch, no autograd | Same forward pass reproduced with `requires_grad=False`, confirms math alignment |
| PyTorch, autograd | `loss.backward()`, manual-vs-autograd gradient match |
| `nn.Module` | Custom `TwoLayerXOR`, `DataLoader`, SGD training loop (~200 epochs) |

Forward pass: $a_1 = W_1 x + b_1,\; h_1 = \mathrm{ReLU}(a_1),\; f = W_2 h_1 + b_2$

Loss: $L = \tfrac{1}{2}(f - y)^2$

## Results

Manual and autograd gradients agree to numerical precision; the `nn.Module`
training loop converges on the XOR-style dataset with loss and gradient-norm
diagnostics tracked throughout training.

## Run it

Colab badge above → Run All. Locally:
`jupyter lab week02_master_capstone.ipynb` → Run All.

## Deliverables

```text
week02_backprop/
├── week02_master_capstone.ipynb   # manual → no-autograd → autograd → nn.Module, one notebook
└── README_week02_capstone.md
```

## Notes

- Dependencies: NumPy, Matplotlib, PyTorch
- Checkpoint not committed (regenerated on each run)
- Foundation for later capstones: deeper networks, optimizers,
  regularization, and sequence/vision models
