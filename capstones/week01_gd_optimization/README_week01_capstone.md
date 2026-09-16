<img src="https://theaiengineer.dev/tae_logo_gw_flatter.png" width="35%" align="right">

# Week 1 Capstone — Gradient Descent Optimization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/the-ai-engineer/blob/main/capstones/week01_gd_optimization/gd_capstone.ipynb)

Implements and visualizes gradient descent (GD) and stochastic gradient
descent (SGD) from scratch on two one-dimensional objectives — a convex
quadratic baseline and a nonconvex, nonsmooth piecewise-cubic objective.

## What's inside

| Section | Contents |
|---|---|
| Objectives | Quadratic baseline (convex, smooth) + nonconvex, nonsmooth piecewise-cubic objective |
| GD | Deterministic gradient descent, step-size sweep |
| SGD | Constant and diminishing step-size schedules |
| Reproducibility | Explicit per-run NumPy generators with recorded seeds; matching seeds pair noise across configurations. |
| Diagnostics | Per-seed SGD summaries, success/non-hit counts, conditional first-hit statistics, and mean/SE comparisons with linearized theory |

## Results

Figures and diagnostic tables are displayed inline in `gd_capstone.ipynb`.
Run all cells and save the notebook to retain the outputs.

## Run it

Colab badge above → Run All. NumPy + Matplotlib computations, no GPU needed,
with a target runtime under two minutes with dependencies installed.
Fresh hosted Colab verification of the revised notebook is pending.

Locally, from the repository root:

```bash
cd capstones/week01_gd_optimization
jupyter lab gd_capstone.ipynb
```

## Deliverables

```text
week01_gd_optimization/
├── gd_capstone.ipynb              # full implementation, plots, diagnostics
└── README_week01_capstone.md
```

## Notes

- Numerical and plotting dependencies: NumPy and Matplotlib.
  HTML tables use IPython.display; local execution requires Jupyter.
- A fresh generator initialized with the same seed reproduces the same noise sequence. Different seeds produce different realizations.
