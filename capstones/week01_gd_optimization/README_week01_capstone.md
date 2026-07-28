<img src="https://theaiengineer.dev/tae_logo_gw_flatter.png" width="35%" align="right">

# Week 1 Capstone — Gradient Descent Optimization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/the-ai-engineer/blob/main/capstones/week01_gd_optimization/gd_capstone.ipynb)

Implements and visualizes gradient descent (GD) and stochastic gradient
descent (SGD) from scratch on two one-dimensional objectives — a convex
quadratic baseline and a non-convex cubic with multiple basins of attraction.

## What's inside

| Section | Contents |
|---|---|
| Objectives | Quadratic baseline (convex, smooth) + cubic (non-convex, multi-basin) |
| GD | Deterministic gradient descent, step-size sweep |
| SGD | Constant and diminishing step-size schedules |
| Reproducibility | Single shared NumPy RNG (`np.random.default_rng(SEED)`) |
| Diagnostics | Final gap, best gap, steps-to-tolerance |

## Results

Eight figures (`assets/fig_01` through `fig_08`) covering the loss landscape,
GD trajectories from multiple initializations, the step-size sweep, SGD vs.
diminishing-SGD paths, and a step-geometry/schedule comparison.

## Run it

Colab badge above → Run All. Pure NumPy + Matplotlib, no GPU needed, runs in
under a minute. Locally: `jupyter lab gd_capstone.ipynb` → Run All.

## Deliverables

```text
week01_gd_optimization/
├── gd_capstone.ipynb              # full implementation, plots, diagnostics
├── assets/                        # 8 generated figures
└── README_week01_capstone.md
```

## Notes

- Dependencies: NumPy and Matplotlib only
- Deterministic given the shared RNG seed; a fresh RNG instance reproduces
  new trajectories
