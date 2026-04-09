<table width="100%">
<tr>
<td style="vertical-align: top;">

<h1>Week 1 Capstone — Gradient Descent Optimization</h1>

This folder contains the Week-1 capstone for *The AI Engineer* program.  
The goal is to implement and visualize basic gradient-based optimization  
methods on simple 1-D functions.

</td>

<td align="right" width="200">
<img src="../../assets/tae_logo.png" alt="TAE Banner" width="160">
</td>
</tr>
</table>


## Overview

This notebook studies gradient descent and stochastic gradient descent on the
piecewise non-smooth objective

$$f(x) = \left|\tfrac{1}{2}x^3 - \tfrac{3}{2}x^2\right| + \tfrac{1}{2}x$$

which has a kink (non-differentiability) at $x = 3$ and a global minimizer at
$x^\star = 1 - \tfrac{2\sqrt{3}}{3} \approx -0.155$.

A convex quadratic baseline $q(x) = \tfrac{1}{2}x^2$ is included as a clean
stability reference for step-size analysis. All runs are seeded for
reproducibility in Colab or locally.

---

## What’s Implemented

- Piecewise non-smooth objective `f(x) = |½x³ − (3/2)x²| + ½x`
- Analytic piecewise derivative with kink handling at `x = 3`
- Finite-difference gradient check (both smooth branches, max error < 1e-8)
- Local step geometry via a tangent-line gradient-step visualization
- Deterministic GD with convergence and divergence guards
- GD trajectories from 6 initializations revealing basin structure
  (basin boundary at x_max = 1 + (2√3)/3 ≈ 2.155)
- Step-size sweep η ∈ {0.01, 0.05, 0.10, 0.20, 0.50} with log-gap diagnostics
- SGD with zero-mean Gaussian gradient noise and fixed seeds
- Constant vs diminishing SGD schedules with objective-gap comparisons
- Compact metrics summary (`final_gap`, `best_gap`, `steps_to_tol`)
- Quadratic baseline `q(x) = ½x²` — stable (η < 2) vs divergent (η = 2.1)
- Global minimizer x★ identified analytically and verified numerically
- Colab-ready reproducibility with NumPy + Matplotlib only, no extra deps

---

## Outputs

- Main optimization figures for GD, SGD, schedule comparison, and the quadratic baseline
- In-notebook metrics summary for `final_gap`, `best_gap`, and `steps_to_tol`
- Seeded, reproducible runs for the sample paths and aggregated diagnostics

---

## File Structure

```text
week01_gd_optimization/
│
├── gd_capstone.ipynb          # Full implementation & generated plots
└── README_week01_capstone.md  # This document
```

The PNG figures are generated when the notebook runs and are not checked in here.

---

## How to Run

The notebook runs top-to-bottom without external data and uses only NumPy + Matplotlib.

It runs on:

- Google Colab  
- Local Jupyter Notebook  
- GitHub Codespaces  

### Open in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/FranQuant/the_ai_engineer_capstones/blob/main/capstones/week01_gd_optimization/gd_capstone.ipynb
)

Dependencies: **NumPy** and **Matplotlib** only
