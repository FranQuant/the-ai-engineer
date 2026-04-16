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

---

## Verification

- Finite-difference gradient check (both smooth branches, max error < 1e-6, asserted at runtime)

---

## Output Figures

| File | Description |
|------|-------------|
| `fig_01_landscape.png` | f(x) and f′(x) on [−1, 4.5]; kink at x=3 and global minimizer marked |
| `fig_02_gd_trajectories.png` | GD trajectories (η=0.05, T=60) split by convergent vs kink-region initializations |
| `fig_02b_protocol_trajectories.png` | Protocol replication (handout Fig 4): x0 ∈ {−1.0, 0.5, 2.0}, η=0.15, overlaid on f(x) |
| `fig_03_step_sweep.png` | Step-size sweep η ∈ {0.05, 0.10, 0.15, 0.20}: x_t and log f-gap vs iteration |
| `fig_03b_step_sensitivity_bar.png` | Bar chart of final objective gap at K=200 for each step size |
| `fig_04_step_geometry.png` | Local geometry of one GD step from x0=2.0 with η=0.2; tangent line and update arrow |
| `fig_05_sgd_paths.png` | SGD sample paths (3 seeds) and aggregated gap with IQR band (20 seeds), constant η |
| `fig_06_sgd_schedule_comparison.png` | Constant vs diminishing schedule: median gap + IQR band across 20 seeds |
| `fig_07_quadratic_sweep.png` | Quadratic baseline: stable step sizes and divergence at η=2.1 |
