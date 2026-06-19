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

The notebook runs top-to-bottom without external data and uses only **NumPy** + **Matplotlib**.

It runs on:

- Google Colab  
- Local Jupyter Notebook  
- GitHub Codespaces  

### Open in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/FranQuant/the-ai-engineer/blob/main/capstones/week01_gd_optimization/gd_capstone.ipynb
)

---

## Verification

- Finite-difference gradient check: the analytic derivative is compared against a centered finite-difference approximation across multiple test points on both sides of the kink at x=3 (avoiding the non-differentiable point itself), with max error < 1e-6 asserted at runtime

---

## Output Figures

| File | Description |
|------|-------------|
| `fig_01_landscape.png` | Objective and derivative across the kink at x=3 |
| `fig_02_gd_trajectories.png` | GD trajectories from convergent and kink-region starts |
| `fig_03_protocol_trajectories.png` | Protocol starts x0 ∈ {−1.0, 0.5, 2.0} with η=0.15 |
| `fig_04_step_sweep.png` | Step-size sweep: x_t and log gap versus iteration |
| `fig_05_step_geometry.png` | Single GD step geometry from x0=2.0 with η=0.2 |
| `fig_06_sgd_paths.png` | SGD sample paths and aggregated gap for constant η |
| `fig_07_sgd_schedule_comparison.png` | Constant vs diminishing SGD schedule across 20 seeds |
| `fig_08_quadratic_sweep.png` | Quadratic baseline showing stable and divergent step sizes |
