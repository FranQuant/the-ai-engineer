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

This notebook explores the behavior of Gradient Descent (GD) and Stochastic Gradient Descent (SGD) on two instructional objectives:

1. **Quadratic baseline**
   • Convex and smooth  
   • Used to study stability and learning-rate effects  
   • Includes the required step-size sweep

2. **Cubic non-convex function**
   • Exhibits multiple basins of attraction  
   • Highlights divergence vs. convergence paths  
   • Demonstrates noisy vs. diminishing-noise SGD trajectories  

All experiments are one-dimensional, enabling direct visualization of optimization dynamics.

---

## What’s Implemented

• Deterministic Gradient Descent  
• Stochastic Gradient Descent with:
  – Constant step size  
  – Diminishing step size  

• **Single shared NumPy RNG for reproducibility**  
  (`rng = np.random.default_rng(SEED)`)  
  Fresh trajectories can be produced by supplying a new RNG instance.

• Convergence diagnostics:
  – Final gap  
  – Best gap  
  – Steps-to-tolerance  

• Visualizations:
  – GD step-size sweep (quadratic)  
  – Cubic function and derivative  
  – GD trajectories from multiple initializations  
  – SGD vs. diminishing-SGD trajectories

---

## File Structure

```text
week01_gd_optimization/
│
├── gd_capstone.ipynb          # Full implementation & plots
└── README_week01_capstone.md  # This document
```

---

## How to Run

The notebook runs top-to-bottom on:

- Google Colab  
- Local Jupyter Notebook  
- GitHub Codespaces  

### Open in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/FranQuant/the_ai_engineer_capstones/blob/main/capstones/week01_gd_optimization/gd_capstone.ipynb
)

Dependencies: **NumPy** and **Matplotlib** only