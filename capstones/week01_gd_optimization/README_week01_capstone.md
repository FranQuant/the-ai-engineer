<img src="https://theaiengineer.dev/tae_logo_gw_flatter.png" width="35%" align="right">

# Week 1 Capstone — Gradient Descent Optimization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/the-ai-engineer/blob/main/capstones/week01_gd_optimization/gd_capstone.ipynb)

Notebook: [gd_capstone.ipynb](gd_capstone.ipynb) · Repository: https://github.com/FranQuant/the-ai-engineer

GD and SGD from scratch on the nonconvex, nonsmooth objective $f(x)=\left|\tfrac12x^3-\tfrac32x^2\right|+\tfrac12x$, with a quadratic stability baseline: analytic minimizer, gradient check, step-size sweep, constant vs diminishing SGD schedules, and final gap, best gap and steps-to-tolerance for every run.

## Run

**Colab:** badge → Runtime → Run all (CPU, under 2 minutes).

**Locally**, from the repository root: `pip install -r requirements.txt`, then `jupyter lab capstones/week01_gd_optimization/gd_capstone.ipynb`.

Seeds are fixed; reruns reproduce every number and figure.
