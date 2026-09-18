<img src="https://theaiengineer.dev/tae_logo_gw_flatter.png" width="35%" align="right">

# The AI Engineer

## Four Capstones · From Optimization to Agentic Systems

A four-capstone portfolio progressing from optimization and backpropagation through transformer training to an auditable, MCP-based agentic incident-command system.

[![The AI Engineer](https://img.shields.io/badge/The%20AI%20Engineer-theaiengineer.dev-2E7D32.svg)](https://theaiengineer.dev/)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-Weeks%201--3%20Ready-F9AB00.svg?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
![Educational Use](https://img.shields.io/badge/Purpose-Educational%20Use-lightgrey.svg)

---

## Weekly Capstones Overview

| Week | Capstone | Key concepts | Access |
| --- | --- | --- | --- |
| **1** | **Gradient Descent Optimization** | GD/SGD dynamics, learning-rate schedules, stochastic noise, and convergence diagnostics. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/the-ai-engineer/blob/main/capstones/week01_gd_optimization/gd_capstone.ipynb) |
| **2** | **Backpropagation** | Manual backpropagation, autograd validation, two-layer neural networks, and XOR classification. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/the-ai-engineer/blob/main/capstones/week02_backprop/week02_master_capstone.ipynb) |
| **3** | **Tiny Transformer** | BPE tokenization, causal self-attention, transformer training, and FOMC text generation. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/the-ai-engineer/blob/main/capstones/week03_transformers/week03_tiny_transformer.ipynb) |
| **4** | **Agentic Incident Command** | MCP client/server workflow, OPAL agent loop, guardrails, structured telemetry, and offline replay. | [![View Submission](https://img.shields.io/badge/View-Week%204%20Submission-6f42c1.svg)](capstones/week04_agentic_incident_command/README_week04_capstone.md) |

---

## Repository Structure

```text
the-ai-engineer/
├── capstones/
│   ├── week01_gd_optimization/         # GD/SGD notebook with inline figures
│   ├── week02_backprop/                # Manual → autograd → nn.Module
│   ├── week03_transformers/            # Tiny transformer + BPE extension
│   └── week04_agentic_incident_command/
│       ├── 01_tool_harness/            # Warm-up: minimal MCP server/client
│       ├── 02_incident_command_agent/  # Primary Week 4 capstone
│       ├── artifacts/                  # Telemetry JSONL + sample summary
│       └── README_week04_capstone.md
├── pytest.ini
├── README.md
└── requirements.txt
```

---

## Environment & Reproducibility

This repository uses a lightweight **pip + venv** workflow and targets **Python 3.11**.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:

- Weeks 1–3 can be reviewed directly on GitHub and opened in Google Colab using the badges above.
- Week 4 runs locally as a multi-process MCP application: one process hosts the server, another runs the agent/client, and the resulting telemetry is written to disk for deterministic replay and audit.

---

## Educational Use

All content in this repository is provided **for educational and illustrative purposes only**. No guarantees are made regarding correctness, performance, reliability, or suitability for any production environment.

© 2026 Francisco Salazar
