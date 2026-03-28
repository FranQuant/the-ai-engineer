<table width="100%">
<tr>
<td style="vertical-align: top;">

<h1>The AI Engineer — Capstone Projects</h1>

This repository contains all capstone projects for <i>The AI Engineer</i> program (Nov 2025 Cohort).  
Each week builds a complete, self-contained project with a clean software-engineering structure, reproducibility, diagnostics, and proper documentation.

</td>

<td align="right" width="200">
  <img src="assets/tae_logo.png" alt="TAE Logo" width="160">
</td>
</tr>
</table>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-yellow?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Colab-Friendly-blue?logo=googlecolab&logoColor=white">
  <img src="https://img.shields.io/badge/License-Educational%20Use-green">
</p>

---

## Weekly Capstones Overview

| Week | Capstone | Summary | Colab Link |
|------|----------|---------|------------|
| **1** | **Gradient Descent Optimization** | Implement GD & SGD from scratch, analyze convergence, step-size sensitivity, and basin-dependent dynamics. | [Open in Colab](https://colab.research.google.com/github/FranQuant/the_ai_engineer_capstones/blob/main/capstones/week01_gd_optimization/gd_capstone.ipynb) |
| **2** | **Backpropagation** | Manual chain rule, custom autograd, tiny MLP, PyTorch autograd, and nn.Module training loop. | [01](https://colab.research.google.com/github/FranQuant/the_ai_engineer_capstones/blob/main/capstones/week02_backprop/01_numpy_manual.ipynb) • [02](https://colab.research.google.com/github/FranQuant/the_ai_engineer_capstones/blob/main/capstones/week02_backprop/02_pytorch_no_autograd.ipynb) • [03](https://colab.research.google.com/github/FranQuant/the_ai_engineer_capstones/blob/main/capstones/week02_backprop/03_pytorch_autograd.ipynb) • [04](https://colab.research.google.com/github/FranQuant/the_ai_engineer_capstones/blob/main/capstones/week02_backprop/04_pytorch_nn_module.ipynb) |
| **3** | **Tiny Transformer** | Build tokenizer, SDPA, MHA, pre-LN transformer block, decoder-only model, training loop, sampling, and a full diagnostics suite. | [Diagnostics Notebook](https://colab.research.google.com/github/FranQuant/the_ai_engineer_capstones/blob/main/capstones/week03_transformers/mini_gpt_diagnostics.ipynb) |
| **4** | **Agent Demo** | Minimal LLM-powered agent with clean abstractions, tracing, telemetry, and a deterministic OPAL loop implementation. | Local only — [README](capstones/week04_agentic_incident_command/README_week04_capstone.md) |

---

## Repository Structure

```text
the_ai_engineer
    ├── artifacts
    │   └── telemetry.jsonl                      # Accumulated OPAL telemetry (all Week 4 runs)
    ├── assets
    │   ├── mcp_server_startup.png
    │   ├── remote_opal_loop.png
    │   ├── tae_logo.png
    │   └── telemetry_jsonl_confirmation.png
    ├── capstones
    │   ├── week01_gd_optimization                # Gradient Descent capstone
    │   │   ├── gd_capstone.ipynb
    │   │   └── README_week01_capstone.md
    │   ├── week02_backprop                       # Backpropagation capstone
    │   │   ├── 01_numpy_manual.ipynb
    │   │   ├── 02_pytorch_no_autograd.ipynb
    │   │   ├── 03_pytorch_autograd.ipynb
    │   │   ├── 04_pytorch_nn_module.ipynb
    │   │   ├── two_layer_xor.pt                 # Saved checkpoint from nb04
    │   │   └── README_week02_capstone.md
    │   ├── week03_transformers                   # Tiny Transformer capstone
    │   │   ├── mini_gpt_diagnostics.ipynb
    │   │   ├── mini_gpt.pt                      # Saved model checkpoint
    │   │   ├── mini_transformer.py
    │   │   ├── multihead_attention.py
    │   │   ├── README_week03_capstone.md
    │   │   ├── run_record.json                  # Auto-generated training run record
    │   │   ├── scaled_dot_product_attention.py
    │   │   ├── train_mini_gpt.py
    │   │   └── transformer_block.py
    │   └── week04_agentic_incident_command       # MCP/Agent demo + OPAL loop
    │       ├── 01_tool_harness
    │       ├── 02_incident_command_agent
    │       │   ├── cli.py
    │       │   ├── demo_remote.py
    │       │   ├── incident_agent.py
    │       │   ├── incident_memory.py
    │       │   ├── incident_planner.py          # Adaptive FSM planner
    │       │   ├── incident_schemas.py
    │       │   ├── mcp_client.py
    │       │   ├── mcp_server.py                # 8 tools incl. append_memory_delta
    │       │   ├── replay.py
    │       │   ├── telemetry.py
    │       │   └── test_tools.py                # pytest suite for tool handlers
    │       ├── artifacts
    │       │   └── telemetry.jsonl
    │       └── README_week04_capstone.md
    ├── README.md
    └── requirements.txt

```
---

## Environment & Reproducibility

This project uses a lightweight **pip + venv** setup for consistency and reproducibility.  
All notebooks and scripts have been tested on **Python 3.11**.

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
---

### License (Educational Use)

All content in this repository is provided **for educational and illustrative purposes only**.  
No guarantees are made regarding correctness, performance, reliability, or suitability for any production environment.

---

### Agentic Systems Notice

Agentic systems — especially those capable of taking actions, orchestrating tools, or modifying state — can introduce **significant safety risks**.

Before using any such system outside a controlled environment, always:

- Validate all outputs manually  
- Run code inside a sandboxed environment  
- Apply strict guardrails and permissions  
- Never connect an agent to real infrastructure without full safety checks  

Use responsibly.

© 2025 Francisco Salazar

---