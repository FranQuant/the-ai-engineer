<img src="https://theaiengineer.dev/tae_logo_gw_flatter.png" width="35%" align="right">

# The AI Engineer

Four capstones, from-scratch: gradient descent, backpropagation, and a tiny transformer (Weeks 1–3, all Colab-ready), culminating in an MCP-based agentic incident-command system (Week 4) — graded on its remote server/client workflow, with a local deterministic run as supporting evidence.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-yellow?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Weeks_1--3-Colab_Ready-blue?logo=googlecolab&logoColor=white">
  <img src="https://img.shields.io/badge/Week_4-Remote_MCP_Workflow-purple">
  <img src="https://img.shields.io/badge/License-Educational%20Use-green">
</p>

---

## Weekly Capstones Overview

| Week  | Capstone                          | Primary artifact                                                                                                                                                                                                                                                                                                                                                                                                              | Access / delivery mode                                                                                                                                                                              |
| ----- | --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1** | **Gradient Descent Optimization** | [`gd_capstone.ipynb`](capstones/week01_gd_optimization/gd_capstone.ipynb)                                                                                                                                                                                                                                                                                                                                                     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/the-ai-engineer/blob/main/capstones/week01_gd_optimization/gd_capstone.ipynb)                   |
| **2** | **Backpropagation**               | [`week02_master_capstone.ipynb`](capstones/week02_backprop/week02_master_capstone.ipynb)                                                                                                                                                                                                                                                                                                                                      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/the-ai-engineer/blob/main/capstones/week02_backprop/week02_master_capstone.ipynb)        |
| **3** | **Tiny Transformer**              | [`week03_tiny_transformer.ipynb`](capstones/week03_transformers/week03_tiny_transformer.ipynb)                                                                                                                                                                                                                                                                                                                                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/the-ai-engineer/blob/main/capstones/week03_transformers/week03_tiny_transformer.ipynb) |
| **4** | **Agentic Incident Command**      | [`demo_remote.py`](capstones/week04_agentic_incident_command/02_incident_command_agent/demo_remote.py) · [`mcp_client.py`](capstones/week04_agentic_incident_command/02_incident_command_agent/mcp_client.py) · [`remote_agent.py`](capstones/week04_agentic_incident_command/02_incident_command_agent/remote_agent.py) · [`mcp_server.py`](capstones/week04_agentic_incident_command/02_incident_command_agent/mcp_server.py) | Remote MCP server/client workflow. The local deterministic runner is supporting evidence.                                                                      |

---

## Repository Structure

```text
the-ai-engineer/
├── capstones/
│   ├── week01_gd_optimization/   # GD/SGD notebook + 8 generated figures
│   ├── week02_backprop/          # Manual → autograd → nn.Module, one notebook
│   ├── week03_transformers/      # Tiny transformer + BPE extension, from scratch
│   └── week04_agentic_incident_command/
│       ├── 01_tool_harness/      # Warm-up: minimal MCP server/client
│       ├── 02_incident_command_agent/  # Primary capstone (graded)
│       ├── artifacts/            # Telemetry JSONL + sample summary
│       └── README_week04_capstone.md
├── pytest.ini
├── README.md
└── requirements.txt
```

---

## Week 4 Verification Entry Points

From the repository root:

```bash
# Terminal A: start the MCP server
python capstones/week04_agentic_incident_command/02_incident_command_agent/mcp_server.py

# Terminal B: run the primary graded remote MCP path
python capstones/week04_agentic_incident_command/02_incident_command_agent/demo_remote.py

# Replay a telemetry trace
python capstones/week04_agentic_incident_command/02_incident_command_agent/cli.py --replay capstones/week04_agentic_incident_command/artifacts/telemetry.jsonl

# Supporting deterministic local run
python capstones/week04_agentic_incident_command/02_incident_command_agent/cli.py

# Run the test suite
pytest capstones/week04_agentic_incident_command/02_incident_command_agent/
```

For Week 4 details, telemetry, guardrails, and architecture notes, see the dedicated Week 4 README:
[`capstones/week04_agentic_incident_command/README_week04_capstone.md`](capstones/week04_agentic_incident_command/README_week04_capstone.md)

Telemetry logs and incident summaries for inspection are stored in `capstones/week04_agentic_incident_command/artifacts/`.

---

## Environment & Reproducibility

This repository uses a lightweight **pip + venv** workflow and targets **Python 3.11**.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:

* Weeks 1–3 can be reviewed directly in GitHub and opened in Colab from the links above.
* Week 4 is designed for a local Python environment because it depends on a live MCP server/client interaction and replayable telemetry artifacts.

---

## License (Educational Use)

All content in this repository is provided **for educational and illustrative purposes only**.
No guarantees are made regarding correctness, performance, reliability, or suitability for any production environment.

© 2026 Francisco Salazar
