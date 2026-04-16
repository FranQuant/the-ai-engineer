<table width="100%">
<tr>
<td style="vertical-align: top;">

<h1>The AI Engineer</h1>

Weeks 1–3 are notebook-centered ML/DL deliverables.  
Week 4 is an MCP-based agentic systems capstone whose <b>primary submission path</b> is the remote MCP server/client workflow, with a local deterministic mirror retained as supporting evidence for replay, debugging, and reviewer validation.

</td>

<td align="right" width="200">
  <img src="assets/tae_logo.png" alt="TAE Logo" width="160">
</td>
</tr>
</table>

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
| **1** | **Gradient Descent Optimization** | [`gd_capstone.ipynb`](capstones/week01_gd_optimization/gd_capstone.ipynb)                                                                                                                                                                                                                                                                                                                                                     | Colab-ready notebook — [Open in Colab](https://colab.research.google.com/github/FranQuant/the_ai_engineer_capstones/blob/main/capstones/week01_gd_optimization/gd_capstone.ipynb)                   |
| **2** | **Backpropagation**               | [`week02_master_capstone.ipynb`](capstones/week02_backprop/week02_master_capstone.ipynb)                                                                                                                                                                                                                                                                                                                                      | Colab-ready master notebook — [Open in Colab](https://colab.research.google.com/github/FranQuant/the_ai_engineer_capstones/blob/main/capstones/week02_backprop/week02_master_capstone.ipynb)        |
| **3** | **Tiny Transformer**              | [`week03_master_capstone.ipynb`](capstones/week03_transformers/week03_master_capstone.ipynb)                                                                                                                                                                                                                                                                                                                                | Colab-ready master notebook — [Open in Colab](https://colab.research.google.com/github/FranQuant/the_ai_engineer_capstones/blob/main/capstones/week03_transformers/week03_master_capstone.ipynb) |
| **4** | **Agentic Incident Command**      | [`demo_remote.py`](capstones/week04_agentic_incident_command/02_incident_command_agent/demo_remote.py) · [`mcp_client.py`](capstones/week04_agentic_incident_command/02_incident_command_agent/mcp_client.py) · [`remote_agent.py`](capstones/week04_agentic_incident_command/02_incident_command_agent/remote_agent.py) · [`mcp_server.py`](capstones/week04_agentic_incident_command/02_incident_command_agent/mcp_server.py) | Remote MCP server/client workflow. The local deterministic runner is supporting evidence.                                                                      |

---

## Repository Structure

Reviewer-facing entrypoints:

```text
the_ai_engineer_capstones/
├── README.md
├── requirements.txt
├── assets/
│   └── tae_logo.png
└── capstones/
    ├── week01_gd_optimization/
    │   ├── gd_capstone.ipynb
    │   └── README_week01_capstone.md
    ├── week02_backprop/
    │   ├── week02_master_capstone.ipynb
    │   └── README_week02_capstone.md
    ├── week03_transformers/
    │   ├── week03_master_capstone.ipynb
    │   ├── mini_gpt.pt
    │   ├── run_record.json
    │   ├── train_mini_gpt.py
    │   └── README_week03_capstone.md
    └── week04_agentic_incident_command/
        ├── README_week04_capstone.md
        ├── artifacts/
        │   └── telemetry.jsonl
        └── 02_incident_command_agent/
            ├── config.py
            ├── demo_remote.py
            ├── remote_agent.py
            ├── mcp_server.py
            ├── mcp_client.py
            ├── cli.py
            ├── replay.py
            └── test_tools.py
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
```

For Week 4 details, telemetry, guardrails, and architecture notes, see the dedicated Week 4 README:
[`capstones/week04_agentic_incident_command/README_week04_capstone.md`](capstones/week04_agentic_incident_command/README_week04_capstone.md)

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

© 2025 Francisco Salazar
