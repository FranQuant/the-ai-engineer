<table width="100%">
<tr>
<td style="vertical-align: top;">

<h1>The AI Engineer</h1>

Notebook-centered ML/DL deliverables for Weeks 1–3, plus an in-progress Week 4 MCP-based agentic incident-command capstone.

</td>

<td align="right" width="200">
  <img src="assets/tae_logo.png" alt="TAE Logo" width="160">
</td>
</tr>
</table>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-yellow?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Weeks_1--3-Colab_Ready-blue?logo=googlecolab&logoColor=white">
  <img src="https://img.shields.io/badge/Week_4-Work_in_Progress-lightgrey">
  <img src="https://img.shields.io/badge/License-Educational%20Use-green">
</p>


---

## Weekly Capstones Overview

| Week  | Capstone                          | Primary artifact                                                                                                                                                                                                                                                                                                                                                                                                              | Access / delivery mode                                                                                                                                                                              |
| ----- | --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1** | **Gradient Descent Optimization** | [`gd_capstone.ipynb`](capstones/week01_gd_optimization/gd_capstone.ipynb)                                                                                                                                                                                                                                                                                                                                                     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/the-ai-engineer/blob/main/capstones/week01_gd_optimization/gd_capstone.ipynb)                   |
| **2** | **Backpropagation**               | [`week02_master_capstone.ipynb`](capstones/week02_backprop/week02_master_capstone.ipynb)                                                                                                                                                                                                                                                                                                                                      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/the-ai-engineer/blob/main/capstones/week02_backprop/week02_master_capstone.ipynb)        |
| **3** | **FOMC Transformer**              | [`week03_master_capstone.ipynb`](capstones/week03_transformers/week03_master_capstone.ipynb) · [guide](capstones/week03_transformers/README_week03_capstone.md) · [canonical results](capstones/week03_transformers/results/canonical/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/the-ai-engineer/blob/week03-capstone-v1/capstones/week03_transformers/week03_master_capstone.ipynb) |
| **4** | **Agentic Incident Command (work in progress)** | [`README_week04_capstone.md`](capstones/week04_agentic_incident_command/README_week04_capstone.md) | Not yet presented as a verified or complete release. |

---

## Repository Structure

```text
the-ai-engineer/
├── assets/
├── capstones/
│   ├── week01_gd_optimization/   # GD optimization notebook (figures regenerate on run)
│   ├── week02_backprop/          # Backprop notebook + diagnostics figure (checkpoint regenerates on run)
│   ├── week03_transformers/      # FOMC Transformer + frozen canonical results
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

## Week 3 release candidate

The Week 3 [notebook](capstones/week03_transformers/week03_master_capstone.ipynb) is the self-contained entry point. The [capstone guide](capstones/week03_transformers/README_week03_capstone.md) documents provenance, architecture, results, limitations, and the six-file [canonical artifact set](capstones/week03_transformers/results/canonical/). The future immutable Colab release is pinned to `week03-capstone-v1` and defaults to validation-only with training disabled.

Week 4 remains work in progress. Its files are preserved, but this README does not claim that release is verified or complete.

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
* Week 4 is still under development and is not part of the verified Week 3 release candidate.

---

## License (Educational Use)

All content in this repository is provided **for educational and illustrative purposes only**.
No guarantees are made regarding correctness, performance, reliability, or suitability for any production environment.

© 2026 Francisco Salazar
