# CLAUDE.md
## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
# Apple Silicon PyTorch:  pip install torch==2.3.1 --extra-index-url https://download.pytorch.org/whl/cpu
# Linux CUDA PyTorch:     pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Run
Weeks 1–2: `jupyter lab` on the primary notebook from repo root.

Week 03 modules (from `capstones/week03_transformers/`):
```bash
python scaled_dot_product_attention.py && python multihead_attention.py \
  && python transformer_block.py && python mini_transformer.py
```

Week 04 local (from `capstones/week04_agentic_incident_command/02_incident_command_agent/`):
```bash
python cli.py                                        # local OPAL loop
python cli.py --replay ../artifacts/telemetry.jsonl  # replay trace
```

Week 04 remote MCP (two terminals, same directory):
```bash
python mcp_server.py   # Terminal A
python demo_remote.py  # Terminal B
```

## Tests
Week 03 — same commands as the module run above; each file has a `__main__` suite.

Week 04:
```bash
cd capstones/week04_agentic_incident_command/02_incident_command_agent
pytest test_tools.py test_integration.py -v
```
Weeks 1–2 have no pytest files; assertions are embedded in the notebooks.

## Active Known Issues
**1. `telemetry.jsonl` grows without bound.**
Reproduction: run `python demo_remote.py` repeatedly — the file and its
`fresh_telemetry_sink()` archives (`telemetry_*.jsonl`) accumulate indefinitely.

**2. Two incompatible `telemetry.py` modules.**
`01_tool_harness/telemetry.py` uses `time.perf_counter()` + hex UUIDs;
`02_incident_command_agent/telemetry.py` uses `time.monotonic()` + full UUID strings
and adds `Budget.consume()`. Do not import one from the other.

**3. `incident_memory.py` cursor handling is a no-op.**
Reproduction: call `memory.get_resource("memory://deltas/recent", cursor="x")` —
returns the full list regardless of cursor value. (`incident_memory.py:83`)
**4. Week 3 corpus is hardcoded in two places.**
`train_mini_gpt.py:62` and `week03_master_capstone.ipynb` cell 4 both define
`tiny_text`. The notebook asserts a SHA256 hash; `train_mini_gpt.py` has no guard.
A silent divergence makes the committed `mini_gpt.pt` produce garbage in the notebook.

**5. No pytest suite for Weeks 1–3.**
`pytest` and `pytest-asyncio` are in `requirements.txt` but Weeks 1–3 have no
`test_*.py` files. Week 3 tests run only via `python <module>.py`.
