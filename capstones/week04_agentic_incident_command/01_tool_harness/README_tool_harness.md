
<table width="100%">
<tr>
<td style="vertical-align: top;">

<h1>MCP Tool Harness (Warm-Up)</h1>

<p>
This module provides a minimal MCP server and client used to explore the
core mechanics of the OPAL loop (Observe → Plan → Act → Learn) before
building the full Incident Command Agent in 02_incident_command_agent/.
</p>

</td>
<td align="right" width="200">
<img src="../../../assets/tae_logo.png" alt="TAE Banner" width="160">
</td>
</tr>
</table>

## Budget Note

The warm-up harness uses a relaxed `40 000 ms` ceiling because it is a
non-production trace demonstration. The capstone agent in
`02_incident_command_agent/` uses `150 ms` per the coaching guide's latency
discipline requirement. The mismatch is intentional and documented here so it
is not mistaken for a bug.

## Components
- **mcp_tool_harness_server.py** — exposes 3 tools and 3 memory:// resources  
- **mcp_tool_harness_client.py** — runs a single deterministic OPAL loop  
- **schemas.py** — JSON schemas for tools and resources  
- **telemetry.py** — structured telemetry + JSONL logging  
- **samples/** — example server/client logs  

## How to Run

**Terminal 1 (server):**
```bash
cd 01_tool_harness
python mcp_tool_harness_server.py
```
**Terminal 2 (client):**

```bash
python mcp_tool_harness_client.py
```

Logs will appear under `samples/`.


## Purpose

This warm-up demonstrates:

- MCP message flow

- Tool invocation

- Resource retrieval

- Telemetry + replay-ready traces

These foundations are used directly in **Module 02**.
