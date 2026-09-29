"""
Demo script: run the remote agent against the local MCP server until the incident is
resolved or escalated, then write the handoff summary and memory snapshot.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict

from config import ARTIFACTS_DIR, MCP_SERVER_URI, MEMORY_SNAPSHOT_PATH, fresh_telemetry_sink
from incident_planner import IncidentPlanner
from mcp_client import MCPClient
from remote_agent import RemoteIncidentAgent
from telemetry import TelemetryLogger


SAMPLE_SUMMARY_PATH = ARTIFACTS_DIR / "sample_summary.md"


def _tool_result(loop: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
    for item in loop.get("results", []):
        if item.get("step", {}).get("name") == tool_name:
            return item.get("result", {})
    return {}


def _top_runbook(loop: Dict[str, Any]) -> Dict[str, Any] | None:
    result = _tool_result(loop, "retrieve_runbook")
    hits = result.get("data", {}).get("results", []) if isinstance(result, dict) else []
    for runbook in hits or []:
        if isinstance(runbook, dict) and runbook.get("title"):
            return runbook
    return None


def _render_sample_summary(incident: Dict[str, Any]) -> str:
    """Human handoff for the whole incident: terminal state, every loop, and what to do next."""
    loops = incident["loops"]
    first, final = loops[0], loops[-1]
    alert = first["observations"].get("alerts_latest", {})
    runbook = _top_runbook(final)
    summary_data = _tool_result(final, "summarize_incident").get("data", {})

    lines = [
        "# Incident Handoff Summary",
        "",
        f"- Terminal state: `{incident['state']}`",
        f"- Escalation reason: `{incident['escalation_reason']}`" if incident["escalation_reason"] else "- Escalation reason: none (resolved)",
        f"- Alert ID: `{alert.get('id', 'unknown')}`",
        f"- Service: `{alert.get('service', 'unknown')}`",
        f"- Symptom: {alert.get('symptom', 'unknown')}",
        f"- Runbook: `{runbook['title']}`" if runbook else "- Runbook: No runbook hits returned.",
        f"- Loops run: {len(loops)}",
        "",
        "## Loops",
    ]
    for loop in loops:
        ctx = loop["ctx"]
        lines.append(f"### {ctx['loop_id']} — correlation ID `{ctx['correlation_id']}`")
        lines.append(f"- Outcome: `{loop['outcome']['status']}` ({loop['outcome']['verdict']})")
        lines.append(f"- Learn note: {loop['learn']['delta_written']['note']}")
        lines.append(f"- Act status: `{loop['act']['status']}`")
        first_step = loop["plan"][0] if loop["plan"] else {}
        lines.append(f"- Plan rationale: {first_step.get('rationale', 'n/a')}")
        for idx, item in enumerate(loop["results"], 1):
            step = item.get("step", {})
            args = {k: v for k, v in step.get("arguments", {}).items() if k != "evidence"}
            lines.append(f"  {idx}. `{step.get('name', 'unknown')}` `{args}`")
        lines.append("")

    lines.extend([
        "## Summary",
        str(summary_data.get("summary", "")) or "_No summary text returned._",
        "",
        "## Recommended Actions",
    ])
    steps = list(runbook.get("steps", [])) if runbook else []
    lines.extend([f"- {action}" for action in steps] or ["- _No runbook actions available._"])
    if incident["state"] == "escalated":
        lines.extend([
            "",
            "## Escalation",
            f"The agent stopped without a healthy diagnostic ({incident['escalation_reason']}). "
            "A human should review the diagnostics above and decide on a restart.",
        ])
    return "\n".join(lines) + "\n"


def _write_sample_summary(incident: Dict[str, Any]) -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_SUMMARY_PATH.write_text(_render_sample_summary(incident), encoding="utf-8")
    return SAMPLE_SUMMARY_PATH


def _write_memory_snapshot(incident: Dict[str, Any]) -> Path:
    snapshot = {
        "run_id": incident["run_id"],
        "terminal_state": incident["state"],
        "escalation_reason": incident["escalation_reason"],
        **incident["memory_snapshot"],
    }
    MEMORY_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return MEMORY_SNAPSHOT_PATH


async def main() -> None:
    """Connect to MCP, loop until resolved or escalated, write artifacts, then close."""
    telemetry = TelemetryLogger(fresh_telemetry_sink())
    client = MCPClient(uri=MCP_SERVER_URI, telemetry=telemetry)
    await client.connect()

    planner = IncidentPlanner(config={})
    agent = RemoteIncidentAgent(client, planner, telemetry)

    try:
        incident = await agent.run_incident()
    finally:
        await client.close()

    summary_path = _write_sample_summary(incident)
    snapshot_path = _write_memory_snapshot(incident)

    print("=== Remote Incident Run ===")
    print(f"State: {incident['state']} ({incident['reason']}); loops: {len(incident['loops'])}")
    print(f"Sample summary written to {summary_path}")
    print(f"Memory snapshot written to {snapshot_path}")


if __name__ == "__main__":
    asyncio.run(main())
