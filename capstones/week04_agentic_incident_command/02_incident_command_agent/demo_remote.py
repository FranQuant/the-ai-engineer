"""
Demo script: run a single remote OPAL loop against the local MCP server.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict

from config import ARTIFACTS_DIR, MCP_SERVER_URI, fresh_telemetry_sink
from incident_planner import IncidentPlanner
from mcp_client import MCPClient
from remote_agent import RemoteIncidentAgent
from telemetry import RunContext, TelemetryLogger, new_correlation_id


SAMPLE_SUMMARY_PATH = ARTIFACTS_DIR / "sample_summary.md"


def _top_runbook(retrieved: list[Dict[str, Any]]) -> Dict[str, Any] | None:
    for runbook in retrieved:
        if isinstance(runbook, dict) and runbook.get("title"):
            return runbook
    return None


def _extract_tool_result(summary: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
    for item in summary.get("results", []):
        step = item.get("step", {})
        if step.get("name") == tool_name:
            return item.get("result", {})
    return {}


def _render_sample_summary(summary: Dict[str, Any], correlation_id: str) -> str:
    observations = summary.get("observations", {})
    alert = observations.get("alerts_latest", {}) if isinstance(observations, dict) else {}
    service = str(alert.get("service", "unknown"))
    alert_id = str(alert.get("id", "unknown"))

    summarize_result = _extract_tool_result(summary, "summarize_incident")
    summarize_data = summarize_result.get("data", {}) if isinstance(summarize_result, dict) else {}
    summary_text = str(summarize_data.get("summary", ""))

    retrieve_result = _extract_tool_result(summary, "retrieve_runbook")
    retrieved_runbooks = []
    if isinstance(retrieve_result, dict):
        retrieved_runbooks = retrieve_result.get("data", {}).get("results", []) or []

    runbook = _top_runbook(retrieved_runbooks)
    recommended_actions = list(runbook.get("steps", [])) if isinstance(runbook, dict) else []

    executed_steps = []
    for item in summary.get("results", []):
        step = item.get("step", {}) if isinstance(item, dict) else {}
        name = step.get("name", "unknown")
        arguments = step.get("arguments", {}) if isinstance(step, dict) else {}
        executed_steps.append((name, arguments))

    lines = [
        "# Remote Incident Summary",
        "",
        f"- Correlation ID: `{correlation_id}`",
        f"- Alert ID: `{alert_id}`",
        f"- Service: `{service}`",
        f"- Runbook: `{runbook.get('title', 'unknown')}`" if isinstance(runbook, dict) else "- Runbook: No runbook hits returned.",
        "",
        "## Executed Plan",
    ]

    for idx, (name, arguments) in enumerate(executed_steps, 1):
        if arguments:
            lines.append(f"{idx}. `{name}` with `{arguments}`")
        else:
            lines.append(f"{idx}. `{name}`")

    lines.extend([
        "",
        "## Summary",
        summary_text or "_No summary text returned._",
        "",
        "## Recommended Actions",
    ])

    if recommended_actions:
        lines.extend([f"- {action}" for action in recommended_actions])
    else:
        lines.append("- _No runbook actions available._")

    return "\n".join(lines) + "\n"


def _write_sample_summary(summary: Dict[str, Any], ctx: RunContext) -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    text = _render_sample_summary(summary, ctx.correlation_id)
    SAMPLE_SUMMARY_PATH.write_text(text, encoding="utf-8")
    return SAMPLE_SUMMARY_PATH


async def main() -> None:
    """Connect to MCP, run one OPAL loop, print summary, then close."""
    telemetry = TelemetryLogger(fresh_telemetry_sink())
    client = MCPClient(uri=MCP_SERVER_URI, telemetry=telemetry)
    await client.connect()

    planner = IncidentPlanner(config={})
    agent = RemoteIncidentAgent(client, planner, telemetry)
    ctx = RunContext(correlation_id=new_correlation_id(), loop_id="loop-1")

    # ------------------------------------------------------------
    # Run a full OPAL loop
    # ------------------------------------------------------------
    summary: Dict[str, Any] = await agent.run_loop(ctx)
    summary_path = _write_sample_summary(summary, ctx)

    print("=== Remote OPAL Summary ===")
    print(summary)
    print(f"Sample summary written to {summary_path}")

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
