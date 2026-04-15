import asyncio
import json

import pytest

from incident_agent import IncidentAgent
from incident_memory import IncidentMemoryStore
from incident_planner import IncidentPlanner
from telemetry import RunContext, TelemetryLogger


@pytest.mark.asyncio
def test_full_incident_loop_emits_tool_calls_and_learn(tmp_path):
    telemetry_path = tmp_path / "telemetry.jsonl"

    memory = IncidentMemoryStore()
    planner = IncidentPlanner(config={})
    telemetry = TelemetryLogger(telemetry_path)
    agent = IncidentAgent(memory, planner, telemetry)

    fixed_correlation_id = "corr-fixed"
    ctx = RunContext(correlation_id=fixed_correlation_id, loop_id="loop-1")

    # Drive the coroutine explicitly so the loop boundary is obvious.
    result = asyncio.run(agent.run_loop(ctx))

    events = [
        json.loads(line)
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert any(
        event["method"] == "retrieve_runbook" and event["status"] == "ok"
        for event in events
    )
    assert any(
        event["method"] == "run_diagnostic" and event["status"] == "ok"
        for event in events
    )
    assert any(event["phase"] == "learn_end" for event in events)

    deltas_recent = memory.get_resource("memory://deltas/recent")
    assert len(deltas_recent["items"]) > 0

    final_summary = result["results"][-1]["result"]["data"]
    assert final_summary["citations"]
    assert final_summary["recommended_actions"]
    assert "Citations:" in final_summary["summary"]
    assert "Recommended actions:" in final_summary["summary"]
