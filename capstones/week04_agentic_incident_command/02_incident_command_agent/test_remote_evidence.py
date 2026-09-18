from __future__ import annotations

import asyncio
from copy import deepcopy

from incident_memory import IncidentMemoryStore
from incident_planner import IncidentPlanner
from mcp_server import tool_summarize_incident
from remote_agent import RemoteIncidentAgent
from telemetry import Budget, RunContext, TelemetryLogger


ALERT = {
    "id": "ALRT-0001",
    "service": "staging-api",
    "symptom": "CPU spike on node-3",
    "severity": "high",
}


class MockClient:
    def __init__(self, diagnostic_status: str = "ok") -> None:
        self.diagnostic_status = diagnostic_status
        self.calls = []
        self.memory = IncidentMemoryStore()

    async def call_tool(self, name, arguments):
        self.calls.append((name, deepcopy(arguments)))
        if name == "retrieve_runbook":
            return {
                "status": "ok",
                "data": {"results": [{"id": "rb-101", "title": "High CPU playbook"}]},
                "metrics": {"latency_ms": 5},
            }
        if name == "run_diagnostic":
            if self.diagnostic_status != "ok":
                return {
                    "status": "error",
                    "error": "diagnostic unavailable",
                    "metrics": {"latency_ms": 7},
                }
            return {
                "status": "ok",
                "data": {
                    "command": arguments["command"],
                    "host": arguments["host"],
                    "stdout": "All pods healthy; CPU normalized.",
                    "stderr": "",
                },
                "metrics": {"latency_ms": 7},
            }
        if name == "summarize_incident":
            return tool_summarize_incident(arguments, self.memory)
        raise AssertionError(f"Unexpected tool call: {name}")


def _run_plan(tmp_path, observations, diagnostic_status="ok"):
    planner = IncidentPlanner(config={})
    plan = planner.plan(observations, Budget(tokens=2000, ms=150, dollars=0.0))
    summarize_step = next(step for step in plan if step["name"] == "summarize_incident")
    assert summarize_step["arguments"]["evidence"] == []

    client = MockClient(diagnostic_status=diagnostic_status)
    agent = RemoteIncidentAgent(client, planner, TelemetryLogger(tmp_path / "telemetry.jsonl"))
    ctx = RunContext(correlation_id="corr-evidence", loop_id="loop-1")
    results = asyncio.run(agent.act(ctx, plan))
    summary_call = next(arguments for name, arguments in client.calls if name == "summarize_incident")
    return plan, results, summary_call


def test_successful_results_become_concrete_summary_evidence(tmp_path):
    plan, results, summary_call = _run_plan(tmp_path, {"alerts_latest": ALERT})

    evidence = summary_call["evidence"]
    assert evidence[0] == "memory://alerts/latest#ALRT-0001"
    assert "memory://runbooks/index#rb-101" in evidence
    diagnostic = next(item for item in evidence if item.startswith("diagnostic:"))
    assert '"step":"step-2"' in diagnostic
    assert '"command":"kubectl top pod"' in diagnostic
    assert '"host":"staging-api"' in diagnostic
    assert '"result":"All pods healthy; CPU normalized."' in diagnostic
    assert "summarize_incident" not in evidence

    assert [step["name"] for step in plan] == [
        "retrieve_runbook",
        "run_diagnostic",
        "summarize_incident",
    ]
    assert [item["step"]["name"] for item in results] == [
        "retrieve_runbook",
        "run_diagnostic",
        "summarize_incident",
    ]
    assert all(set(item["result"]) >= {"status", "data", "metrics"} for item in results)


def test_failed_diagnostic_is_not_reported_as_success(tmp_path):
    _, results, summary_call = _run_plan(
        tmp_path,
        {"alerts_latest": ALERT},
        diagnostic_status="error",
    )

    assert not any(item.startswith("diagnostic:") for item in summary_call["evidence"])
    summary = results[-1]["result"]["data"]["summary"]
    assert "pods healthy" not in summary
    assert "CPU normalized" not in summary
    assert "No successful diagnostic result was available" in summary


def test_skipped_diagnostic_is_not_reported_as_success(tmp_path):
    crash_alert = dict(ALERT, symptom="pod crashloop after deploy")
    plan, results, summary_call = _run_plan(tmp_path, {"alerts_latest": crash_alert})

    assert [step["name"] for step in plan] == ["retrieve_runbook", "summarize_incident"]
    assert not any(item.startswith("diagnostic:") for item in summary_call["evidence"])
    summary = results[-1]["result"]["data"]["summary"]
    assert "pods healthy" not in summary
    assert "CPU normalized" not in summary
    assert "No successful diagnostic result was available" in summary
