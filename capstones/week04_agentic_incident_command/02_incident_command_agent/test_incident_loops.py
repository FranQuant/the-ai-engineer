"""
Tests for multi-loop incident runs: terminal states (resolved / escalated), the max-loop
guardrail, per-loop correlation and loop IDs on client AND server events, delta contents,
the telemetry-snapshot resource, per-tool token accounting, and truthful phase-end status.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
import websockets

import incident_loop
from incident_agent import IncidentAgent
from incident_memory import IncidentMemoryStore
from incident_planner import IncidentPlanner
from incident_schemas import get_tool_schemas, tool_cost_tokens
from mcp_client import MCPClient
from mcp_server import call_tool, handle_session
from remote_agent import RemoteIncidentAgent
from config import DEFAULT_MAX_LOOPS as MAX_LOOPS
from telemetry import Budget, RunContext, TelemetryEvent, TelemetryLogger


def _read_events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


async def _remote_incident(tmp_path, max_loops=None, planner=None):
    """Run the remote agent against a real in-process WebSocket MCP server."""
    trace = tmp_path / "telemetry.jsonl"
    logger = TelemetryLogger(trace)
    memory = IncidentMemoryStore()

    async def handler(ws):
        await handle_session(ws, logger, memory)

    async with websockets.serve(handler, "127.0.0.1", 0, subprotocols=["mcp"]) as server:
        port = server.sockets[0].getsockname()[1]
        client = MCPClient(uri=f"ws://127.0.0.1:{port}/mcp", telemetry=logger)
        await client.connect()
        agent = RemoteIncidentAgent(client, planner or IncidentPlanner(config={}), logger)
        if max_loops is not None:
            agent.max_loops = max_loops
        try:
            incident = await agent.run_incident()
        finally:
            await client.close()
    return incident, _read_events(trace)


# ---------------------------------------------------------------------------
# Loop semantics
# ---------------------------------------------------------------------------

def test_two_loops_end_escalated_with_distinct_ids(tmp_path):
    incident, events = asyncio.run(_remote_incident(tmp_path))

    assert incident["state"] == "escalated"
    assert [loop["ctx"]["loop_id"] for loop in incident["loops"]] == ["loop-1", "loop-2"]
    corr_ids = [loop["ctx"]["correlation_id"] for loop in incident["loops"]]
    assert len(set(corr_ids)) == 2

    # Each correlation ID appears on client events (owner "agent") AND server events, with its own loop ID.
    for corr_id, loop_id in zip(corr_ids, ["loop-1", "loop-2"]):
        mine = [e for e in events if e["correlation_id"] == corr_id]
        assert {e["loop_id"] for e in mine} == {loop_id}
        assert {e["budget_owner"] for e in mine} == {"agent", "server"}
        assert any(e["phase"] == "rpc_send" for e in mine)

    terminal = [e for e in events if e["phase"] == "incident_end"]
    assert len(terminal) == 1
    assert terminal[0]["status"] == "escalated"
    assert terminal[0]["payload"]["state"] == "escalated"


def test_planner_decision_escalates_after_two_loops(tmp_path):
    incident, events = asyncio.run(_remote_incident(tmp_path))

    assert MAX_LOOPS == 3  # the guardrail is only a backstop
    assert len(incident["loops"]) == 2 < MAX_LOOPS
    assert incident["state"] == "escalated"
    assert incident["escalation_reason"] == "evidence_inconclusive_after_runbook_check"
    assert incident["reason"] == incident["escalation_reason"]

    decisions = [(e["loop_id"], e["method"], e["payload"]["reason"]) for e in events if e["phase"] == "plan_decision"]
    assert [d[:2] for d in decisions] == [("loop-1", "continue"), ("loop-2", "escalate")]
    assert decisions[1][2] == "evidence_inconclusive_after_runbook_check"
    assert not [e for e in events if e["phase"] == "loop_guardrail"]

    end = next(e for e in events if e["phase"] == "incident_end")
    assert end["payload"]["escalation_reason"] == "evidence_inconclusive_after_runbook_check"


class _NeverDecidesPlanner(IncidentPlanner):
    """Forced-inconclusive planner: never resolves or escalates, so only max_loops can stop the run."""

    @staticmethod
    def decide(loop_outcomes):
        return {"action": "continue", "reason": "forced inconclusive", "ran": []}


def test_max_loops_guardrail_is_the_backstop_when_planner_never_decides(tmp_path):
    incident, events = asyncio.run(_remote_incident(tmp_path, max_loops=2, planner=_NeverDecidesPlanner(config={})))

    assert incident["state"] == "escalated"
    assert incident["escalation_reason"] == "max_loops_reached"
    assert len(incident["loops"]) == 2
    guardrails = [e for e in events if e["phase"] == "loop_guardrail"]
    assert [g["loop_id"] for g in guardrails] == ["loop-2"]
    assert guardrails[0]["payload"] == {"reason": "max_loops_reached", "allowed": 2}


def test_loop_two_plan_cites_loop_one_deltas(tmp_path):
    incident, _ = asyncio.run(_remote_incident(tmp_path))
    loop1, loop2 = incident["loops"]

    commands = [
        item["step"]["arguments"]["command"]
        for loop in (loop1, loop2)
        for item in loop["results"]
        if item["step"]["name"] == "run_diagnostic"
    ]
    assert commands == [incident_loop.FIRST_CHECK, incident_loop.FURTHER_CHECK]

    assert all(step["cites_deltas"] == [] for step in loop1["plan"])
    for step in loop2["plan"]:
        cited = step["cites_deltas"]
        assert [c["loop_id"] for c in cited] == ["loop-1"]
        assert cited[0]["note"] == loop1["learn"]["delta_written"]["note"]
        assert "loop-1" in step["rationale"]


def test_resolved_when_diagnostic_is_healthy(tmp_path, monkeypatch):
    monkeypatch.setitem(
        incident_loop._DIAGNOSTIC_FIXTURES,
        incident_loop.FIRST_CHECK,
        {"verdict": "healthy", "stdout": "All pods healthy; CPU normalized."},
    )
    incident, events = asyncio.run(_remote_incident(tmp_path))

    assert incident["state"] == "resolved"
    assert incident["escalation_reason"] is None
    assert len(incident["loops"]) == 1
    assert [e["status"] for e in events if e["phase"] == "incident_end"] == ["ok"]


def test_max_loop_guardrail_stops_the_run(tmp_path):
    incident, events = asyncio.run(_remote_incident(tmp_path, max_loops=1))

    assert incident["state"] == "escalated"
    assert incident["escalation_reason"] == "max_loops_reached"
    assert len(incident["loops"]) == 1
    guardrails = [e for e in events if e["phase"] == "loop_guardrail"]
    assert len(guardrails) == 1
    assert guardrails[0]["payload"]["reason"] == "max_loops_reached"


def test_local_agent_uses_same_loop_semantics(tmp_path):
    telemetry = TelemetryLogger(tmp_path / "telemetry.jsonl")
    agent = IncidentAgent(IncidentMemoryStore(), IncidentPlanner(config={}), telemetry)

    incident = asyncio.run(agent.run_incident())

    assert incident["state"] == "escalated"
    assert incident["escalation_reason"] == "evidence_inconclusive_after_runbook_check"
    assert [loop["ctx"]["loop_id"] for loop in incident["loops"]] == ["loop-1", "loop-2"]
    assert len({loop["ctx"]["correlation_id"] for loop in incident["loops"]}) == 2
    assert incident["loops"][1]["plan"][0]["cites_deltas"][0]["loop_id"] == "loop-1"


# ---------------------------------------------------------------------------
# Deltas
# ---------------------------------------------------------------------------

def test_every_delta_carries_alert_id_iso_timestamp_and_note(tmp_path):
    incident, _ = asyncio.run(_remote_incident(tmp_path))
    items = incident["memory_snapshot"]["deltas"]["items"]

    assert items
    for delta in items:
        assert delta["alert_id"] == "ALRT-0001"
        datetime.fromisoformat(delta["timestamp"])
        assert delta["key"] == f"{delta['alert_id']}@{delta['timestamp']}"

    loop_deltas = [d for d in items if d.get("kind") == "loop_outcome"]
    assert [d["loop_id"] for d in loop_deltas] == ["loop-1", "loop-2"]
    assert loop_deltas[0]["note"].startswith("Diagnosed staging-api, ran kubectl top pod, outcome inconclusive")
    assert any(d.get("action") == "summarize_incident" for d in items)


def test_memory_store_fills_missing_delta_fields():
    memory = IncidentMemoryStore()
    stored = memory.append_delta({"action": "manual"})

    assert stored["alert_id"] == "ALRT-0001"
    datetime.fromisoformat(stored["timestamp"])


# ---------------------------------------------------------------------------
# Server: loop ID, phase label, telemetry snapshot
# ---------------------------------------------------------------------------

class _WS:
    def __init__(self, messages):
        self._queue = [json.dumps(m) for m in messages]
        self.sent = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._queue:
            raise StopAsyncIteration
        return self._queue.pop(0)

    async def send(self, data):
        self.sent.append(json.loads(data))


def _serve(tmp_path, messages):
    ws = _WS(messages)
    logger = TelemetryLogger(tmp_path / "tel.jsonl")
    asyncio.run(handle_session(ws, logger, IncidentMemoryStore()))
    return ws.sent, _read_events(tmp_path / "tel.jsonl")


def test_server_uses_loop_id_and_phase_from_request_meta(tmp_path):
    meta = {"correlationId": "c-1", "loopId": "loop-7", "phase": "learn"}
    _, events = _serve(tmp_path, [
        {"jsonrpc": "2.0", "id": 1, "method": "getResource",
         "params": {"uri": "memory://deltas/recent", "_meta": meta}},
    ])

    assert (events[0]["loop_id"], events[0]["phase"], events[0]["correlation_id"]) == ("loop-7", "learn", "c-1")
    assert events[0]["budget_owner"] == "server"


def test_server_without_meta_has_no_hardcoded_loop(tmp_path):
    _, events = _serve(tmp_path, [
        {"jsonrpc": "2.0", "id": 1, "method": "getResource", "params": {"uri": "memory://deltas/recent"}},
    ])

    assert events[0]["loop_id"] == "unscoped"
    assert events[0]["phase"] == "observe"


def test_telemetry_snapshot_resource_is_advertised_and_readable(tmp_path):
    sent, _ = _serve(tmp_path, [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "getResource",
         "params": {"uri": "memory://telemetry/snapshot", "_meta": {"loopId": "loop-1"}}},
    ])

    uris = [r["uri"] for r in sent[0]["result"]["resources"]]
    assert "memory://telemetry/snapshot" in uris
    snapshot = sent[1]["result"]
    assert snapshot["total_events"] == 1  # the initialize call logged before this read
    assert snapshot["last_events"][0]["method"] == "initialize"


def test_agent_reads_telemetry_snapshot_before_planning(tmp_path):
    incident, events = asyncio.run(_remote_incident(tmp_path))
    loop1 = incident["loops"][0]

    assert "total_events" in loop1["observations"]["telemetry_snapshot"]
    reads = [
        (i, e) for i, e in enumerate(events)
        if e["phase"] == "rpc_send"
        and e["payload"]["request"].get("params", {}).get("uri") == "memory://telemetry/snapshot"
    ]
    plan_start = next(i for i, e in enumerate(events) if e["phase"] == "plan_start")
    assert reads and reads[0][0] < plan_start


# ---------------------------------------------------------------------------
# Budgets and status
# ---------------------------------------------------------------------------

def test_tool_cost_tokens_match_advertised_hints():
    memory = IncidentMemoryStore()
    calls = {
        "retrieve_runbook": {"query": "cpu"},
        "run_diagnostic": {"command": "kubectl top pod", "host": "staging-api"},
        "summarize_incident": {"alert_id": "ALRT-0001", "evidence": []},
        "write_plan": {"plan": []},
        "append_memory_delta": {"delta": {"note": "x"}},
        "create_incident": {"id": "INC-9", "title": "t"},
        "add_evidence": {"content": "c"},
        "append_delta": {"action": "a"},
    }
    assert set(calls) == set(get_tool_schemas())
    for name, arguments in calls.items():
        metrics = call_tool(memory, name, arguments)["metrics"]
        assert metrics["cost_tokens"] == tool_cost_tokens(name), name
        assert metrics["cost_dollars"] == 0.0


def test_logger_does_not_debit_the_budget(tmp_path):
    budget = Budget(tokens=100, ms=100, dollars=0.0)
    logger = TelemetryLogger(tmp_path / "t.jsonl")
    logger.log(TelemetryEvent("c", "loop-1", "act_step", "m", "ok", 40, budget, {}))

    assert (budget.tokens, budget.ms) == (100, 100)


def _act_with_results(tmp_path, results, tokens=None):
    client = MagicMock()
    client.call_tool = AsyncMock(side_effect=results)
    telemetry = TelemetryLogger(tmp_path / "telemetry.jsonl")
    agent = RemoteIncidentAgent(client, MagicMock(), telemetry)
    if tokens is not None:
        agent.budget.tokens = tokens
    steps = [
        {"type": "callTool", "step_id": f"step-{i}", "name": "retrieve_runbook", "arguments": {"query": "cpu"}}
        for i in (1, 2, 3)
    ]
    out = asyncio.run(agent.act(RunContext("c", "loop-1"), steps))
    return out, _read_events(tmp_path / "telemetry.jsonl")


def _ok(cost):
    return {"status": "ok", "data": {}, "metrics": {"latency_ms": 1, "cost_tokens": cost}}


def test_agent_aborts_when_token_budget_is_exhausted(tmp_path):
    results, events = _act_with_results(tmp_path, [_ok(10), _ok(10), _ok(10)], tokens=15)

    assert len(results) == 2
    guardrail = [e for e in events if e["phase"] == "act_guardrail"]
    assert [g["payload"]["reason"] for g in guardrail] == ["token_budget_exceeded"]
    act_end = next(e for e in events if e["phase"] == "act_end")
    assert act_end["status"] == "guardrail_stop"
    assert act_end["payload"]["stop_reason"] == "token_budget_exceeded"


def test_act_end_reports_error_when_a_step_failed(tmp_path):
    failed = {"status": "error", "error": "boom", "metrics": {"latency_ms": 1}}
    _, events = _act_with_results(tmp_path, [_ok(1), failed, _ok(1)])

    assert next(e for e in events if e["phase"] == "act_end")["status"] == "error"


def test_act_end_is_ok_when_everything_succeeds(tmp_path):
    _, events = _act_with_results(tmp_path, [_ok(1), _ok(1), _ok(1)])

    assert next(e for e in events if e["phase"] == "act_end")["status"] == "ok"


@pytest.mark.parametrize("field", ["tokens", "ms"])
def test_incident_escalates_when_loop_budget_is_exhausted(tmp_path, field):
    async def _run():
        client = MagicMock()
        client.set_context = MagicMock()
        client.initialize = AsyncMock(return_value={})
        alert = {"id": "ALRT-0001", "service": "staging-api", "symptom": "CPU spike", "severity": "high", "items": []}
        client.get_resource = AsyncMock(return_value=alert)
        cost = {"cost_tokens": 5000} if field == "tokens" else {"latency_ms": 500}
        client.call_tool = AsyncMock(return_value={"status": "ok", "data": {}, "metrics": cost})
        agent = RemoteIncidentAgent(client, IncidentPlanner(config={}), TelemetryLogger(tmp_path / "t.jsonl"))
        return await agent.run_incident()

    incident = asyncio.run(_run())

    assert incident["state"] == "escalated"
    assert incident["reason"] == f"{'token' if field == 'tokens' else 'ms'}_budget_exceeded"
    assert len(incident["loops"]) == 1
