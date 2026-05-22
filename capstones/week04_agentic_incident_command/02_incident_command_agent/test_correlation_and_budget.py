"""
Tests for (1) correlation-id consistency across client+server telemetry,
(2) ms-budget guardrail halting act execution on breach,
(3) server rejects non-dict params without crashing handle_session, and
(4) mcp_client raises a RuntimeError on recv timeout.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).parent))

import pytest

from config import ARTIFACTS_DIR, DEFAULT_BUDGET_MS
from incident_memory import IncidentMemoryStore
from mcp_client import MCPClient
from mcp_server import handle_session
from remote_agent import RemoteIncidentAgent
from telemetry import RunContext, TelemetryLogger


# ---------------------------------------------------------------------------
# Shared mock WebSocket for server-side unit tests
# ---------------------------------------------------------------------------

class _MockWS:
    """Minimal async-iterable WebSocket stub for testing handle_session."""

    def __init__(self, messages: list[str]) -> None:
        self._queue = list(messages)
        self.sent: list[dict] = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._queue:
            raise StopAsyncIteration
        return self._queue.pop(0)

    async def send(self, data: str) -> None:
        self.sent.append(json.loads(data))


# ---------------------------------------------------------------------------
# Test 1: correlation-id is consistent across client send/recv and server
#         handling for every JSON-RPC request id in the telemetry artifact.
# ---------------------------------------------------------------------------

def test_telemetry_correlation_ids_consistent():
    telemetry_path = ARTIFACTS_DIR / "telemetry.jsonl"
    assert telemetry_path.exists(), (
        f"telemetry.jsonl not found at {telemetry_path}; "
        "run demo_remote.py against a live server first."
    )

    events = [
        json.loads(line)
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # Map each JSON-RPC id to the set of correlation_ids seen in any event
    # that carries that id in its request or response payload.
    id_to_corr_ids: dict[int, set[str]] = defaultdict(set)

    for event in events:
        payload = event.get("payload", {})
        req = payload.get("request") or {}
        resp = payload.get("response") or {}
        rpc_id = req.get("id") if req.get("id") is not None else resp.get("id")
        if rpc_id is not None:
            id_to_corr_ids[rpc_id].add(event["correlation_id"])

    assert id_to_corr_ids, "No JSON-RPC ids found in telemetry — artifact may be empty."

    mismatches = {
        rpc_id: corr_ids
        for rpc_id, corr_ids in id_to_corr_ids.items()
        if len(corr_ids) != 1
    }
    assert not mismatches, (
        "These JSON-RPC ids have mixed correlation_ids across client/server events:\n"
        + "\n".join(f"  id={k}: {v}" for k, v in mismatches.items())
    )


# ---------------------------------------------------------------------------
# Test 2: act() emits act_guardrail/ms_budget_exceeded when a tool result's
#         latency_ms exhausts the budget.
# ---------------------------------------------------------------------------

def test_ms_budget_guardrail_fires_on_over_budget_latency(tmp_path):
    over_budget_latency = DEFAULT_BUDGET_MS + 10  # guaranteed to exhaust budget

    mock_client = MagicMock()
    mock_client.call_tool = AsyncMock(return_value={
        "status": "ok",
        "data": {},
        "metrics": {"latency_ms": over_budget_latency},
    })

    mock_planner = MagicMock()
    telemetry_path = tmp_path / "telemetry.jsonl"
    telemetry = TelemetryLogger(telemetry_path)

    agent = RemoteIncidentAgent(mock_client, mock_planner, telemetry)

    steps = [
        {
            "type": "callTool",
            "step_id": "step-1",
            "name": "run_diagnostic",
            "arguments": {"command": "kubectl top pod", "host": "staging-api"},
        },
        {
            "type": "callTool",
            "step_id": "step-2",
            "name": "summarize_incident",
            "arguments": {"alert_id": "ALRT-0001", "evidence": []},
        },
    ]

    ctx = RunContext(correlation_id="test-budget-corr", loop_id="loop-budget-test")
    results = asyncio.run(agent.act(ctx, steps))

    events = [
        json.loads(line)
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    guardrail_events = [e for e in events if e["phase"] == "act_guardrail"]
    assert guardrail_events, "Expected at least one act_guardrail event."

    ms_exceeded = [
        e for e in guardrail_events
        if e.get("payload", {}).get("reason") == "ms_budget_exceeded"
    ]
    assert ms_exceeded, (
        f"Expected act_guardrail with reason='ms_budget_exceeded', "
        f"got: {[e['payload'] for e in guardrail_events]}"
    )

    # Execution must have been halted: only the first step should have run.
    assert len(results) == 1, (
        f"Expected execution to stop after 1 step, but got {len(results)} results."
    )


# ---------------------------------------------------------------------------
# Test 3: handle_session returns -32602 for non-dict params without crashing.
# ---------------------------------------------------------------------------

def test_server_non_dict_params_returns_32602(tmp_path):
    bad_request = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": "bad_string",
    })
    ws = _MockWS([bad_request])
    memory = IncidentMemoryStore()
    logger = TelemetryLogger(tmp_path / "tel.jsonl")

    asyncio.run(handle_session(ws, logger, memory))

    assert ws.sent, "Expected at least one response from handle_session."
    resp = ws.sent[0]
    assert "error" in resp, f"Expected error response, got: {resp}"
    assert resp["error"]["code"] == -32602, (
        f"Expected -32602 Invalid params, got code={resp['error']['code']}"
    )


# ---------------------------------------------------------------------------
# Test 4: handle_session does not crash when _meta is a non-dict value;
#         the request still succeeds using the server's fallback correlation id.
# ---------------------------------------------------------------------------

def test_server_non_dict_meta_does_not_crash(tmp_path):
    request_with_bad_meta = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"_meta": "not-a-dict"},
    })
    ws = _MockWS([request_with_bad_meta])
    memory = IncidentMemoryStore()
    logger = TelemetryLogger(tmp_path / "tel.jsonl")

    asyncio.run(handle_session(ws, logger, memory))

    assert ws.sent, "Expected at least one response from handle_session."
    resp = ws.sent[0]
    assert "result" in resp, (
        f"Expected successful initialize response, got: {resp}"
    )
    assert "error" not in resp


# ---------------------------------------------------------------------------
# Test 5: MCPClient.rpc raises RuntimeError with 'RPC timeout' on recv hang.
# ---------------------------------------------------------------------------

def test_rpc_recv_timeout_raises_runtime_error(tmp_path):
    async def _run():
        async def _hang():
            await asyncio.sleep(10.0)
            return "{}"

        mock_ws = MagicMock()
        mock_ws.send = AsyncMock()
        mock_ws.recv = _hang

        client = MCPClient(
            uri="ws://fake",
            telemetry=TelemetryLogger(tmp_path / "tel.jsonl"),
            rpc_timeout_s=0.02,
        )
        client._ws = mock_ws
        client.ctx = RunContext(correlation_id="test-timeout", loop_id="loop-timeout")

        with pytest.raises(RuntimeError, match="RPC timeout"):
            await client.rpc("initialize", {
                "protocolVersion": "2024-11-05",
                "clientInfo": {"name": "test", "version": "0.0"},
            })

    asyncio.run(_run())
