"""
pytest tests for incident command agent tool handlers.

Imports handlers directly (not via the WebSocket server) so tests run
without a live MCP server or event loop.
"""

import sys
from pathlib import Path

# Ensure local modules are importable regardless of where pytest is invoked
sys.path.insert(0, str(Path(__file__).parent))

import pytest
from mcp_server import tool_retrieve_runbook, tool_run_diagnostic, tool_summarize_incident
from incident_memory import IncidentMemoryStore


# ---------------------------------------------------------------------------
# Thin wrappers matching the spec's handle_* naming convention
# ---------------------------------------------------------------------------

def handle_retrieve_runbook(arguments):
    return tool_retrieve_runbook(arguments)


def handle_run_diagnostic(arguments):
    return tool_run_diagnostic(arguments)


def handle_summarize_incident(arguments):
    """Wrap tool with a fresh memory store (server normally injects this)."""
    memory = IncidentMemoryStore()
    return tool_summarize_incident(arguments, memory)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_retrieve_runbook_returns_ok():
    result = handle_retrieve_runbook({"query": "CPU spike", "top_k": 2})
    assert result["status"] == "ok"
    assert "data" in result and "metrics" in result


def test_retrieve_runbook_top_k_respected():
    result = handle_retrieve_runbook({"query": "cpu", "top_k": 1})
    assert result["status"] == "ok"
    assert len(result["data"]["results"]) <= 1


def test_run_diagnostic_returns_ok():
    result = handle_run_diagnostic(
        {"command": "kubectl get pods", "host": "staging-api"}
    )
    assert result["status"] == "ok"
    assert "latency_ms" in result["metrics"]


def test_run_diagnostic_echo_command():
    result = handle_run_diagnostic(
        {"command": "kubectl top pod", "host": "prod-api"}
    )
    assert result["status"] == "ok"
    assert result["data"]["command"] == "kubectl top pod"
    assert result["data"]["host"] == "prod-api"


def test_summarize_incident_returns_ok():
    result = handle_summarize_incident(
        {"alert_id": "ALRT-TEST", "evidence": ["CPU at 92%"]}
    )
    assert result["status"] == "ok"
    assert len(result["data"]["summary"]) > 10


def test_summarize_incident_contains_alert_id():
    result = handle_summarize_incident(
        {"alert_id": "ALRT-XYZ", "evidence": []}
    )
    assert result["status"] == "ok"
    assert "ALRT-XYZ" in result["data"]["summary"]


def test_all_handlers_return_metrics_envelope():
    """All tool results must carry the standard {status, data, metrics} envelope."""
    handlers_and_args = [
        (handle_retrieve_runbook, {"query": "crash", "top_k": 3}),
        (handle_run_diagnostic, {"command": "kubectl logs", "host": "node-1"}),
        (handle_summarize_incident, {"alert_id": "ALRT-001", "evidence": []}),
    ]
    for fn, args in handlers_and_args:
        result = fn(args)
        assert result["status"] == "ok", f"{fn.__name__} status not ok"
        assert "data" in result, f"{fn.__name__} missing data"
        assert "metrics" in result, f"{fn.__name__} missing metrics"
        assert "latency_ms" in result["metrics"], f"{fn.__name__} missing latency_ms"
