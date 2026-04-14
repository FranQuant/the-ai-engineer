# ======================================================================
# MCP server for the Incident Command Agent (Week 04)
# ======================================================================

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List

import websockets

from incident_memory import IncidentMemoryStore
from config import (
    DEFAULT_BUDGET_DOLLARS,
    DEFAULT_BUDGET_MS,
    DEFAULT_BUDGET_TOKENS,
    MCP_PROTOCOL_VERSION,
    MCP_SERVER_HOST,
    MCP_SERVER_NAME,
    MCP_SERVER_PORT,
    MCP_SERVER_URI,
    MCP_SERVER_VERSION,
    MCP_SUBPROTOCOL,
    TELEMETRY_SINK,
)
from incident_schemas import get_tool_schemas, resource_descriptions, tool_descriptions
from telemetry import (
    Budget,
    RunContext,
    TelemetryEvent,
    TelemetryLogger,
    new_correlation_id,
    timed,
)

# ---------------------------------------------------------------------------
# Canned fixtures
# ---------------------------------------------------------------------------

RUNBOOKS: List[Dict[str, Any]] = [
    {
        "id": "rb-101",
        "title": "High CPU playbook",
        "service": "staging-api",
        "steps": [
            "Check pod CPU across nodes",
            "Capture logs before restart",
            "Restart service if CPU > 90% for 5 minutes",
        ],
    },
    {
        "id": "rb-202",
        "title": "Crashloop restart",
        "service": "staging-api",
        "steps": [
            "Describe pods",
            "Gather events",
            "Redeploy with rollback flag",
        ],
    },
]

ALERT = {
    "id": "ALRT-0001",
    "service": "staging-api",
    "symptom": "CPU spike on node-3",
    "severity": "high",
    "detected_at": "2025-11-23T09:00:00Z",
}

RESOURCE_FIXTURES: Dict[str, Any] = {
    "memory://alerts/latest": ALERT,
    "memory://runbooks/index": RUNBOOKS,
}

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _type_matches(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True


def validate_arguments(schema: Dict[str, Any], arguments: Dict[str, Any]):
    errors = {}

    for field in schema.get("required", []):
        if field not in arguments:
            errors[field] = "Missing required field"

    properties = schema.get("properties", {})
    for name, value in arguments.items():
        if name not in properties:
            continue

        expected_type = properties[name].get("type")
        if expected_type and not _type_matches(value, expected_type):
            errors[name] = f"Expected {expected_type}"

        if expected_type == "integer":
            minimum = properties[name].get("minimum")
            maximum = properties[name].get("maximum")
            if minimum is not None and value < minimum:
                errors[name] = f"Must be >= {minimum}"
            if maximum is not None and value > maximum:
                errors[name] = f"Must be <= {maximum}"

    return (len(errors) == 0, errors)


# ---------------------------------------------------------------------------
# Unified envelope 
# ---------------------------------------------------------------------------

def _envelope(data: Dict[str, Any], latency_ms: int, cost_tokens: int = 10):
    return {
        "status": "ok",
        "data": data,
        "metrics": {
            "latency_ms": latency_ms,
            "cost_tokens": cost_tokens,
        },
    }


# ---------------------------------------------------------------------------
# Tool handlers 
# ---------------------------------------------------------------------------

def tool_retrieve_runbook(arguments: Dict[str, Any]):
    query = str(arguments.get("query", "")).lower()
    top_k = int(arguments.get("top_k", 3))

    hits = [
        rb for rb in RUNBOOKS
        if query in rb["title"].lower()
        or any(query in step.lower() for step in rb["steps"])
    ]

    return _envelope({"results": hits[:top_k]}, latency_ms=5)


def tool_run_diagnostic(arguments: Dict[str, Any]):
    data = {
        "command": arguments.get("command"),
        "host": arguments.get("host"),
        "stdout": "All pods healthy; CPU normalized.",
        "stderr": "",
    }
    return _envelope(data, latency_ms=7)


def tool_summarize_incident(arguments: Dict[str, Any], memory: IncidentMemoryStore):
    alert_id = arguments.get("alert_id") or ALERT["id"]
    evidence = arguments.get("evidence", [])

    summary = (
        f"Incident {alert_id}: CPU spikes observed on staging-api. "
        "Diagnostics show pods healthy and CPU normalized. "
        "Recommend restart if sustained > 90% for 5 minutes. "
        "Capture logs before restart; monitor for recurrence."
    )

    delta = {
        "alert_id": alert_id,
        "action": "summarize_incident",
        "summary": summary,
        "evidence": evidence,
    }
    memory.append_delta(delta)

    return _envelope({"summary": summary, "citations": evidence}, latency_ms=6)


# Write OPAL plan into memory://plans/current
def tool_write_plan(arguments: Dict[str, Any], memory: IncidentMemoryStore):
    plan = arguments.get("plan", [])
    memory.write_plan(plan)
    return _envelope({"written": True, "plan_length": len(plan)}, latency_ms=1)


def tool_append_memory_delta(arguments: Dict[str, Any], memory: IncidentMemoryStore):
    """Append a structured loop-outcome delta to memory://deltas/recent."""
    delta = arguments.get("delta", arguments)
    result = memory.append_delta(delta)
    return _envelope({"appended": True, "delta": result}, latency_ms=2)


# ---------------------------------------------------------------------------
# TOOL_DISPATCH
# ---------------------------------------------------------------------------

TOOL_DISPATCH = {
    "retrieve_runbook": tool_retrieve_runbook,
    "run_diagnostic": tool_run_diagnostic,
    "summarize_incident": tool_summarize_incident,
    "create_incident": None,          # handled dynamically
    "add_evidence": None,             # handled dynamically
    "append_delta": None,             # handled dynamically
    "write_plan": tool_write_plan,
    "append_memory_delta": None,      # handled dynamically (requires memory)
}


SERVER_BUDGET = Budget(
    tokens=DEFAULT_BUDGET_TOKENS,
    ms=DEFAULT_BUDGET_MS,
    dollars=DEFAULT_BUDGET_DOLLARS,
)


# ---------------------------------------------------------------------------
# Resource helpers
# ---------------------------------------------------------------------------

def capabilities_payload(memory: IncidentMemoryStore):
    del memory
    return {
        "protocolVersion": MCP_PROTOCOL_VERSION,
        "serverInfo": {
            "name": MCP_SERVER_NAME,
            "version": MCP_SERVER_VERSION,
        },
        "capabilities": {"sampling": {"available": False}},
        "tools": tool_descriptions(),
        "resources": resource_descriptions(),
    }


def get_resource(memory: IncidentMemoryStore, uri: str, cursor=None):
    if uri in RESOURCE_FIXTURES:
        return RESOURCE_FIXTURES[uri]

    if uri in ("memory://memory/deltas", "memory://deltas", "memory://deltas/recent"):
        return memory.get_resource("memory://deltas/recent")

    return memory.get_resource(uri, cursor)


def call_tool(memory, name, arguments):
    
    if name == "write_plan":
        return tool_write_plan(arguments, memory)

    if name in ("summarize_incident",):
        return tool_summarize_incident(arguments, memory)

    if name == "create_incident":
        incident_id = arguments.get("id", "INC-001")
        result = memory.update_incident(incident_id, arguments)
        return _envelope(result, latency_ms=4)

    if name == "add_evidence":
        result = memory.write_evidence(arguments)
        return _envelope(result, latency_ms=3)

    if name == "append_delta":
        result = memory.append_delta(arguments)
        return _envelope(result, latency_ms=2)

    if name == "append_memory_delta":
        return tool_append_memory_delta(arguments, memory)

    # Normal tools
    if name in TOOL_DISPATCH and TOOL_DISPATCH[name] is not None:
        return TOOL_DISPATCH[name](arguments)

    raise ValueError(f"Unknown tool '{name}'")


# ---------------------------------------------------------------------------
# JSON-RPC session — latency + budget 
# ---------------------------------------------------------------------------

def _validation_error_response(req_id, details):
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32602, "message": "Invalid params", "data": details},
    }


async def handle_session(ws, logger, memory):
    ctx = RunContext(correlation_id=new_correlation_id(), loop_id="loop-1")
    tool_schemas = get_tool_schemas()

    async for raw in ws:

        try:
            request = json.loads(raw)
        except Exception:
            await ws.send(json.dumps({
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            }))
            continue

        if not isinstance(request, dict):
            await ws.send(json.dumps({
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32600, "message": "Invalid Request"},
            }))
            continue

        if request.get("jsonrpc") != "2.0":
            await ws.send(json.dumps({
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {"code": -32600, "message": "Invalid Request: jsonrpc must be '2.0'"},
            }))
            continue

        if "method" not in request:
            await ws.send(json.dumps({
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {"code": -32600, "message": "Invalid Request: missing method"},
            }))
            continue

        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {}) or {}

        if req_id is None:
            continue  # notification

        phase = "observe" if method in ("initialize", "getResource") else "act"
        status = "ok"

        # timing
        start_time = time.perf_counter()

        try:
            if method == "initialize":
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": capabilities_payload(memory),
                }

            elif method == "getResource":
                uri = params.get("uri")
                cursor = params.get("cursor")
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": get_resource(memory, uri, cursor),
                }

            elif method == "callTool":
                name = params.get("name")
                arguments = params.get("arguments", {}) or {}

                schema = tool_schemas.get(name)
                if schema:
                    valid, errors = validate_arguments(schema, arguments)
                    if not valid:
                        status = "error"
                        response = _validation_error_response(req_id, errors)
                        await ws.send(json.dumps(response))
                        continue

                latency_ms, result = timed(call_tool, memory, name, arguments)

                SERVER_BUDGET.tokens -= 10
                SERVER_BUDGET.ms -= latency_ms

                response = {"jsonrpc": "2.0", "id": req_id, "result": result}

            else:
                status = "error"
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": "Method not found"},
                }

        except Exception as exc:
            status = "error"
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32603, "message": str(exc)},
            }

        full_latency_ms = int((time.perf_counter() - start_time) * 1000)

        logger.log(
            TelemetryEvent(
                correlation_id=ctx.correlation_id,
                loop_id=ctx.loop_id,
                phase=phase,
                method=method,
                status=status,
                latency_ms=full_latency_ms,
                budget=SERVER_BUDGET,
                payload={"request": request, "response": response},
            )
        )

        await ws.send(json.dumps(response))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

async def main():
    memory = IncidentMemoryStore(
        {
            "incidents": {"INC-000": {"id": "INC-000", "title": "Seed incident", "severity": "low"}},
            "evidence": {"EV-000": {"id": "EV-000", "content": "Seed evidence", "source": "seed"}},
            "deltas": [],
            "plans": {},
        }
    )

    logger = TelemetryLogger(TELEMETRY_SINK)

    async def handler(ws):
        await handle_session(ws, logger, memory)

    server = await websockets.serve(
        handler,
        MCP_SERVER_HOST,
        MCP_SERVER_PORT,
        subprotocols=[MCP_SUBPROTOCOL],
    )
    print(f"MCP server listening on {MCP_SERVER_URI}")
    await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
