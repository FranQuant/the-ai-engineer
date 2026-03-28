"""
JSON schemas for tools and resources used by the Incident Command Agent.

Responsibilities:
- Define input/output schemas for all MCP tools.
- Define resource shapes for memory:// URIs.
- Provide registry helpers for server capabilities payload.
"""

from __future__ import annotations

from typing import Dict, List


def get_tool_schemas() -> Dict[str, Dict[str, object]]:
    """Return mapping of tool name to JSON schema definitions (used for validation)."""
    return {
        "retrieve_runbook": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keyword to match against runbook titles and steps",
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "Maximum number of runbook snippets to return (default 3)",
                },
            },
            "required": ["query"],
        },
        "run_diagnostic": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell or kubectl command to execute in the sandbox",
                },
                "host": {
                    "type": "string",
                    "description": "Target host or cluster context for the diagnostic",
                },
            },
            "required": ["command", "host"],
        },
        "summarize_incident": {
            "type": "object",
            "properties": {
                "alert_id": {
                    "type": "string",
                    "description": "Unique identifier of the alert being summarized",
                },
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "Evidence strings or tool result references to include in the summary",
                },
            },
            "required": ["alert_id"],
        },
        "create_incident": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Unique incident identifier (e.g. INC-001)",
                },
                "title": {
                    "type": "string",
                    "description": "Short human-readable incident title",
                },
                "severity": {
                    "type": "string",
                    "description": "Severity level: low | medium | high | critical",
                },
            },
            "required": ["id", "title"],
        },
        "add_evidence": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Optional unique identifier for the evidence item",
                },
                "content": {
                    "type": "string",
                    "description": "Evidence payload text or structured data",
                },
                "source": {
                    "type": "string",
                    "description": "Origin of the evidence (e.g. diagnostic, runbook, manual)",
                },
            },
            "required": ["content"],
        },
        "append_delta": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Short label for the action recorded in this delta",
                },
                "details": {
                    "type": "object",
                    "description": "Arbitrary key-value details associated with the action",
                },
            },
            "required": ["action"],
        },
        "write_plan": {
            "type": "object",
            "properties": {
                "plan": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Ordered list of OPAL step descriptors to persist",
                },
            },
            "required": ["plan"],
        },
        "append_memory_delta": {
            "type": "object",
            "properties": {
                "delta": {
                    "type": "object",
                    "description": "Delta payload capturing loop outcome to append to memory://deltas/recent",
                },
            },
            "required": ["delta"],
        },
    }


def get_resource_schemas() -> Dict[str, Dict[str, object]]:
    """Return mapping of memory:// resource URIs to JSON schema definitions."""
    return {
        "memory://incidents/{id}": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "title": {"type": "string"},
                "severity": {"type": "string"},
            },
            "required": ["id"],
        },
        "memory://evidence/{id}": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "content": {"type": "string"},
                "source": {"type": "string"},
            },
            "required": ["id", "content"],
        },
        "memory://deltas/recent": {
            "type": "object",
            "properties": {
                "items": {"type": "array"},
            },
            "required": ["items"],
        },
        "memory://plans/current": {
            "type": "object",
            "properties": {
                "plan": {"type": "array"},
            },
            "required": ["plan"],
        },
        "memory://alerts/latest": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "service": {"type": "string"},
                "symptom": {"type": "string"},
                "severity": {"type": "string"},
            },
            "required": ["id", "service", "symptom", "severity"],
        },
        "memory://runbooks/index": {
            "type": "array",
            "items": {"type": "object"},
        },
        "memory://memory/deltas": {
            "type": "object",
            "properties": {
                "items": {"type": "array"},
            },
            "required": ["items"],
        },
    }


# Human-readable descriptions and cost hints per tool
_TOOL_META: Dict[str, Dict[str, object]] = {
    "retrieve_runbook": {
        "description": (
            "Search runbooks by keyword and return the top-k most relevant snippets. "
            "Used during Observe or Act to surface applicable remediation procedures."
        ),
        "latency_hint_ms": 5,
        "cost_hint_tokens": 10,
    },
    "run_diagnostic": {
        "description": (
            "Execute a sandboxed diagnostic command on the specified host and return stdout/stderr. "
            "Supports kubectl, log-fetch, and metric queries."
        ),
        "latency_hint_ms": 7,
        "cost_hint_tokens": 15,
    },
    "summarize_incident": {
        "description": (
            "Produce a 3-4 sentence triage summary for the given alert with recommended actions. "
            "Incorporates the evidence strings passed in the request."
        ),
        "latency_hint_ms": 6,
        "cost_hint_tokens": 20,
    },
    "create_incident": {
        "description": (
            "Create or update an incident record in memory://incidents/{id}. "
            "Returns the stored incident payload."
        ),
        "latency_hint_ms": 4,
        "cost_hint_tokens": 5,
    },
    "add_evidence": {
        "description": (
            "Append an evidence item to the memory store for later correlation and summarization."
        ),
        "latency_hint_ms": 3,
        "cost_hint_tokens": 5,
    },
    "append_delta": {
        "description": (
            "Append a structured action delta to memory://deltas/recent for OPAL Learn-phase replay."
        ),
        "latency_hint_ms": 2,
        "cost_hint_tokens": 3,
    },
    "write_plan": {
        "description": (
            "Persist the current OPAL plan to memory://plans/current for inspection and replay."
        ),
        "latency_hint_ms": 1,
        "cost_hint_tokens": 3,
    },
    "append_memory_delta": {
        "description": (
            "Append a structured memory delta capturing loop outcome to memory://deltas/recent."
        ),
        "latency_hint_ms": 2,
        "cost_hint_tokens": 3,
    },
}


def tool_descriptions() -> List[Dict[str, object]]:
    """Return tool descriptors combining name, description, inputSchema, and cost hints."""
    schemas = get_tool_schemas()
    return [
        {
            "name": name,
            "description": _TOOL_META[name]["description"],
            "inputSchema": schema,
            "latency_hint_ms": _TOOL_META[name]["latency_hint_ms"],
            "cost_hint_tokens": _TOOL_META[name]["cost_hint_tokens"],
        }
        for name, schema in schemas.items()
    ]


def resource_descriptions() -> List[Dict[str, object]]:
    """Return resource descriptors for server capabilities."""
    schemas = get_resource_schemas()
    return [
        {"uri": uri, "schema": schema, "description": f"Resource for {uri}"}
        for uri, schema in schemas.items()
    ]
