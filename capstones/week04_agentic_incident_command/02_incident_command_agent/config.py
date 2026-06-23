"""
Shared Week 4 configuration for the incident command capstone.

Keep this module intentionally small so the server, client, and agent
entrypoints all read the same defaults without introducing extra layers.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
TELEMETRY_SINK = ARTIFACTS_DIR / "telemetry.jsonl"

MCP_PROTOCOL_VERSION = "2024-11-05"
MCP_SUBPROTOCOL = "mcp"
MCP_SERVER_HOST = "127.0.0.1"
MCP_SERVER_PORT = 8765
MCP_SERVER_URI = f"ws://{MCP_SERVER_HOST}:{MCP_SERVER_PORT}/mcp"
MCP_SERVER_NAME = "week04-agentic-incident-command"
MCP_SERVER_VERSION = "1.0.0"
MCP_CLIENT_NAME = "week04-agentic-incident-client"
MCP_CLIENT_VERSION = "1.0.0"

DEFAULT_BUDGET_TOKENS = 2000
DEFAULT_BUDGET_MS = 150
DEFAULT_BUDGET_DOLLARS = 0.0
DEFAULT_MAX_STEPS = 5
DEFAULT_MAX_FAILURES = 2


def fresh_telemetry_sink() -> Path:
    """Return TELEMETRY_SINK after archiving any existing file with a datestamp."""
    if TELEMETRY_SINK.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        TELEMETRY_SINK.rename(
            TELEMETRY_SINK.with_name(f"telemetry_{stamp}.jsonl")
        )
    return TELEMETRY_SINK
