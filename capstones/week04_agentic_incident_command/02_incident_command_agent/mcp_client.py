"""
JSON-RPC 2.0 WebSocket client for the Incident MCP server with telemetry.
Now MCP-compliant with:
- Proper content[]/mimeType unwrapping (Fix #7B)
- Strict JSON-RPC error handling
- Request/response validation helpers
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import websockets

from config import (
    DEFAULT_BUDGET_DOLLARS,
    DEFAULT_BUDGET_MS,
    DEFAULT_BUDGET_TOKENS,
    MCP_CLIENT_NAME,
    MCP_CLIENT_VERSION,
    MCP_PROTOCOL_VERSION,
    MCP_SERVER_URI,
    MCP_SUBPROTOCOL,
)
from telemetry import Budget, RunContext, TelemetryEvent, TelemetryLogger


# ---------------------------------------------------------------------------
# Helper functions for MCP envelopes 
# ---------------------------------------------------------------------------

def _unwrap_mcp_result(result: Dict[str, Any]) -> Any:
    """
    MCP-compliant result parser.

    Correct behavior:
    - If the server returns MCP `content[]` blocks, unwrap them.
    - Otherwise, PRESERVE the full envelope (status, data, metrics).
    """

    # MCP content[] shape
    if isinstance(result, dict) and "content" in result:
        content = result.get("content") or []
        if isinstance(content, list) and content:
            item = content[0]
            if item.get("type") == "text":
                return item.get("text")
            if item.get("type") == "json":
                return item.get("data")
        # If content[] exists but can't be interpreted, return envelope intact
        return result

    # (This preserves status + metrics needed for guardrails)
    return result



# ---------------------------------------------------------------------------
# Client implementation
# ---------------------------------------------------------------------------

class MCPClient:
    def __init__(
        self,
        uri: str = MCP_SERVER_URI,
        telemetry: Optional[TelemetryLogger] = None,
        budget: Optional[Budget] = None,
    ) -> None:
        self.uri = uri
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._next_id = 1
        self.telemetry = telemetry
        self.budget = budget or Budget(
            tokens=DEFAULT_BUDGET_TOKENS,
            ms=DEFAULT_BUDGET_MS,
            dollars=DEFAULT_BUDGET_DOLLARS,
        )
        self.ctx: Optional[RunContext] = None

        # Fix 5C
        self.server_info: Optional[Dict[str, Any]] = None
        self.server_protocol: Optional[str] = None

    def set_context(self, ctx: RunContext) -> None:
        """Attach a run context for correlation-id aware telemetry."""
        self.ctx = ctx

    async def connect(self) -> None:
        """Open WebSocket connection using MCP subprotocol."""
        self._ws = await websockets.connect(self.uri, subprotocols=[MCP_SUBPROTOCOL])
        self._next_id = 1

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    # -----------------------------------------------------------------------
    # Core JSON-RPC call
    # -----------------------------------------------------------------------

    async def rpc(self, method: str, params: Dict[str, Any]) -> Any:
        """Send a JSON-RPC 2.0 request and return the unwrapped MCP result."""
        if self._ws is None:
            raise RuntimeError("Client is not connected")

        req_id = self._next_id
        self._next_id += 1

        request = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}

        # Telemetry: send
        if self.telemetry and self.ctx:
            self.telemetry.log(
                TelemetryEvent(
                    correlation_id=self.ctx.correlation_id,
                    loop_id=self.ctx.loop_id,
                    phase="rpc_send",
                    method=method,
                    status="ok",
                    latency_ms=0,
                    budget=self.budget,
                    payload={"request": request},
                )
            )

        await self._ws.send(json.dumps(request))
        raw = await self._ws.recv()

        try:
            response = json.loads(raw)
        except Exception:
            raise RuntimeError(f"Malformed JSON response: {raw!r}")

        # Telemetry: receive
        if self.telemetry and self.ctx:
            self.telemetry.log(
                TelemetryEvent(
                    correlation_id=self.ctx.correlation_id,
                    loop_id=self.ctx.loop_id,
                    phase="rpc_recv",
                    method=method,
                    status="ok" if "error" not in response else "error",
                    latency_ms=0,
                    budget=self.budget,
                    payload={"response": response},
                )
            )

        # JSON-RPC error
        if "error" in response:
            raise RuntimeError(response["error"])

        # Extract result
        result = response.get("result")
        if result is None:
            raise RuntimeError(f"Missing 'result' field in response: {response}")

        # MCP unwrap (Fix #7B)
        return _unwrap_mcp_result(result)

    # -----------------------------------------------------------------------
    # MCP methods
    # -----------------------------------------------------------------------

    async def initialize(self) -> Any:
        params = {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "clientInfo": {"name": MCP_CLIENT_NAME, "version": MCP_CLIENT_VERSION},
        }

        result = await self.rpc("initialize", params)

        # Save server metadata
        if isinstance(result, dict):
            self.server_info = result.get("serverInfo")
            self.server_protocol = result.get("protocolVersion")

        return result

    async def get_resource(self, uri: str, cursor: Optional[str] = None) -> Any:
        params: Dict[str, Any] = {"uri": uri}
        if cursor is not None:
            params["cursor"] = cursor
        return await self.rpc("getResource", params)

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        return await self.rpc("callTool", {"name": name, "arguments": arguments})
