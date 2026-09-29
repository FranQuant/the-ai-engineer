"""
Structured telemetry utilities for the Incident Command Agent.

Responsibilities:
- Generate correlation IDs and loop IDs.
- Track budgets (tokens, milliseconds, dollars); tools and the agent debit them, the logger does not.
- Emit telemetry events for observe/plan/act/learn phases.
- Provide JSONL logger compatible with warm-up harness patterns.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def new_correlation_id() -> str:
    """Return a unique correlation ID for sessions."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Budget with consumption support 
# ---------------------------------------------------------------------------

@dataclass
class Budget:
    tokens: int
    ms: int
    dollars: float

    def consume(self, latency_ms: int = 0, tokens_used: int = 0, dollars_used: float = 0.0) -> None:
        """
        Decrement available budget after an action.

        Safe-subtraction: budgets will not go below zero.
        """
        self.tokens = max(0, self.tokens - tokens_used)
        self.ms = max(0, self.ms - latency_ms)
        self.dollars = max(0.0, self.dollars - dollars_used)


@dataclass
class RunContext:
    correlation_id: str
    loop_id: str


@dataclass
class TelemetryEvent:
    correlation_id: str
    loop_id: str
    phase: str
    method: str
    status: str
    latency_ms: int
    budget: Budget
    payload: Dict[str, Any]
    # Who owns the budget object in this event: "agent" (per-loop, also used by the
    # client) or "server" (session budget enforced by the MCP server).
    budget_owner: str = "agent"


# ---------------------------------------------------------------------------
# Logger that applies budget consumption 
# ---------------------------------------------------------------------------

class TelemetryLogger:
    def __init__(self, sink: Path) -> None:
        """Initialize telemetry logger with JSONL sink."""
        self.sink = sink
        self._recent: List[Dict[str, str]] = []
        self._by_phase: Dict[str, int] = {}
        self._by_loop: Dict[str, int] = {}

    def log(self, event: TelemetryEvent) -> None:
        """
        Record a telemetry event.

        The logger only records. Budgets are debited by whoever performs the work
        (the agent for tool cost and latency, the server for its session budget),
        so an event never changes the budget it reports.
        """
        record = asdict(event)
        record["timestamp"] = time.time()
        line = json.dumps(record)

        self._by_phase[event.phase] = self._by_phase.get(event.phase, 0) + 1
        self._by_loop[event.loop_id] = self._by_loop.get(event.loop_id, 0) + 1
        self._recent.append(
            {"loop_id": event.loop_id, "phase": event.phase, "method": event.method, "status": event.status}
        )
        del self._recent[:-5]

        # Echo to console
        print(line)

        # Ensure directory exists
        self.sink.parent.mkdir(parents=True, exist_ok=True)

        # Append to JSONL file
        with self.sink.open("a", encoding="utf-8") as fp:
            fp.write(line + "\n")

    def snapshot(self) -> Dict[str, Any]:
        """Deterministic summary of what this logger has recorded so far (memory://telemetry/snapshot)."""
        return {
            "total_events": sum(self._by_phase.values()),
            "by_phase": dict(sorted(self._by_phase.items())),
            "by_loop": dict(sorted(self._by_loop.items())),
            "last_events": list(self._recent),
        }


# ---------------------------------------------------------------------------
# Helper for local timing
# ---------------------------------------------------------------------------

def timed(fn, *args, **kwargs) -> Tuple[int, Any]:
    """Measure latency and return (latency_ms, result)."""
    start = time.monotonic()
    result = fn(*args, **kwargs)
    latency_ms = int((time.monotonic() - start) * 1000)
    return latency_ms, result
