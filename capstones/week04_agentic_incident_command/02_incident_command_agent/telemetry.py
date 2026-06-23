"""
Structured telemetry utilities for the Incident Command Agent.

Responsibilities:
- Generate correlation IDs and loop IDs.
- Track budgets (tokens, milliseconds, dollars).
- Emit telemetry events for observe/plan/act/learn phases.
- Provide JSONL logger compatible with warm-up harness patterns.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


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


# ---------------------------------------------------------------------------
# Logger that applies budget consumption 
# ---------------------------------------------------------------------------

class TelemetryLogger:
    def __init__(self, sink: Path) -> None:
        """Initialize telemetry logger with JSONL sink."""
        self.sink = sink

    def log(self, event: TelemetryEvent) -> None:
        """
        Record telemetry event and consume budget.

        Budget consumption per event:
        - tokens: 1 token per event
        - ms: use event.latency_ms
        - dollars: unchanged (0.0)
        """
        event.budget.consume(
            latency_ms=event.latency_ms,
            tokens_used=1,
            dollars_used=0.0,
        )

        # Serialize event
        record = asdict(event)
        record["timestamp"] = time.time()
        line = json.dumps(record)

        # Echo to console
        print(line)

        # Ensure directory exists
        self.sink.parent.mkdir(parents=True, exist_ok=True)

        # Append to JSONL file
        with self.sink.open("a", encoding="utf-8") as fp:
            fp.write(line + "\n")


# ---------------------------------------------------------------------------
# Helper for local timing
# ---------------------------------------------------------------------------

def timed(fn, *args, **kwargs) -> Tuple[int, Any]:
    """Measure latency and return (latency_ms, result)."""
    start = time.monotonic()
    result = fn(*args, **kwargs)
    latency_ms = int((time.monotonic() - start) * 1000)
    return latency_ms, result
