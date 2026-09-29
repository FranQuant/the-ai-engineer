"""
Structured telemetry utilities for the MCP tool harness warm-up.
Aligned with TAE Week-04 Incident Command blueprint and engcode-main.
Provides:
- correlation IDs
- budgets
- run context
- telemetry event model
- JSONL logger (deterministic, replay-friendly)
- timing helper
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple


# ============================================================
# Correlation IDs
# ============================================================

def new_correlation_id() -> str:
    """Return a unique hex correlation ID for each OPAL loop."""
    return uuid.uuid4().hex


# ============================================================
# Budget model (placeholder values; enforced later)
# ============================================================

@dataclass
class Budget:
    tokens: int = 2000       # not enforced in warm-up
    ms: int = 40000          # global per-loop runtime ceiling
    dollars: float = 0.25    # placeholder for LLM cost tracking


# ============================================================
# Run context
# ============================================================

@dataclass
class RunContext:
    """Context for OPAL loop correlation."""
    correlation_id: str
    loop_id: str


# ============================================================
# Telemetry event model
# ============================================================

@dataclass
class TelemetryEvent:
    correlation_id: str
    loop_id: str
    phase: str          # observe | plan | act | learn
    method: str         # initialize | getResource | callTool | plan | memory_write
    status: str         # ok | error
    latency_ms: int
    budget: Budget
    payload: Dict[str, Any]


# ============================================================
# Telemetry logger (JSONL sink)
# ============================================================

class TelemetryLogger:
    """
    Append JSON-lines telemetry for deterministic replay.
    Also prints events to stdout for development-time visibility.
    """

    def __init__(self, sink: Path) -> None:
        self.sink = sink
        self.sink.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: TelemetryEvent) -> None:
        record = asdict(event)
        record["timestamp"] = time.time()
        line = json.dumps(record, ensure_ascii=True)
        with self.sink.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line)


# ============================================================
# Timing helper
# ============================================================

def timed(fn, *args, **kwargs) -> Tuple[int, Any]:
    """
    Measure wall-clock latency in milliseconds and return (latency_ms, result).
    Used for capabilities, resources, and tools.
    """
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    latency_ms = int((time.perf_counter() - start) * 1000)
    return latency_ms, result
