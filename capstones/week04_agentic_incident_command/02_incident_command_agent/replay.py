"""
Telemetry replay runner for the Incident Command Agent.

Usage:
    python replay.py <path-to-telemetry-jsonl>
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from telemetry import Budget, TelemetryEvent


@dataclass
class ReplayEvent:
    raw: Dict[str, Any]
    phase: str
    method: str
    status: str
    latency_ms: int
    payload: Dict[str, Any]
    timestamp: float


def load_events(path: Path) -> List[ReplayEvent]:
    """Load telemetry events from a JSONL file and normalize."""
    events: List[ReplayEvent] = []
    with path.open("r", encoding="utf-8") as fp:
        for idx, line in enumerate(fp):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            events.append(
                ReplayEvent(
                    raw=raw,
                    phase=raw.get("phase", "unknown"),
                    method=raw.get("method", "unknown"),
                    status=raw.get("status", "unknown"),
                    latency_ms=int(raw.get("latency_ms", 0)),
                    payload=raw.get("payload", {}) or {},
                    timestamp=float(raw.get("timestamp", idx)),  # fall back to order if missing
                )
            )
    # Preserve file order; if timestamps exist, the numeric value will still maintain ordering with idx fallback.
    events.sort(key=lambda e: e.timestamp)
    return events


class ReplayRunner:
    def __init__(self, events: Iterable[ReplayEvent]) -> None:
        self.events = list(events)

    def replay(self) -> None:
        """Print a human-readable reconstruction of the OPAL loop."""
        for idx, ev in enumerate(self.events, start=1):
            header = f"[{idx:03d}] {ev.phase} :: {ev.method} :: {ev.status} :: {ev.latency_ms}ms"
            print(header)
            payload = ev.payload or {}
            # Show key subsets for readability.
            if "request" in payload:
                req = payload.get("request", {})
                print(f"      request: {json.dumps(req)}")
            if "response" in payload:
                res = payload.get("response", {})
                print(f"      response: {json.dumps(res)}")
            if "errors" in payload:
                print(f"      errors: {json.dumps(payload.get('errors'))}")
            if "plan" in payload:
                print(f"      plan: {json.dumps(payload.get('plan'))}")
            if "steps" in payload:
                print(f"      steps: {json.dumps(payload.get('steps'))}")
            if "results" in payload:
                print(f"      results: {json.dumps(payload.get('results'))}")
            if "delta" in payload:
                print(f"      delta: {json.dumps(payload.get('delta'))}")

    @staticmethod
    def simulate_tool_envelope(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Optional deterministic simulation of tool results."""
        return {
            "status": "ok",
            "data": {"tool": name, "echo": arguments},
            "metrics": {"latency_ms": 1, "cost_tokens": 0},
        }


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Replay telemetry events.")
    parser.add_argument("path", type=Path, help="Path to telemetry JSONL file")
    args = parser.parse_args(argv)

    events = load_events(args.path)
    if not events:
        print("No events found.")
        return

    runner = ReplayRunner(events)
    runner.replay()


if __name__ == "__main__":
    main()
