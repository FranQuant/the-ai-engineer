"""
CLI entrypoint for the Incident Command Agent with replay support.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Optional

from config import TELEMETRY_SINK
from incident_agent import IncidentAgent
from incident_memory import IncidentMemoryStore
from incident_planner import IncidentPlanner
from replay import load_events, ReplayRunner
from telemetry import RunContext, TelemetryLogger, new_correlation_id


async def run_agent() -> None:
    memory = IncidentMemoryStore()
    planner = IncidentPlanner(config={})
    telemetry = TelemetryLogger(TELEMETRY_SINK)
    agent = IncidentAgent(memory, planner, telemetry)

    ctx = RunContext(correlation_id=new_correlation_id(), loop_id="loop-1")
    summary = await agent.run_loop(ctx)
    print(summary)


def run_replay(path: Path) -> None:
    events = load_events(path)
    runner = ReplayRunner(events)
    runner.replay()


async def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Incident Command Agent CLI")
    parser.add_argument("--replay", type=Path, help="Path to telemetry JSONL file to replay")
    args = parser.parse_args(argv)

    if args.replay:
        run_replay(args.replay)
    else:
        await run_agent()


if __name__ == "__main__":
    asyncio.run(main())
