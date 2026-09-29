"""
Loop-control helpers shared by the remote and local Incident Command agents.

Responsibilities:
- Deterministic diagnostic fixtures (each command has a fixed verdict).
- Assess the outcome of one OPAL loop: resolved or unresolved.
- Build the Learn delta note: "Diagnosed <service>, ran <tool/command>, outcome <result>".
- Name the terminal states an incident run can end in.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

STATE_RESOLVED = "resolved"
STATE_ESCALATED = "escalated"

# Commands the planner may run, in the order it tries them.
FIRST_CHECK = "kubectl top pod"
FURTHER_CHECK = "kubectl logs deploy/staging-api --tail=50"
CHECKS = (FIRST_CHECK, FURTHER_CHECK)

# Fixture output per command. Only the "healthy" verdict resolves an incident.
_DIAGNOSTIC_FIXTURES: Dict[str, Dict[str, str]] = {
    FIRST_CHECK: {
        "verdict": "inconclusive",
        "stdout": "node-3 CPU at 88% of limit; no pod restarts; cause not identified.",
    },
    FURTHER_CHECK: {
        "verdict": "degraded",
        "stdout": "Repeated GC pauses; sustained CPU 92% on node-3; root cause needs human review.",
    },
}
_HEALTHY_FIXTURE = {
    "verdict": "healthy",
    "stdout": "All pods healthy; CPU normalized.",
}


def diagnostic_fixture(command: Optional[str]) -> Dict[str, str]:
    """Return the deterministic verdict and stdout for a diagnostic command."""
    return dict(_DIAGNOSTIC_FIXTURES.get(str(command or "").strip(), _HEALTHY_FIXTURE))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _diagnostic_data(results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the data of the last successful run_diagnostic result, if any."""
    found = None
    for item in results:
        step = item.get("step", {})
        result = item.get("result", {})
        if step.get("name") != "run_diagnostic" or not isinstance(result, dict):
            continue
        if result.get("status") == "ok" and isinstance(result.get("data"), dict):
            found = result["data"]
    return found


def assess_results(results: List[Dict[str, Any]], service: str) -> Dict[str, Any]:
    """Classify one loop as resolved or unresolved and describe what was run."""
    diagnostic = _diagnostic_data(results)
    if diagnostic is None:
        executed = [item.get("step", {}).get("name") for item in results]
        return {
            "status": "unresolved",
            "verdict": "no_diagnostic",
            "service": service,
            "ran": executed[-1] if executed else "nothing",
            "command": None,
            "result": "no successful diagnostic",
        }

    verdict = str(diagnostic.get("verdict", "inconclusive"))
    return {
        "status": "resolved" if verdict == "healthy" else "unresolved",
        "verdict": verdict,
        "service": service,
        "ran": "run_diagnostic",
        "command": diagnostic.get("command"),
        "result": " ".join(str(diagnostic.get("stdout", "")).split()),
    }


def build_loop_delta(
    ctx_loop_id: str,
    alert_id: str,
    outcome: Dict[str, Any],
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the memory delta written at the end of a loop (Learn)."""
    ran = outcome.get("command") or outcome.get("ran")
    note = (
        f"Diagnosed {outcome.get('service')}, ran {ran}, "
        f"outcome {outcome.get('verdict')}: {outcome.get('result')}"
    )
    return {
        "kind": "loop_outcome",
        "alert_id": alert_id,
        "timestamp": utc_now_iso(),
        "loop_id": ctx_loop_id,
        "run_id": run_id,
        "command": outcome.get("command"),
        "outcome": outcome.get("status"),
        "verdict": outcome.get("verdict"),
        "note": note,
    }
