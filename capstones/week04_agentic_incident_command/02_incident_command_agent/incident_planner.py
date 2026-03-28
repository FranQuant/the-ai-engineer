"""
Planner for the Incident Command Agent.

Implements an adaptive FSM that uses observation content to select
the appropriate tool sequence for each OPAL loop, rather than always
returning a fixed hardcoded plan.

MCP-compliant:
- Uses `arguments` (not `input`)
- Adds step metadata (step_id, type)
- Produces observation-driven OPAL plan
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class IncidentPlanner:
    def __init__(self, config: Dict[str, Any]) -> None:
        """Configure planner with model/tooling parameters."""
        self.config = config

    # ------------------------------------------------------------------
    # Core planning
    # ------------------------------------------------------------------
    def plan(
        self,
        observations: Dict[str, Any],
        budget: Any,
    ) -> List[Dict[str, Any]]:
        """Select a tool sequence based on observation content.

        Returns an ordered list of OPAL step dicts:
          [{"step_id": "...", "type": "callTool", "name": "...", "arguments": {...}}, ...]
        """
        obs_text = str(observations).lower()

        # High-severity CPU / memory incidents → full diagnostic path
        if any(kw in obs_text for kw in ["cpu", "memory", "spike", "high"]):
            tool_sequence = [
                "retrieve_runbook",
                "run_diagnostic",
                "summarize_incident",
            ]
        # Deployment / crashloop → runbook + summary (no raw diagnostic)
        elif any(kw in obs_text for kw in ["deploy", "crash", "pod", "fail"]):
            tool_sequence = [
                "retrieve_runbook",
                "summarize_incident",
            ]
        # Default fallback
        else:
            tool_sequence = [
                "retrieve_runbook",
                "run_diagnostic",
                "summarize_incident",
            ]

        logger.info("Plan selected: %s", tool_sequence)

        # Extract alert context from observations to populate step arguments
        alert = observations.get("alerts_latest") or {}
        alert_id = alert.get("id", "ALRT-0001") if isinstance(alert, dict) else "ALRT-0001"
        service = alert.get("service", "staging-api") if isinstance(alert, dict) else "staging-api"
        symptom = alert.get("symptom", "incident") if isinstance(alert, dict) else "incident"

        _step_args: Dict[str, Dict[str, Any]] = {
            "retrieve_runbook": {"query": symptom, "top_k": 2},
            "run_diagnostic": {"command": "kubectl top pod", "host": service},
            "create_incident": {
                "id": "INC-001",
                "title": f"Investigate {symptom}",
                "severity": "medium",
            },
            "add_evidence": {
                "id": "EV-001",
                "content": "Diagnostics and runbook retrieved",
                "source": "system",
            },
            "summarize_incident": {
                "alert_id": alert_id,
                "evidence": list(tool_sequence),
            },
        }

        return [
            {
                "step_id": f"step-{i}",
                "type": "callTool",
                "name": name,
                "arguments": _step_args.get(name, {}),
            }
            for i, name in enumerate(tool_sequence, 1)
        ]
