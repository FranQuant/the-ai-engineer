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

    @staticmethod
    def _alert_context(observations: Dict[str, Any]) -> Dict[str, str]:
        alert = observations.get("alerts_latest") if isinstance(observations, dict) else {}
        alert = alert if isinstance(alert, dict) else {}
        return {
            "symptom": str(alert.get("symptom", "")).strip().lower(),
            "severity": str(alert.get("severity", "")).strip().lower(),
            "service": str(alert.get("service", "staging-api")).strip() or "staging-api",
            "alert_id": str(alert.get("id", "ALRT-0001")).strip() or "ALRT-0001",
        }

    @staticmethod
    def _classify_incident(symptom: str, severity: str) -> str:
        if any(token in symptom for token in ("crash", "crashloop", "deploy", "rollout", "pod", "fail", "restart")):
            return "crash_or_deploy"
        if any(token in symptom for token in ("cpu", "memory", "spike")) or severity in {"high", "critical"}:
            return "cpu_spike"
        return "fallback"

    @staticmethod
    def _budget_is_tight(budget: Any) -> bool:
        if budget is None:
            return False
        tokens = getattr(budget, "tokens", None)
        ms = getattr(budget, "ms", None)
        if isinstance(budget, dict):
            tokens = budget.get("tokens", tokens)
            ms = budget.get("ms", ms)
        try:
            tokens = int(tokens)
        except (TypeError, ValueError):
            tokens = 0
        try:
            ms = int(ms)
        except (TypeError, ValueError):
            ms = 0
        return tokens <= 20 or ms <= 20

    @staticmethod
    def _runbook_query(incident_class: str) -> str:
        if incident_class == "crash_or_deploy":
            return "crash"
        return "cpu"

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
        alert = self._alert_context(observations)
        incident_class = self._classify_incident(alert["symptom"], alert["severity"])
        budget_tight = self._budget_is_tight(budget)

        # Keep the planner simple: default paths stay the same, and tight budget trims the longest path.
        if budget_tight and incident_class in {"cpu_spike", "fallback"}:
            tool_sequence = ["retrieve_runbook", "summarize_incident"]
        elif incident_class == "crash_or_deploy":
            tool_sequence = ["retrieve_runbook", "summarize_incident"]
        else:
            tool_sequence = ["retrieve_runbook", "run_diagnostic", "summarize_incident"]

        logger.info("Plan selected: %s", tool_sequence)

        runbook_query = self._runbook_query(incident_class)
        symptom = alert["symptom"] or "incident"
        service = alert["service"]
        alert_id = alert["alert_id"]

        _step_args: Dict[str, Dict[str, Any]] = {
            "retrieve_runbook": {"query": runbook_query, "top_k": 2},
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
