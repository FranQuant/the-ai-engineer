"""
Incident Command OPAL orchestrator with guardrails and telemetry.

Aligned with:
- Planner steps using `arguments` instead of `input`
- Standardized envelopes: {"status", "data", "metrics"}
- Basic observation of key memory surfaces
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from config import (
    DEFAULT_BUDGET_DOLLARS,
    DEFAULT_BUDGET_MS,
    DEFAULT_BUDGET_TOKENS,
    DEFAULT_MAX_FAILURES,
    DEFAULT_MAX_LOOPS,
    DEFAULT_MAX_STEPS,
)
from incident_loop import (
    STATE_ESCALATED,
    STATE_RESOLVED,
    assess_results,
    build_loop_delta,
    diagnostic_fixture,
)
from incident_memory import IncidentMemoryStore
from incident_planner import IncidentPlanner
from incident_schemas import tool_cost_tokens
from telemetry import Budget, RunContext, TelemetryEvent, TelemetryLogger, new_correlation_id


class IncidentAgent:
    def __init__(
        self,
        memory: IncidentMemoryStore,
        planner: IncidentPlanner,
        telemetry: TelemetryLogger,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the agent with memory, planner, telemetry, and configuration."""
        self.memory = memory
        self.planner = planner
        self.telemetry = telemetry
        self.config = config or {}
        self.budget = self._new_budget()
        self.max_steps = DEFAULT_MAX_STEPS
        self.max_latency_ms = DEFAULT_BUDGET_MS
        self.max_failures = DEFAULT_MAX_FAILURES
        self.max_loops = DEFAULT_MAX_LOOPS
        self.run_id: Optional[str] = None
        self.last_act: Dict[str, Any] = {"status": "ok", "stop_reason": None}
        self.LOCAL_TOOLS = {
            "retrieve_runbook": self._local_retrieve_runbook,
            "run_diagnostic": self._local_run_diagnostic,
            "summarize_incident": self._local_summarize_incident,
            "create_incident": self._local_create_incident,
            "add_evidence": self._local_add_evidence,
            "append_delta": self._local_append_delta,
        }

    @staticmethod
    def _new_budget() -> Budget:
        """Per-loop budget. cost_dollars is recorded as 0.0 and never enforced."""
        return Budget(
            tokens=DEFAULT_BUDGET_TOKENS,
            ms=DEFAULT_BUDGET_MS,
            dollars=DEFAULT_BUDGET_DOLLARS,
        )

    # ------------------------------------------------------------------
    # OPAL: Observe
    # ------------------------------------------------------------------
    async def observe(self, ctx: RunContext) -> Dict[str, Any]:
        """Collect capabilities and resources needed for planning."""
        self.telemetry.log(
            TelemetryEvent(
                correlation_id=ctx.correlation_id,
                loop_id=ctx.loop_id,
                phase="observe_start",
                method="list_resources",
                status="ok",
                latency_ms=0,
                budget=self.budget,
                payload={},
            )
        )

        observations: Dict[str, Any] = {
            "run_id": self.run_id,
            "resources": self.memory.list_resources(),
            "telemetry_snapshot": self.telemetry.snapshot(),
        }

        try:
            observations["alerts_latest"] = self.memory.get_resource("memory://alerts/latest")
        except Exception:
            observations["alerts_latest"] = None

        try:
            observations["runbooks_index"] = self.memory.get_resource("memory://runbooks/index")
        except Exception:
            observations["runbooks_index"] = None

        try:
            observations["deltas_recent"] = self.memory.get_resource("memory://deltas/recent")
        except Exception:
            observations["deltas_recent"] = None

        missing = any(observations[key] is None for key in ("alerts_latest", "runbooks_index", "deltas_recent"))
        self.telemetry.log(
            TelemetryEvent(
                correlation_id=ctx.correlation_id,
                loop_id=ctx.loop_id,
                phase="observe_end",
                method="list_resources",
                status="error" if missing else "ok",
                latency_ms=0,
                budget=self.budget,
                payload=observations,
            )
        )
        return observations

    # ------------------------------------------------------------------
    # OPAL: Plan
    # ------------------------------------------------------------------
    async def plan(self, ctx: RunContext, observations: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Produce an ordered plan (callTool and memory operations) under budget constraints."""
        self.telemetry.log(
            TelemetryEvent(
                correlation_id=ctx.correlation_id,
                loop_id=ctx.loop_id,
                phase="plan_start",
                method="planner",
                status="ok",
                latency_ms=0,
                budget=self.budget,
                payload={"observations": observations},
            )
        )

        plan = self.planner.plan(observations, self.budget)

        plan_status = "ok"
        if len(plan) > self.max_steps:
            self.telemetry.log(
                TelemetryEvent(
                    correlation_id=ctx.correlation_id,
                    loop_id=ctx.loop_id,
                    phase="plan_guardrail",
                    method="max_steps",
                    status="error",
                    latency_ms=0,
                    budget=self.budget,
                    payload={"reason": "max_steps_exceeded", "allowed": self.max_steps},
                )
            )
            plan = plan[: self.max_steps]
            plan_status = "guardrail_stop"

        self.telemetry.log(
            TelemetryEvent(
                correlation_id=ctx.correlation_id,
                loop_id=ctx.loop_id,
                phase="plan_end",
                method="planner",
                status=plan_status,
                latency_ms=0,
                budget=self.budget,
                payload={"plan": plan},
            )
        )
        return plan

    # ------------------------------------------------------------------
    # OPAL: Act
    # ------------------------------------------------------------------
    async def act(self, ctx: RunContext, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute planned steps via local tools and record results."""
        results: List[Dict[str, Any]] = []
        cumulative_latency = 0
        failure_count = 0
        stop_reason: Optional[str] = None

        self.telemetry.log(
            TelemetryEvent(
                correlation_id=ctx.correlation_id,
                loop_id=ctx.loop_id,
                phase="act_start",
                method="callTool_batch",
                status="ok",
                latency_ms=0,
                budget=self.budget,
                payload={"steps": steps},
            )
        )

        for step in steps:
            if len(results) >= self.max_steps:
                self.telemetry.log(
                    TelemetryEvent(
                        correlation_id=ctx.correlation_id,
                        loop_id=ctx.loop_id,
                        phase="act_guardrail",
                        method="max_steps",
                        status="error",
                        latency_ms=0,
                        budget=self.budget,
                        payload={"reason": "max_steps_exceeded"},
                    )
                )
                stop_reason = "max_steps_exceeded"
                break

            if step.get("type") == "callTool":
                name = step.get("name", "")
                arguments = step.get("arguments", {}) or {}

                tool_fn = self.LOCAL_TOOLS.get(name)
                if not tool_fn:
                    result = {"status": "error", "data": {"error": f"Unknown local tool: {name}"}, "metrics": {"latency_ms": 0}}
                else:
                    result = tool_fn(arguments)
                    result["metrics"]["cost_tokens"] = tool_cost_tokens(name)
                    result["metrics"]["cost_dollars"] = 0.0

                results.append({"step": step, "result": result})

                metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
                latency_ms = int(metrics.get("latency_ms", 0) or 0)
                cumulative_latency += latency_ms
                self.budget.consume(
                    latency_ms=latency_ms,
                    tokens_used=int(metrics.get("cost_tokens", 0) or 0),
                )

                self.telemetry.log(
                    TelemetryEvent(
                        correlation_id=ctx.correlation_id,
                        loop_id=ctx.loop_id,
                        phase="act_step",
                        method=name,
                        status=result.get("status", "ok") if isinstance(result, dict) else "ok",
                        latency_ms=latency_ms,
                        budget=self.budget,
                        payload={"step": step, "result": result},
                    )
                )

                if result.get("status") != "ok":
                    failure_count += 1

                if cumulative_latency > self.max_latency_ms:
                    self.telemetry.log(
                        TelemetryEvent(
                            correlation_id=ctx.correlation_id,
                            loop_id=ctx.loop_id,
                            phase="act_guardrail",
                            method="latency_budget",
                            status="error",
                            latency_ms=latency_ms,
                            budget=self.budget,
                            payload={
                                "reason": "latency_budget_exceeded",
                                "cumulative_ms": cumulative_latency,
                            },
                        )
                    )
                    stop_reason = "ms_budget_exceeded"
                    break

                if self.budget.tokens <= 0:
                    self.telemetry.log(
                        TelemetryEvent(
                            correlation_id=ctx.correlation_id,
                            loop_id=ctx.loop_id,
                            phase="act_guardrail",
                            method="token_budget",
                            status="error",
                            latency_ms=latency_ms,
                            budget=self.budget,
                            payload={"reason": "token_budget_exceeded", "tokens_left": self.budget.tokens},
                        )
                    )
                    stop_reason = "token_budget_exceeded"
                    break

                if failure_count >= self.max_failures:
                    self.telemetry.log(
                        TelemetryEvent(
                            correlation_id=ctx.correlation_id,
                            loop_id=ctx.loop_id,
                            phase="act_guardrail",
                            method="max_failures",
                            status="error",
                            latency_ms=latency_ms,
                            budget=self.budget,
                            payload={
                                "reason": "max_failures_exceeded",
                                "failures": failure_count,
                            },
                        )
                    )
                    stop_reason = "max_failures_exceeded"
                    break

            else:
                results.append({"step": step, "result": {"status": "ok", "data": {}, "metrics": {"latency_ms": 0}}})

        if stop_reason:
            act_status = "guardrail_stop"
        elif failure_count:
            act_status = "error"
        else:
            act_status = "ok"
        self.last_act = {"status": act_status, "stop_reason": stop_reason}

        self.telemetry.log(
            TelemetryEvent(
                correlation_id=ctx.correlation_id,
                loop_id=ctx.loop_id,
                phase="act_end",
                method="callTool_batch",
                status=act_status,
                latency_ms=0,
                budget=self.budget,
                payload={
                    "results": results,
                    "cumulative_latency_ms": cumulative_latency,
                    "failures": failure_count,
                    "stop_reason": stop_reason,
                },
            )
        )
        return results

    # ------------------------------------------------------------------
    # OPAL: Learn
    # ------------------------------------------------------------------
    async def learn(
        self,
        ctx: RunContext,
        observations: Dict[str, Any],
        results: List[Dict[str, Any]],
        outcome: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Write deltas, summaries, and updated state back to memory."""
        alert = observations.get("alerts_latest") if isinstance(observations, dict) else None
        alert = alert if isinstance(alert, dict) else {}
        outcome = outcome or assess_results(results, str(alert.get("service", "unknown")))

        self.telemetry.log(
            TelemetryEvent(
                correlation_id=ctx.correlation_id,
                loop_id=ctx.loop_id,
                phase="learn_start",
                method="memory_write",
                status="ok",
                latency_ms=0,
                budget=self.budget,
                payload={},
            )
        )

        # Structured deltas
        for item in results:
            step = item.get("step", {})
            result = item.get("result", {})
            delta = {
                "action": "completed_step",
                "step_name": step.get("name"),
                "step_id": step.get("step_id"),
                "status": result.get("status"),
            }
            self.memory.write_delta(delta)

        # -----------------------------------------------------------
        # write_plan 
        # -----------------------------------------------------------
        executed_plan = [item.get("step", {}) for item in results]
        self.memory.write_plan(executed_plan)
        # -----------------------------------------------------------

        # Loop-outcome delta: "Diagnosed X, ran Y, outcome Z" (alert_id + ISO timestamp).
        self.memory.write_delta(
            build_loop_delta(ctx.loop_id, str(alert.get("id", "unknown")), outcome, self.run_id)
        )

        learn_result = {"deltas_written": len(results) + 1}

        self.telemetry.log(
            TelemetryEvent(
                correlation_id=ctx.correlation_id,
                loop_id=ctx.loop_id,
                phase="learn_end",
                method="memory_write",
                status="ok",
                latency_ms=0,
                budget=self.budget,
                payload=learn_result,
            )
        )
        return learn_result

    # ------------------------------------------------------------------
    # OPAL: Full loop
    # ------------------------------------------------------------------
    async def run_loop(self, ctx: RunContext) -> Dict[str, Any]:
        self.budget = self._new_budget()
        observations = await self.observe(ctx)
        plan_steps = await self.plan(ctx, observations)
        results = await self.act(ctx, plan_steps)
        alert = observations.get("alerts_latest")
        service = str(alert.get("service", "unknown")) if isinstance(alert, dict) else "unknown"
        outcome = assess_results(results, service)
        learn_result = await self.learn(ctx, observations, results, outcome)
        return {
            "observations": observations,
            "plan": plan_steps,
            "results": results,
            "learn": learn_result,
            "outcome": outcome,
            "act": dict(self.last_act),
        }

    async def run_incident(self) -> Dict[str, Any]:
        """Loop until resolved or escalated (same terminal-state rules as the remote agent)."""
        self.run_id = new_correlation_id()
        loops: List[Dict[str, Any]] = []
        state, reason = STATE_ESCALATED, "max_loops_reached"
        outcomes: List[Dict[str, Any]] = []

        for number in range(1, self.max_loops + 1):
            ctx = RunContext(correlation_id=new_correlation_id(), loop_id=f"loop-{number}")
            result = await self.run_loop(ctx)
            result["ctx"] = {"correlation_id": ctx.correlation_id, "loop_id": ctx.loop_id}
            loops.append(result)

            outcomes.append(result["outcome"])
            if result["act"]["stop_reason"] in ("ms_budget_exceeded", "token_budget_exceeded"):
                state, reason = STATE_ESCALATED, result["act"]["stop_reason"]
                break

            decision = self.planner.decide(outcomes)
            self.telemetry.log(
                TelemetryEvent(
                    correlation_id=ctx.correlation_id,
                    loop_id=ctx.loop_id,
                    phase="plan_decision",
                    method=decision["action"],
                    status="ok",
                    latency_ms=0,
                    budget=self.budget,
                    payload=decision,
                )
            )
            if decision["action"] == "resolve":
                state, reason = STATE_RESOLVED, decision["reason"]
                break
            if decision["action"] == "escalate":
                state, reason = STATE_ESCALATED, decision["reason"]
                break
            if number == self.max_loops:
                self.telemetry.log(
                    TelemetryEvent(
                        correlation_id=ctx.correlation_id,
                        loop_id=ctx.loop_id,
                        phase="loop_guardrail",
                        method="max_loops",
                        status="error",
                        latency_ms=0,
                        budget=self.budget,
                        payload={"reason": "max_loops_reached", "allowed": self.max_loops},
                    )
                )

        last = loops[-1]["ctx"]
        self.telemetry.log(
            TelemetryEvent(
                correlation_id=last["correlation_id"],
                loop_id=last["loop_id"],
                phase="incident_end",
                method="terminal_state",
                status="ok" if state == STATE_RESOLVED else STATE_ESCALATED,
                latency_ms=0,
                budget=self.budget,
                payload={
                    "state": state,
                    "reason": reason,
                    "escalation_reason": reason if state == STATE_ESCALATED else None,
                },
            )
        )
        return {
            "state": state,
            "reason": reason,
            "escalation_reason": reason if state == STATE_ESCALATED else None,
            "run_id": self.run_id,
            "loops": loops,
        }

    # ------------------------------------------------------------------
    # Local deterministic tools
    # ------------------------------------------------------------------

    def _local_retrieve_runbook(self, args: Dict[str, Any]) -> Dict[str, Any]:
        query = str(args.get("query", "")).lower()
        top_k = int(args.get("top_k", 1))
        runbooks = self.memory.get_resource("memory://runbooks/index")

        matches = [
            rb
            for rb in runbooks
            if isinstance(rb, dict)
            and (
                query in rb.get("title", "").lower()
                or any(query in step.lower() for step in rb.get("steps", []))
            )
        ]

        # -----------------------------------------------------------
        # MCP envelope 
        # -----------------------------------------------------------
        return {
            "status": "ok",
            "data": matches[:top_k],
            "metrics": {"latency_ms": 1},
        }

    def _local_run_diagnostic(self, args: Dict[str, Any]) -> Dict[str, Any]:
        fixture = diagnostic_fixture(args.get("command"))
        return {
            "status": "ok",
            "data": {
                "output": fixture["stdout"],
                "stdout": fixture["stdout"],
                "verdict": fixture["verdict"],
                "command": args.get("command"),
                "host": args.get("host"),
            },
            "metrics": {"latency_ms": 1},
        }

    def _local_summarize_incident(self, args: Dict[str, Any]) -> Dict[str, Any]:
        alert_id = str(args.get("alert_id", "ALRT-0001"))
        citations = list(args.get("evidence", [])) or [
            "memory://alerts/latest",
            "memory://runbooks/index",
        ]
        recommended_actions = [
            "Review the runbook guidance for the affected service.",
            "Keep monitoring CPU and restart only if the spike persists.",
        ]
        summary = (
            f"Incident {alert_id}: CPU spike on staging-api was triaged using "
            f"retrieval and diagnostics. Citations: {', '.join(citations)}. "
            f"Recommended actions: {', '.join(recommended_actions)}."
        )
        return {
            "status": "ok",
            "data": {
                "summary": summary,
                "citations": citations,
                "recommended_actions": recommended_actions,
                "args": args,
            },
            "metrics": {"latency_ms": 1},
        }

    def _local_create_incident(self, args: Dict[str, Any]) -> Dict[str, Any]:
        incident = {
            "id": args.get("id", "INC-LOCAL"),
            "title": args.get("title", ""),
            "severity": args.get("severity", "medium"),
        }
        self.memory.update_incident(incident["id"], incident)
        return {
            "status": "ok",
            "data": incident,
            "metrics": {"latency_ms": 1},
        }

    def _local_add_evidence(self, args: Dict[str, Any]) -> Dict[str, Any]:
        evidence = {
            "id": args.get("id"),
            "content": args.get("content"),
            "source": args.get("source"),
        }
        self.memory.write_evidence(evidence)
        return {
            "status": "ok",
            "data": evidence,
            "metrics": {"latency_ms": 1},
        }

    def _local_append_delta(self, args: Dict[str, Any]) -> Dict[str, Any]:
        self.memory.write_delta(args)
        return {
            "status": "ok",
            "data": args,
            "metrics": {"latency_ms": 1},
        }
