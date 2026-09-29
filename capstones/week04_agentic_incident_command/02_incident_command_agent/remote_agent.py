"""
Remote Incident Agent that drives planning and action through an MCP client with guardrails and telemetry.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from config import (
    DEFAULT_BUDGET_DOLLARS,
    DEFAULT_BUDGET_MS,
    DEFAULT_BUDGET_TOKENS,
    DEFAULT_MAX_FAILURES,
    DEFAULT_MAX_LOOPS,
    DEFAULT_MAX_STEPS,
)
from incident_loop import STATE_ESCALATED, STATE_RESOLVED, assess_results, build_loop_delta
from telemetry import Budget, RunContext, TelemetryEvent, TelemetryLogger, new_correlation_id

_BUDGET_STOPS = ("ms_budget_exceeded", "token_budget_exceeded")


class RemoteIncidentAgent:
    def __init__(self, mcp_client, planner, telemetry: TelemetryLogger) -> None:
        self.client = mcp_client
        self.planner = planner
        self.telemetry = telemetry
        self.budget = self._new_budget()
        self.max_steps = DEFAULT_MAX_STEPS
        self.max_latency_ms = DEFAULT_BUDGET_MS
        self.max_failures = DEFAULT_MAX_FAILURES
        self.max_loops = DEFAULT_MAX_LOOPS
        self.run_id: Optional[str] = None
        self.last_act: Dict[str, Any] = {"status": "ok", "stop_reason": None}

    @staticmethod
    def _new_budget() -> Budget:
        """Per-loop budget. cost_dollars is recorded as 0.0 and never enforced."""
        return Budget(
            tokens=DEFAULT_BUDGET_TOKENS,
            ms=DEFAULT_BUDGET_MS,
            dollars=DEFAULT_BUDGET_DOLLARS,
        )

    def _set_phase(self, phase: Optional[str]) -> None:
        setter = getattr(self.client, "set_phase", None)
        if setter is not None:
            setter(phase)

    def _charge(self, result: Any) -> int:
        """Debit the loop budget with a tool result's reported cost; return its latency."""
        metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
        latency_ms = int(metrics.get("latency_ms", 0) or 0)
        self.budget.consume(
            latency_ms=latency_ms,
            tokens_used=int(metrics.get("cost_tokens", 0) or 0),
            dollars_used=float(metrics.get("cost_dollars", 0.0) or 0.0),
        )
        return latency_ms

    @staticmethod
    def _summary_evidence(
        summary_step: Dict[str, Any],
        prior_results: List[Dict[str, Any]],
    ) -> List[str]:
        """Build deterministic citations from successful work in this execution."""
        arguments = summary_step.get("arguments", {}) or {}
        alert_id = str(arguments.get("alert_id", "")).strip()
        evidence = [f"memory://alerts/latest#{alert_id}"] if alert_id else []

        for item in prior_results:
            step = item.get("step", {})
            result = item.get("result", {})
            if not isinstance(result, dict) or result.get("status") != "ok":
                continue

            data = result.get("data", {})
            if step.get("name") == "retrieve_runbook" and isinstance(data, dict):
                runbooks = data.get("results", [])
                if isinstance(runbooks, list):
                    for runbook in runbooks:
                        runbook_id = runbook.get("id") if isinstance(runbook, dict) else None
                        if runbook_id:
                            evidence.append(f"memory://runbooks/index#{runbook_id}")

            if step.get("name") == "run_diagnostic" and isinstance(data, dict):
                captured = data.get("stdout") or data.get("stderr") or data.get("output") or "<no output>"
                captured = " ".join(str(captured).split())[:200]
                diagnostic = {
                    "command": data.get("command") or step.get("arguments", {}).get("command"),
                    "host": data.get("host") or step.get("arguments", {}).get("host"),
                    "result": captured,
                    "step": step.get("step_id"),
                }
                evidence.append(
                    "diagnostic:" + json.dumps(diagnostic, sort_keys=True, separators=(",", ":"))
                )

        return list(dict.fromkeys(evidence))

    # ------------------------------------------------------------------
    # OPAL: Observe
    # ------------------------------------------------------------------
    async def observe(self, ctx: RunContext) -> Dict[str, Any]:
        """Fetch capabilities and resources (alerts, runbooks, memory deltas, telemetry) from the MCP server."""
        self._set_phase("observe")
        self._log(ctx, "observe_start", "mcp_observe_batch", "ok", {})

        capabilities = await self.client.initialize()
        alerts_latest = await self.client.get_resource("memory://alerts/latest")
        runbooks_index = await self.client.get_resource("memory://runbooks/index")
        deltas_recent = await self.client.get_resource("memory://deltas/recent")

        status = "ok"
        try:
            telemetry_snapshot = await self.client.get_resource("memory://telemetry/snapshot")
        except Exception as exc:
            telemetry_snapshot = {"error": str(exc)}
            status = "error"

        observations = {
            "run_id": self.run_id,
            "capabilities": capabilities,
            "alerts_latest": alerts_latest,
            "runbooks_index": runbooks_index,
            "deltas_recent": deltas_recent,
            "telemetry_snapshot": telemetry_snapshot,
        }

        self._log(ctx, "observe_end", "mcp_observe_batch", status, observations)
        return observations

    # ------------------------------------------------------------------
    # OPAL: Plan
    # ------------------------------------------------------------------
    async def plan(self, ctx: RunContext, observations: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run planner locally under budget constraints."""
        self._set_phase("plan")
        self._log(ctx, "plan_start", "planner", "ok", {"observations": observations})

        plan = self.planner.plan(observations, self.budget)

        status = "ok"
        if len(plan) > self.max_steps:
            self._log(
                ctx, "plan_guardrail", "max_steps", "error",
                {"reason": "max_steps_exceeded", "allowed": self.max_steps},
            )
            plan = plan[: self.max_steps]
            status = "guardrail_stop"

        self._log(ctx, "plan_end", "planner", status, {"plan": plan})
        return plan

    # ------------------------------------------------------------------
    # OPAL: Act
    # ------------------------------------------------------------------
    async def act(self, ctx: RunContext, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute callTool steps via MCP client with guardrails (steps, ms, tokens, failures)."""
        self._set_phase("act")
        results: List[Dict[str, Any]] = []
        cumulative_latency = 0
        failure_count = 0
        stop_reason: Optional[str] = None

        self._log(ctx, "act_start", "callTool_batch", "ok", {"steps": steps})

        def guardrail(method: str, reason: str, latency_ms: int, **extra: Any) -> None:
            self._log(
                ctx, "act_guardrail", method, "error", {"reason": reason, **extra},
                latency_ms=latency_ms,
            )

        for step in steps:

            # Guardrail: max steps
            if len(results) >= self.max_steps:
                guardrail("max_steps", "max_steps_exceeded", 0)
                stop_reason = "max_steps_exceeded"
                break

            if step.get("type") == "callTool":

                name = step.get("name", "")
                arguments = step.get("arguments", {}) or {}

                if name == "summarize_incident":
                    arguments = dict(arguments)
                    arguments["evidence"] = self._summary_evidence(step, results)
                    step["arguments"] = arguments

                try:
                    result = await self.client.call_tool(name, arguments)
                    status = result.get("status") if isinstance(result, dict) else "ok"
                except Exception as exc:
                    result = {
                        "status": "error",
                        "error": str(exc),
                        "metrics": {"latency_ms": 0},
                    }
                    status = "error"

                results.append({"step": step, "result": result})

                latency_ms = self._charge(result)
                cumulative_latency += latency_ms

                # Guardrail: ms budget
                if self.budget.ms <= 0:
                    guardrail("ms_budget", "ms_budget_exceeded", latency_ms, cumulative_ms=cumulative_latency)
                    stop_reason = "ms_budget_exceeded"
                    break

                # Guardrail: token budget (actual per-tool cost_tokens)
                if self.budget.tokens <= 0:
                    guardrail("token_budget", "token_budget_exceeded", latency_ms, tokens_left=self.budget.tokens)
                    stop_reason = "token_budget_exceeded"
                    break

                # Guardrail: failure budget
                if status != "ok":
                    failure_count += 1

                if failure_count >= self.max_failures:
                    guardrail("max_failures", "max_failures_exceeded", latency_ms, failures=failure_count)
                    stop_reason = "max_failures_exceeded"
                    break

            else:
                # Non-callTool step
                results.append({
                    "step": step,
                    "result": {"status": "ok", "data": {}, "metrics": {"latency_ms": 0}},
                })

        if stop_reason:
            act_status = "guardrail_stop"
        elif failure_count:
            act_status = "error"
        else:
            act_status = "ok"
        self.last_act = {"status": act_status, "stop_reason": stop_reason}

        self._log(
            ctx, "act_end", "callTool_batch", act_status,
            {
                "results": results,
                "cumulative_latency_ms": cumulative_latency,
                "failures": failure_count,
                "stop_reason": stop_reason,
            },
        )

        return results

    # ------------------------------------------------------------------
    # OPAL: Learn
    # ------------------------------------------------------------------
    async def learn(
        self,
        ctx: RunContext,
        observations: Optional[Dict[str, Any]] = None,
        plan: Optional[List[Dict[str, Any]]] = None,
        action_result: Optional[List[Dict[str, Any]]] = None,
        outcome: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Fetch recent deltas, persist the current plan, then write this loop's outcome."""
        self._set_phase("learn")
        self._log(ctx, "learn_start", "getResource_batch", "ok", {})

        deltas = await self.client.get_resource("memory://deltas/recent")

        plan_to_persist = list(plan or [])
        self._charge(await self.client.call_tool("write_plan", {"plan": plan_to_persist}))
        current_plan = await self.client.get_resource("memory://plans/current")

        # Write the outcome of this loop back to memory: "Diagnosed X, ran Y, outcome Z".
        observations = observations or {}
        alert = observations.get("alerts_latest") if isinstance(observations, dict) else None
        alert = alert if isinstance(alert, dict) else {}
        outcome = outcome or assess_results(action_result or [], str(alert.get("service", "unknown")))
        delta = build_loop_delta(ctx.loop_id, str(alert.get("id", "unknown")), outcome, self.run_id)

        status = "ok"
        try:
            self._charge(await self.client.call_tool("append_memory_delta", {"delta": delta}))
        except Exception as exc:
            # Non-fatal: the loop still completes, but make the failed Learn write
            # visible in the trace instead of silently discarding it.
            status = "error"
            self._log(
                ctx, "learn_guardrail", "append_memory_delta", "error",
                {"reason": "append_memory_delta_failed", "error": str(exc)},
            )

        learn_result = {
            "deltas": deltas,
            "plan": current_plan,
            "delta_written": delta,
            "plan_written": plan_to_persist,
        }

        self._log(ctx, "learn_end", "getResource_batch", status, learn_result)
        return learn_result

    # ------------------------------------------------------------------
    # Full OPAL loop
    # ------------------------------------------------------------------
    async def run_loop(self, ctx: RunContext) -> Dict[str, Any]:
        self.budget = self._new_budget()
        self.client.set_context(ctx)
        setter = getattr(self.client, "set_budget", None)
        if setter is not None:
            setter(self.budget)

        observations = await self.observe(ctx)
        plan_steps = await self.plan(ctx, observations)
        results = await self.act(ctx, plan_steps)
        alert = observations.get("alerts_latest")
        service = str(alert.get("service", "unknown")) if isinstance(alert, dict) else "unknown"
        outcome = assess_results(results, service)
        learn_result = await self.learn(
            ctx,
            observations=observations,
            plan=plan_steps,
            action_result=results,
            outcome=outcome,
        )

        return {
            "observations": observations,
            "plan": plan_steps,
            "results": results,
            "learn": learn_result,
            "outcome": outcome,
            "act": dict(self.last_act),
            "budget_remaining": {
                "tokens": self.budget.tokens,
                "ms": self.budget.ms,
                "dollars": self.budget.dollars,
            },
        }

    # ------------------------------------------------------------------
    # Incident run: loop until resolved or escalated
    # ------------------------------------------------------------------
    async def run_incident(self) -> Dict[str, Any]:
        """Run OPAL loops (loop-1, loop-2, ...) until a terminal state: resolved or escalated.

        Each loop gets its own correlation ID. The run ends `resolved` when a diagnostic
        comes back healthy, and `escalated` when the loop budget (ms/tokens) is exhausted or
        max_loops is reached without a resolution.
        """
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
            if result["act"]["stop_reason"] in _BUDGET_STOPS:
                state, reason = STATE_ESCALATED, result["act"]["stop_reason"]
                break

            decision = self.planner.decide(outcomes)
            self._log(ctx, "plan_decision", decision["action"], "ok", decision)
            if decision["action"] == "resolve":
                state, reason = STATE_RESOLVED, decision["reason"]
                break
            if decision["action"] == "escalate":
                state, reason = STATE_ESCALATED, decision["reason"]
                break
            if number == self.max_loops:
                self._log(
                    ctx, "loop_guardrail", "max_loops", "error",
                    {"reason": "max_loops_reached", "allowed": self.max_loops},
                )

        last = loops[-1]["ctx"]
        ctx = RunContext(correlation_id=last["correlation_id"], loop_id=last["loop_id"])
        memory_snapshot = await self._memory_snapshot(ctx)
        self._log(
            ctx, "incident_end", "terminal_state",
            "ok" if state == STATE_RESOLVED else STATE_ESCALATED,
            {
                "state": state,
                "reason": reason,
                "escalation_reason": reason if state == STATE_ESCALATED else None,
                "loops": [
                    {"loop_id": item["ctx"]["loop_id"], "correlation_id": item["ctx"]["correlation_id"],
                     "outcome": item["outcome"]}
                    for item in loops
                ],
            },
        )
        return {
            "state": state,
            "reason": reason,
            "escalation_reason": reason if state == STATE_ESCALATED else None,
            "run_id": self.run_id,
            "loops": loops,
            "memory_snapshot": memory_snapshot,
        }

    async def _memory_snapshot(self, ctx: RunContext) -> Dict[str, Any]:
        """Read the server's memory resources at the end of the run (written to memory_snapshot.json)."""
        self.client.set_context(ctx)
        self._set_phase("learn")
        return {
            "alert": await self.client.get_resource("memory://alerts/latest"),
            "deltas": await self.client.get_resource("memory://deltas/recent"),
            "plan": await self.client.get_resource("memory://plans/current"),
            "telemetry": await self.client.get_resource("memory://telemetry/snapshot"),
        }

    # ------------------------------------------------------------------
    # Telemetry helper
    # ------------------------------------------------------------------
    def _log(
        self,
        ctx: RunContext,
        phase: str,
        method: str,
        status: str,
        payload: Dict[str, Any],
        latency_ms: int = 0,
    ) -> None:
        self.telemetry.log(
            TelemetryEvent(
                correlation_id=ctx.correlation_id,
                loop_id=ctx.loop_id,
                phase=phase,
                method=method,
                status=status,
                latency_ms=latency_ms,
                budget=self.budget,
                payload=payload,
            )
        )
