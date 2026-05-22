"""
Remote Incident Agent that drives planning and action through an MCP client with guardrails and telemetry.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from config import (
    DEFAULT_BUDGET_DOLLARS,
    DEFAULT_BUDGET_MS,
    DEFAULT_BUDGET_TOKENS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_STEPS,
)
from telemetry import Budget, RunContext, TelemetryEvent, TelemetryLogger


class RemoteIncidentAgent:
    def __init__(self, mcp_client, planner, telemetry: TelemetryLogger) -> None:
        self.client = mcp_client
        self.planner = planner
        self.telemetry = telemetry
        self.budget = Budget(
            tokens=DEFAULT_BUDGET_TOKENS,
            ms=DEFAULT_BUDGET_MS,
            dollars=DEFAULT_BUDGET_DOLLARS,
        )
        self.max_steps = DEFAULT_MAX_STEPS
        self.max_latency_ms = DEFAULT_BUDGET_MS
        self.max_retries = DEFAULT_MAX_RETRIES

    # ------------------------------------------------------------------
    # OPAL: Observe
    # ------------------------------------------------------------------
    async def observe(self, ctx: RunContext) -> Dict[str, Any]:
        """Fetch capabilities/resources from the MCP server."""
        self.telemetry.log(
            TelemetryEvent(
                correlation_id=ctx.correlation_id,
                loop_id=ctx.loop_id,
                phase="observe_start",
                method="mcp_observe_batch",
                status="ok",
                latency_ms=0,
                budget=self.budget,
                payload={},
            )
        )

        capabilities = await self.client.initialize()
        alerts_latest = await self.client.get_resource("memory://alerts/latest")
        runbooks_index = await self.client.get_resource("memory://runbooks/index")
        deltas_recent = await self.client.get_resource("memory://deltas/recent")

        observations = {
            "capabilities": capabilities,
            "alerts_latest": alerts_latest,
            "runbooks_index": runbooks_index,
            "deltas_recent": deltas_recent,
        }

        self.telemetry.log(
            TelemetryEvent(
                correlation_id=ctx.correlation_id,
                loop_id=ctx.loop_id,
                phase="observe_end",
                method="mcp_observe_batch",
                status="ok",
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
        """Run planner locally under budget constraints."""
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

        self.telemetry.log(
            TelemetryEvent(
                correlation_id=ctx.correlation_id,
                loop_id=ctx.loop_id,
                phase="plan_end",
                method="planner",
                status="ok",
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
        """Execute callTool steps via MCP client with guardrails."""
        results: List[Dict[str, Any]] = []
        cumulative_latency = 0
        failure_count = 0

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

            # Guardrail: max steps
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
                break

            if step.get("type") == "callTool":

                name = step.get("name", "")
                arguments = step.get("arguments", {}) or {}   # <-- FIXED

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

                metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
                latency_ms = int(metrics.get("latency_ms", 0) or 0)
                cumulative_latency += latency_ms
                self.budget.consume(latency_ms=latency_ms)

                # Guardrail: ms budget
                if self.budget.ms <= 0:
                    self.telemetry.log(
                        TelemetryEvent(
                            correlation_id=ctx.correlation_id,
                            loop_id=ctx.loop_id,
                            phase="act_guardrail",
                            method="ms_budget",
                            status="error",
                            latency_ms=latency_ms,
                            budget=self.budget,
                            payload={
                                "reason": "ms_budget_exceeded",
                                "cumulative_ms": cumulative_latency,
                            },
                        )
                    )
                    break

                # Guardrail: retries
                if status != "ok":
                    failure_count += 1

                if failure_count > self.max_retries:
                    self.telemetry.log(
                        TelemetryEvent(
                            correlation_id=ctx.correlation_id,
                            loop_id=ctx.loop_id,
                            phase="act_guardrail",
                            method="max_retries",
                            status="error",
                            latency_ms=latency_ms,
                            budget=self.budget,
                            payload={
                                "reason": "max_retries_exceeded",
                                "failures": failure_count,
                            },
                        )
                    )
                    break

            else:
                # Non-callTool step 
                results.append({
                    "step": step,
                    "result": {"status": "ok", "data": {}, "metrics": {"latency_ms": 0}},
                })

        self.telemetry.log(
            TelemetryEvent(
                correlation_id=ctx.correlation_id,
                loop_id=ctx.loop_id,
                phase="act_end",
                method="callTool_batch",
                status="ok",
                latency_ms=0,
                budget=self.budget,
                payload={"results": results, "cumulative_latency_ms": cumulative_latency},
            )
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
    ) -> Dict[str, Any]:
        """Fetch recent deltas, persist the current plan, then write this loop's outcome."""
        self.telemetry.log(
            TelemetryEvent(
                correlation_id=ctx.correlation_id,
                loop_id=ctx.loop_id,
                phase="learn_start",
                method="getResource_batch",
                status="ok",
                latency_ms=0,
                budget=self.budget,
                payload={},
            )
        )

        deltas = await self.client.get_resource("memory://deltas/recent")

        plan_to_persist = list(plan or [])
        await self.client.call_tool("write_plan", {"plan": plan_to_persist})
        current_plan = await self.client.get_resource("memory://plans/current")

        # Write the outcome of this loop back to memory
        delta = {
            "loop_id": ctx.loop_id,
            "timestamp": time.time(),
            "observation_summary": str(observations)[:200] if observations else "",
            "plan": str(plan)[:200] if plan else "",
            "outcome": str(action_result)[:200] if action_result else "",
        }
        try:
            await self.client.call_tool("append_memory_delta", {"delta": delta})
        except Exception:
            pass  # non-fatal: telemetry continues even if write fails

        learn_result = {
            "deltas": deltas,
            "plan": current_plan,
            "delta_written": delta,
            "plan_written": plan_to_persist,
        }

        self.telemetry.log(
            TelemetryEvent(
                correlation_id=ctx.correlation_id,
                loop_id=ctx.loop_id,
                phase="learn_end",
                method="getResource_batch",
                status="ok",
                latency_ms=0,
                budget=self.budget,
                payload=learn_result,
            )
        )

        return learn_result

    # ------------------------------------------------------------------
    # Full OPAL loop
    # ------------------------------------------------------------------
    async def run_loop(self, ctx: RunContext) -> Dict[str, Any]:
        self.client.set_context(ctx)

        observations = await self.observe(ctx)
        plan_steps = await self.plan(ctx, observations)
        results = await self.act(ctx, plan_steps)
        learn_result = await self.learn(
            ctx,
            observations=observations,
            plan=plan_steps,
            action_result=results,
        )

        return {
            "observations": observations,
            "plan": plan_steps,
            "results": results,
            "learn": learn_result,
        }
