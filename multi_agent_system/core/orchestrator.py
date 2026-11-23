"""Orchestrator: owns the loop, shared state, guardrails, and retry/replan/abort decisions."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from agents.base import BaseAgent
from agents.planner import PlannerAgent
from .state import AgentResult, PlanStep, SharedState, Status
from .validation import topological_order

log = logging.getLogger("orchestrator")


@dataclass
class Config:
    max_steps: int = 20              # total step attempts across the run
    max_retries_per_step: int = 2
    max_replans: int = 1
    timeout_s: float = 300.0
    token_budget: int = 200_000
    backoff_s: float = 0.0           # sleep between retries (0 for tests)


@dataclass
class RunReport:
    status: str                      # "success" | "partial" | "failed"
    answer: str
    reason: str
    state: SharedState


class Orchestrator:
    def __init__(self, planner: PlannerAgent, agents: dict[str, BaseAgent],
                 synthesizer: BaseAgent, config: Config | None = None):
        self.planner = planner
        self.agents = agents
        self.synthesizer = synthesizer
        self.cfg = config or Config()

    # ------------------------------------------------------------------ main loop
    def run(self, goal: str) -> RunReport:
        state = SharedState(goal=goal)
        t0 = time.time()

        plan_res = self.planner.run(goal)
        state.tokens_used += plan_res.metadata.get("tokens", 0)
        if plan_res.status != Status.SUCCESS:
            return RunReport("failed", "", plan_res.error, state)
        state.plan = plan_res.metadata["steps"]
        state.assumptions = plan_res.metadata["assumptions"]

        while True:
            reason = self._budget_exceeded(state, t0)
            if reason:
                return self._finish(state, "partial", reason)

            step = self._next_ready(state)
            if step is None:
                pending = [s for s in state.plan if s.id not in state.results]
                if not pending:
                    break                                   # all steps done
                return self._finish(state, "partial", "remaining steps are blocked by failed dependencies")

            ok = self._execute_with_retries(step, state)
            if ok:
                continue

            # step failed after retries: replan the remainder, or abort with partial results
            if state.replans < self.cfg.max_replans and self._replan(state, step):
                continue
            return self._finish(state, "partial", f"step '{step.id}' failed: {state.failed.get(step.id)}")

        return self._finish(state, "success", "all steps completed")

    # ------------------------------------------------------------------ pieces
    def _budget_exceeded(self, state: SharedState, t0: float) -> str:
        if state.total_steps >= self.cfg.max_steps:
            return f"max steps ({self.cfg.max_steps}) reached"
        if time.time() - t0 > self.cfg.timeout_s:
            return "overall timeout reached"
        if state.tokens_used >= self.cfg.token_budget:
            return "token budget exhausted"
        return ""

    def _next_ready(self, state: SharedState) -> PlanStep | None:
        """First step (in topological order) whose dependencies all succeeded."""
        order = topological_order(
            [PlanStep(s.id, "", "", [d for d in s.depends_on if d in {p.id for p in state.plan}]) for s in state.plan])
        by_id = {s.id: s for s in state.plan}
        for sid in order:
            step = by_id[sid]
            if sid in state.results or sid in state.failed:
                continue
            if all(d in state.results for d in step.depends_on):
                return step
        return None

    def _context_for(self, step: PlanStep, state: SharedState) -> dict[str, Any]:
        """Executor/retriever see only their step + its dependencies' outputs, never full history."""
        return {
            "expected_output": step.expected_output,
            "success_criterion": step.success_criterion,
            "dependencies": {d: state.results[d].output for d in step.depends_on if d in state.results},
        }

    @staticmethod
    def _output_ok(result: AgentResult) -> str:
        """Check output before marking done (stops bad data propagating). Returns error or ''."""
        if not result.output.strip():
            return "empty output"
        return ""

    def _execute_with_retries(self, step: PlanStep, state: SharedState) -> bool:
        agent = self.agents[step.agent]
        ctx = self._context_for(step, state)
        error = ""
        for attempt in range(1, self.cfg.max_retries_per_step + 2):
            if state.total_steps >= self.cfg.max_steps:
                error = "max steps reached"
                break
            state.total_steps += 1
            t = time.time()
            try:
                result = agent.run(step.description, ctx)
            except Exception as e:  # agents shouldn't raise, but never let one crash the loop
                result = AgentResult(Status.FAILED, error=f"{type(e).__name__}: {e}")
            state.tokens_used += result.metadata.get("tokens", 0)

            problem = result.error if result.status != Status.SUCCESS else self._output_ok(result)
            state.history.append({
                "step": step.id, "agent": step.agent, "attempt": attempt,
                "input": step.description[:80], "tool_calls": result.metadata.get("tool_calls", []),
                "latency_s": round(time.time() - t, 3), "status": "ok" if not problem else "error",
                "error": problem})
            log.info("step=%s agent=%s attempt=%d status=%s", step.id, step.agent, attempt,
                     "ok" if not problem else problem)

            if not problem:
                state.results[step.id] = result
                return True
            error = problem
            if result.status == Status.FAILED and result.metadata.get("fatal"):
                break
            if self.cfg.backoff_s:
                time.sleep(self.cfg.backoff_s * attempt)
        state.failed[step.id] = error
        return False

    def _replan(self, state: SharedState, failed_step: PlanStep) -> bool:
        state.replans += 1
        completed = {sid: r.output for sid, r in state.results.items()}
        res = self.planner.run(state.goal, {"completed": completed, "failed_step": failed_step.id,
                                            "error": state.failed.get(failed_step.id, "")})
        state.tokens_used += res.metadata.get("tokens", 0)
        if res.status != Status.SUCCESS:
            return False
        # keep completed steps, drop everything unfinished (incl. the failed one), add new plan
        state.plan = [s for s in state.plan if s.id in state.results] + res.metadata["steps"]
        state.failed.pop(failed_step.id, None)
        state.history.append({"event": "replan", "after_step": failed_step.id,
                              "new_steps": [s.id for s in res.metadata["steps"]]})
        return True

    def _finish(self, state: SharedState, status: str, reason: str) -> RunReport:
        """Final synthesis from whatever completed (so partial runs still return something useful)."""
        if not state.results:
            return RunReport("failed", "", reason, state)
        ctx = {"expected_output": "A clear final answer to the goal, noting any gaps",
               "success_criterion": "Directly addresses the goal using only completed step outputs",
               "dependencies": {sid: r.output for sid, r in state.results.items()}}
        task = f"Write the final answer for the goal: {state.goal}"
        if status != "success":
            task += f" (the run was incomplete: {reason}; say so explicitly)"
        try:
            res = self.synthesizer.run(task, ctx)
        except Exception as e:
            res = AgentResult(Status.FAILED, error=str(e))
        state.tokens_used += res.metadata.get("tokens", 0)
        answer = res.output if res.status == Status.SUCCESS else \
            "\n\n".join(r.output for r in state.results.values())
        return RunReport(status, answer, reason, state)
