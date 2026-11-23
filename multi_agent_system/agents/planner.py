"""Planner agent: goal -> validated, structured plan (a DAG of steps)."""
from __future__ import annotations

import json
from typing import Any

from core.llm import LLMClient
from core.state import AgentResult, Status
from core.validation import PlanValidationError, validate_plan
from .base import BaseAgent

PLANNER_PROMPT = """ROLE:PLANNER
You break a goal into an ordered plan of small, checkable sub-tasks.
Reply with ONLY a JSON object (no markdown, no commentary):
{{"assumptions": ["..."],
  "steps": [{{"id": "s1", "description": "...", "agent": "<one of {agents}>",
             "depends_on": [], "expected_output": "...", "success_criterion": "..."}}]}}
Rules: unique step ids; depends_on must reference real step ids; no cycles;
state assumptions explicitly; give every step a concrete success_criterion.
Agents: retriever = gathers information using external APIs; executor = transforms, computes, writes."""


class PlannerAgent(BaseAgent):
    name = "planner"

    def __init__(self, llm: LLMClient, known_agents: set[str], max_plan_retries: int = 2, **kw):
        super().__init__(llm, PLANNER_PROMPT.format(agents=sorted(known_agents)), **kw)
        self.known_agents = known_agents
        self.max_plan_retries = max_plan_retries

    def run(self, task: str, context: dict[str, Any] | None = None) -> AgentResult:
        """task = the user goal. context may hold 'completed' and 'failed' for replanning."""
        context = context or {}
        existing = set(context.get("completed", {}))
        if existing:
            prompt = (f"REPLAN\nGOAL: {task}\n"
                      f"COMPLETED STEPS (do not redo): {json.dumps(context['completed'])[:3000]}\n"
                      f"FAILED STEP: {context.get('failed_step')} -> {context.get('error')}\n"
                      f"Plan ONLY the remaining work. Use new step ids not in {sorted(existing)}; "
                      f"you may depend on completed ids.")
        else:
            prompt = f"GOAL: {task}"
        messages = [{"role": "user", "content": prompt}]
        tokens, last_error = 0, ""

        for attempt in range(self.max_plan_retries + 1):
            resp = self.llm.create(self.system_prompt, messages)
            tokens += resp.usage.get("input_tokens", 0) + resp.usage.get("output_tokens", 0)
            try:
                assumptions, steps = validate_plan(resp.text, self.known_agents, existing)
                return AgentResult(Status.SUCCESS, output=resp.text,
                                   metadata={"steps": steps, "assumptions": assumptions,
                                             "tokens": tokens, "attempts": attempt + 1})
            except PlanValidationError as e:
                last_error = str(e)
                # re-prompt with the exact validation error
                messages += [{"role": "assistant", "content": resp.text or "(empty)"},
                             {"role": "user", "content": f"Your plan was invalid: {last_error}. "
                                                         f"Return corrected JSON only."}]
        return AgentResult(Status.FAILED, error=f"planner could not produce a valid plan: {last_error}",
                           metadata={"tokens": tokens})
