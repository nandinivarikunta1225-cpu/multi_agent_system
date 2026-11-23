"""BaseAgent: the OOP backbone. Every agent obeys the same contract."""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Any

from core.llm import LLMClient
from core.state import AgentResult, Status
from core.tools import ToolRegistry


class BaseAgent(ABC):
    name: str = "base"

    def __init__(self, llm: LLMClient, system_prompt: str, tools: ToolRegistry | None = None,
                 max_iters: int = 6):
        self.llm = llm
        self.system_prompt = system_prompt
        self.tools = tools or ToolRegistry()
        self.max_iters = max_iters

    @abstractmethod
    def run(self, task: str, context: dict[str, Any] | None = None) -> AgentResult:
        """Do one task. Must return an AgentResult and never raise into the orchestrator."""

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def format_task(task: str, context: dict[str, Any] | None) -> str:
        """Render the task plus ONLY the slice of context this agent needs."""
        context = context or {}
        parts = [f"TASK: {task}"]
        if context.get("expected_output"):
            parts.append(f"EXPECTED OUTPUT: {context['expected_output']}")
        if context.get("success_criterion"):
            parts.append(f"SUCCESS CRITERION: {context['success_criterion']}")
        deps = context.get("dependencies") or {}
        body = "\n\n".join(f"[{sid}]\n{out}" for sid, out in deps.items()) or "(none)"
        parts.append(f"CONTEXT:\n{body}")
        return "\n".join(parts)

    @staticmethod
    def _collect_evidence(result: dict) -> list[dict]:
        return [{"source": r["source"], "snippet": r.get("snippet", "")}
                for r in result.get("results", []) if isinstance(r, dict) and "source" in r]

    def run_with_tools(self, messages: list[dict]) -> AgentResult:
        """The agent loop: call model -> run requested tools -> feed results back -> repeat.

        The LLM never executes code; this loop does, and it has a hard iteration cap.
        """
        start = time.time()
        tokens = 0
        evidence: list[dict] = []
        tool_log: list[dict] = []
        tool_defs = self.tools.definitions() or None

        for _ in range(self.max_iters):
            resp = self.llm.create(self.system_prompt, messages, tools=tool_defs)
            tokens += resp.usage.get("input_tokens", 0) + resp.usage.get("output_tokens", 0)

            if resp.stop_reason != "tool_use" or not resp.tool_calls:
                return AgentResult(
                    status=Status.SUCCESS, output=resp.text.strip(),
                    metadata={"tokens": tokens, "latency_s": round(time.time() - start, 3),
                              "evidence": evidence, "tool_calls": tool_log})

            messages.append({"role": "assistant", "content": resp.raw_content})
            results_block = []
            for call in resp.tool_calls:
                result = self.tools.execute(call.name, call.args)   # allow-list + validation inside
                tool_log.append({"tool": call.name, "args": call.args, "ok": "error" not in result})
                evidence.extend(self._collect_evidence(result))
                results_block.append({"type": "tool_result", "tool_use_id": call.id,
                                      "content": json.dumps(result)})
            messages.append({"role": "user", "content": results_block})

        return AgentResult(status=Status.FAILED, error="max iterations reached",
                           metadata={"tokens": tokens, "latency_s": round(time.time() - start, 3),
                                     "tool_calls": tool_log})
