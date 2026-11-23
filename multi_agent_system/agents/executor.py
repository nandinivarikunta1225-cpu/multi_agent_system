"""Executor agent: performs a step (transform / calculate / draft) from dependency outputs."""
from __future__ import annotations

from typing import Any

from core.llm import LLMClient
from core.state import AgentResult
from core.tools import ToolRegistry
from .base import BaseAgent

EXECUTOR_PROMPT = """ROLE:EXECUTOR
You carry out exactly one step. Use only the CONTEXT given (outputs of the steps you depend on).
Do not search for new information; that is the retriever's job. Be concise, structured, and
keep source URLs attached to any fact you reuse. Meet the stated SUCCESS CRITERION."""


class ExecutorAgent(BaseAgent):
    name = "executor"

    def __init__(self, llm: LLMClient, tools: ToolRegistry | None = None, **kw):
        super().__init__(llm, EXECUTOR_PROMPT, tools=tools, **kw)

    def run(self, task: str, context: dict[str, Any] | None = None) -> AgentResult:
        messages = [{"role": "user", "content": self.format_task(task, context)}]
        return self.run_with_tools(messages)
