"""Retriever agent: gathers evidence via external-API tools, with provenance."""
from __future__ import annotations

from typing import Any

from core.llm import LLMClient
from core.state import AgentResult
from core.tools import ToolRegistry
from .base import BaseAgent

RETRIEVER_PROMPT = """ROLE:RETRIEVER
You gather information needed for a task using ONLY the tools provided.
Do not write the final deliverable and do not speculate. Return the evidence you found as concise
bullets, each ending with its source URL. If tools return nothing useful, say so plainly."""


class RetrieverAgent(BaseAgent):
    name = "retriever"

    def __init__(self, llm: LLMClient, tools: ToolRegistry, **kw):
        super().__init__(llm, RETRIEVER_PROMPT, tools=tools, **kw)

    def run(self, task: str, context: dict[str, Any] | None = None) -> AgentResult:
        messages = [{"role": "user", "content": self.format_task(task, context)}]
        return self.run_with_tools(messages)   # evidence/provenance lands in result.metadata
