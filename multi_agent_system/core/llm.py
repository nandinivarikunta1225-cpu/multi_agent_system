"""LLM client abstraction: real (Anthropic), scripted (tests) and demo (offline)."""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class LLMResponse:
    text: str = ""
    stop_reason: str = "end_turn"           # "tool_use" when the model wants a tool
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})

    @property
    def raw_content(self) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        if self.text:
            blocks.append({"type": "text", "text": self.text})
        for c in self.tool_calls:
            blocks.append({"type": "tool_use", "id": c.id, "name": c.name, "input": c.args})
        return blocks


class LLMClient(ABC):
    @abstractmethod
    def create(self, system: str, messages: list[dict], tools: list[dict] | None = None,
               max_tokens: int = 2000) -> LLMResponse: ...


class AnthropicClient(LLMClient):
    """Thin wrapper over the Anthropic Messages API (needs ANTHROPIC_API_KEY)."""

    def __init__(self, model: str | None = None):
        import anthropic  # lazy import so tests/demo work without the SDK
        self._client = anthropic.Anthropic()
        self.model = model or os.environ.get("AGENT_MODEL", "claude-sonnet-5-5")

    def create(self, system, messages, tools=None, max_tokens=2000):
        kwargs: dict[str, Any] = dict(model=self.model, system=system, messages=messages,
                                      max_tokens=max_tokens)
        if tools:
            kwargs["tools"] = tools
        resp = self._client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in resp.content if b.type == "tool_use"]
        return LLMResponse(
            text=text,
            stop_reason="tool_use" if resp.stop_reason == "tool_use" else resp.stop_reason,
            tool_calls=calls,
            usage={"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens},
        )


class HuggingFaceClient(LLMClient):
    """OpenAI-compatible client for Hugging Face Inference Providers."""

    def __init__(self, model: str | None = None):
        from openai import OpenAI

        token = os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN is required when LLM_PROVIDER=huggingface")
        self._client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=token,
        )
        self.model = model or os.environ.get(
            "HF_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash:fastest"
        )

    @staticmethod
    def _openai_messages(system: str, messages: list[dict]) -> list[dict]:
        converted = [{"role": "system", "content": system}]
        for message in messages:
            content = message["content"]
            if not isinstance(content, list):
                converted.append({"role": message["role"], "content": content})
                continue

            if message["role"] == "assistant":
                text = "".join(
                    block["text"] for block in content if block.get("type") == "text"
                )
                calls = [
                    {
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(block["input"]),
                        },
                    }
                    for block in content
                    if block.get("type") == "tool_use"
                ]
                assistant_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": text or None,
                }
                if calls:
                    assistant_message["tool_calls"] = calls
                converted.append(assistant_message)
            else:
                converted.extend(
                    {
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"],
                        "content": block["content"],
                    }
                    for block in content
                    if block.get("type") == "tool_result"
                )
        return converted

    def create(self, system, messages, tools=None, max_tokens=2000):
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._openai_messages(system, messages),
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["input_schema"],
                    },
                }
                for tool in tools
            ]
        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message
        calls = [
            ToolCall(
                id=call.id,
                name=call.function.name,
                args=json.loads(call.function.arguments),
            )
            for call in (message.tool_calls or [])
        ]
        usage = response.usage
        return LLMResponse(
            text=message.content or "",
            stop_reason="tool_use" if choice.finish_reason == "tool_calls" else choice.finish_reason,
            tool_calls=calls,
            usage={
                "input_tokens": usage.prompt_tokens if usage else 0,
                "output_tokens": usage.completion_tokens if usage else 0,
            },
        )


class ScriptedLLM(LLMClient):
    """Returns canned responses in order. Used for deterministic unit tests."""

    def __init__(self, responses: list[LLMResponse | str]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, system, messages, tools=None, max_tokens=2000):
        self.calls.append({"system": system, "messages": [dict(m) for m in messages], "tools": tools})
        if not self._responses:
            raise AssertionError("ScriptedLLM ran out of responses")
        r = self._responses.pop(0)
        return LLMResponse(text=r) if isinstance(r, str) else r


class DemoLLM(LLMClient):
    """Rule-based stand-in so the whole pipeline runs offline (`--demo`).

    It is NOT intelligent: it just emits a valid plan, one tool call, and templated text,
    which is enough to exercise every code path in the system.
    """

    def create(self, system, messages, tools=None, max_tokens=2000):
        if "ROLE:PLANNER" in system:
            return self._plan(messages)
        if "ROLE:RETRIEVER" in system:
            return self._retrieve(messages)
        return self._execute(messages)

    @staticmethod
    def _first_text(messages) -> str:
        c = messages[0]["content"]
        return c if isinstance(c, str) else json.dumps(c)

    def _plan(self, messages) -> LLMResponse:
        goal = self._first_text(messages).split("GOAL:", 1)[-1].split("\n")[0].strip()
        plan = {
            "assumptions": ["Public encyclopedic sources are sufficient", "Audience is a general reader"],
            "steps": [
                {"id": "s1", "agent": "retriever", "depends_on": [],
                 "description": f"Search for background information about: {goal}",
                 "expected_output": "Evidence snippets with sources",
                 "success_criterion": "At least one sourced snippet"},
                {"id": "s2", "agent": "executor", "depends_on": ["s1"],
                 "description": "Extract the key facts from the evidence",
                 "expected_output": "Bullet list of facts",
                 "success_criterion": "Facts are traceable to sources"},
                {"id": "s3", "agent": "executor", "depends_on": ["s2"],
                 "description": "Draft a structured summary (overview, key points, open questions)",
                 "expected_output": "Structured summary",
                 "success_criterion": "Has overview and key points"},
            ],
        }
        return LLMResponse(text=json.dumps(plan), usage={"input_tokens": 50, "output_tokens": 120})

    def _retrieve(self, messages) -> LLMResponse:
        for m in messages:
            if isinstance(m["content"], list):
                for block in m["content"]:
                    if block.get("type") == "tool_result":
                        data = json.loads(block["content"])
                        if "error" in data:
                            return LLMResponse(text=f"Retrieval failed: {data['error']}")
                        lines = [f"- {r['title']}: {r['snippet']} (source: {r['source']})"
                                 for r in data.get("results", [])]
                        return LLMResponse(text="Evidence:\n" + "\n".join(lines or ["- none found"]),
                                           usage={"input_tokens": 80, "output_tokens": 60})
        task = self._first_text(messages).split("TASK:", 1)[-1].split("\n")[0]
        query = task.split(":", 1)[-1].strip()
        call = ToolCall("call_1", "search_wikipedia", {"query": query, "limit": 3})
        return LLMResponse(stop_reason="tool_use", tool_calls=[call],
                           usage={"input_tokens": 40, "output_tokens": 20})

    def _execute(self, messages) -> LLMResponse:
        text = self._first_text(messages)
        task = text.split("TASK:", 1)[-1].split("\n")[0].strip()
        ctx = text.split("CONTEXT:", 1)[-1].strip()
        return LLMResponse(text=f"[demo output] {task}\n{ctx[:600]}",
                           usage={"input_tokens": 100, "output_tokens": 80})
