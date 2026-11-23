import sys
from types import SimpleNamespace

from core.llm import AnthropicClient, HuggingFaceClient, ToolCall


def test_anthropic_client_uses_supported_messages_parameters(monkeypatch):
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="reply")],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=2, output_tokens=3),
    )
    calls = {}

    class FakeMessages:
        def create(self, **kwargs):
            calls.update(kwargs)
            return response

    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        SimpleNamespace(Anthropic=lambda: SimpleNamespace(messages=FakeMessages())),
    )
    client = AnthropicClient(model="test-model")

    result = client.create("system prompt", [{"role": "user", "content": "hello"}])

    assert result.text == "reply"
    assert calls == {
        "model": "test-model",
        "system": "system prompt",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 2000,
    }


def test_huggingface_client_maps_tools_and_tool_results(monkeypatch):
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call-1",
                            function=SimpleNamespace(
                                name="search",
                                arguments='{"query":"agents"}',
                            ),
                        )
                    ],
                ),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=7),
    )
    calls = {}

    class FakeCompletions:
        def create(self, **kwargs):
            calls.update(kwargs)
            return response

    fake_openai = SimpleNamespace(
        OpenAI=lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions())
        )
    )
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    client = HuggingFaceClient(model="test-model")

    result = client.create(
        "system prompt",
        [
            {"role": "user", "content": "find agents"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call-0",
                        "name": "search",
                        "input": {"query": "agents"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call-0",
                        "content": '{"results":[]}',
                    }
                ],
            },
        ],
        tools=[
            {
                "name": "search",
                "description": "Search",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
    )

    assert result.stop_reason == "tool_use"
    assert result.tool_calls == [ToolCall("call-1", "search", {"query": "agents"})]
    assert result.usage == {"input_tokens": 5, "output_tokens": 7}
    assert calls["messages"] == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "find agents"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-0",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": '{"query": "agents"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-0", "content": '{"results":[]}'},
    ]
    assert calls["tools"][0]["function"]["name"] == "search"


def test_huggingface_client_requires_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)

    try:
        HuggingFaceClient()
    except ValueError as exc:
        assert "HF_TOKEN is required" in str(exc)
    else:
        raise AssertionError("Expected missing Hugging Face token to fail")
