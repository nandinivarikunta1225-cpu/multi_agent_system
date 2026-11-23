import pytest
from core.tools import (HTTPFailure, Tool, ToolRegistry, make_wikipedia_search_tool,
                        validate_args, ToolArgumentError)


def make_http(sequence):
    """sequence items: dict (success) or Exception (raised)."""
    calls = {"n": 0}

    def http(url, timeout=8.0):
        item = sequence[min(calls["n"], len(sequence) - 1)]
        calls["n"] += 1
        if isinstance(item, Exception):
            raise item
        return item
    return http, calls


OK = {"query": {"search": [{"title": "Foo Bar", "snippet": 'a <span class="searchmatch">b</span>'}]}}


def test_success_has_provenance():
    http, _ = make_http([OK])
    res = make_wikipedia_search_tool(http).fn(query="foo")
    assert res["results"][0]["source"] == "https://en.wikipedia.org/wiki/Foo_Bar"
    assert "<span" not in res["results"][0]["snippet"]


def test_transient_error_retries_then_succeeds():
    http, calls = make_http([HTTPFailure("HTTP 503", True), OK])
    res = make_wikipedia_search_tool(http, sleep=lambda s: None).fn(query="foo")
    assert "results" in res and calls["n"] == 2


def test_persistent_failure_returns_error_object_not_exception():
    http, calls = make_http([HTTPFailure("timeout", True)])
    res = make_wikipedia_search_tool(http, sleep=lambda s: None).fn(query="foo")
    assert res == {"error": "timeout"} and calls["n"] == 3


def test_non_transient_error_does_not_retry():
    http, calls = make_http([HTTPFailure("HTTP 404", False)])
    res = make_wikipedia_search_tool(http, sleep=lambda s: None).fn(query="foo")
    assert "error" in res and calls["n"] == 1


def test_empty_results():
    http, _ = make_http([{"query": {"search": []}}])
    assert make_wikipedia_search_tool(http).fn(query="zzz")["results"] == []


def test_registry_allow_list_and_validation():
    reg = ToolRegistry([make_wikipedia_search_tool(make_http([OK])[0])])
    assert "unknown tool" in reg.execute("rm_rf", {})["error"]
    assert "missing required" in reg.execute("search_wikipedia", {})["error"]
    assert "must be integer" in reg.execute("search_wikipedia", {"query": "x", "limit": "3"})["error"]
    assert "unknown argument" in reg.execute("search_wikipedia", {"query": "x", "bogus": 1})["error"]


def test_registry_catches_tool_exceptions():
    boom = Tool("boom", "d", {"type": "object", "properties": {}}, lambda: 1 / 0)
    assert "ZeroDivisionError" in ToolRegistry([boom]).execute("boom", {})["error"]
