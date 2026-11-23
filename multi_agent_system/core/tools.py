"""Tool definitions, argument validation, and robust external-API wrappers."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable


class ToolArgumentError(ValueError):
    pass


_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool,
          "array": list, "object": dict}


def validate_args(schema: dict, args: dict) -> dict:
    """Minimal JSON-schema check: required keys, known keys, basic types."""
    if not isinstance(args, dict):
        raise ToolArgumentError("arguments must be an object")
    props = schema.get("properties", {})
    for key in schema.get("required", []):
        if key not in args:
            raise ToolArgumentError(f"missing required argument '{key}'")
    for key, val in args.items():
        if key not in props:
            raise ToolArgumentError(f"unknown argument '{key}'")
        expected = _TYPES.get(props[key].get("type", ""))
        if expected and (not isinstance(val, expected) or (props[key]["type"] == "integer" and isinstance(val, bool))):
            raise ToolArgumentError(f"argument '{key}' must be {props[key]['type']}")
    return args


@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    fn: Callable[..., dict]

    def definition(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.schema}


class ToolRegistry:
    """Per-agent allow-list of tools (least privilege). execute() never raises."""

    def __init__(self, tools: list[Tool] | None = None):
        self._tools = {t.name: t for t in (tools or [])}

    def __bool__(self):
        return bool(self._tools)

    def definitions(self) -> list[dict]:
        return [t.definition() for t in self._tools.values()]

    def execute(self, name: str, args: dict) -> dict:
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"unknown tool '{name}'. Available: {sorted(self._tools)}"}
        try:
            return tool.fn(**validate_args(tool.schema, args))
        except ToolArgumentError as e:
            return {"error": f"invalid arguments: {e}"}
        except Exception as e:  # feed errors back to the model, never crash the loop
            return {"error": f"{type(e).__name__}: {e}"}


# --------------------------------------------------------------------- HTTP layer
class HTTPFailure(Exception):
    def __init__(self, message: str, transient: bool):
        super().__init__(message)
        self.transient = transient


def default_http_get(url: str, timeout: float = 8.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "multi-agent-demo/1.0 (educational)"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                raise HTTPFailure(f"HTTP {resp.status}", transient=resp.status >= 500)
            return json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as e:
        raise HTTPFailure(f"HTTP {e.code}", transient=e.code == 429 or e.code >= 500)
    except (urllib.error.URLError, TimeoutError) as e:
        raise HTTPFailure(f"network error: {e}", transient=True)
    except json.JSONDecodeError:
        raise HTTPFailure("response was not valid JSON", transient=False)


def with_retries(fn: Callable[[], dict], attempts: int = 3, base_delay: float = 0.5,
                 sleep: Callable[[float], None] = time.sleep) -> dict:
    """Retry transient failures with exponential backoff; return an error object at the end."""
    last = "unknown error"
    for i in range(attempts):
        try:
            return fn()
        except HTTPFailure as e:
            last = str(e)
            if not e.transient:
                break
            if i < attempts - 1:
                sleep(base_delay * (2 ** i))
    return {"error": last}


def make_wikipedia_search_tool(http_get: Callable[..., dict] = default_http_get,
                               sleep: Callable[[float], None] = time.sleep) -> Tool:
    def search(query: str, limit: int = 3) -> dict:
        limit = max(1, min(limit, 5))
        url = ("https://en.wikipedia.org/w/api.php?action=query&list=search&format=json"
               f"&srlimit={limit}&srsearch={urllib.parse.quote(query)}")
        data = with_retries(lambda: http_get(url), sleep=sleep)
        if "error" in data:
            return data
        hits = data.get("query", {}).get("search", [])
        if not hits:
            return {"results": [], "note": "no results"}
        results = []
        for h in hits:
            snippet = h.get("snippet", "").replace('<span class="searchmatch">', "").replace("</span>", "")
            results.append({
                "title": h.get("title", ""),
                "snippet": snippet,
                "source": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(h.get("title", "").replace(" ", "_")),
            })
        return {"results": results}

    return Tool(
        name="search_wikipedia",
        description="Search Wikipedia and return titles, snippets and source URLs.",
        schema={"type": "object",
                "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
                "required": ["query"]},
        fn=search,
    )


def canned_http_get(url: str, timeout: float = 8.0) -> dict:
    """Offline stand-in for Wikipedia used by --demo."""
    return {"query": {"search": [
        {"title": "Multi-agent system",
         "snippet": 'A <span class="searchmatch">multi-agent</span> system is made of several interacting agents.'},
        {"title": "Intelligent agent",
         "snippet": "An agent perceives its environment and takes actions to achieve goals."},
    ]}}
