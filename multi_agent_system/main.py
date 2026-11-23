"""CLI entry point.

  python main.py --demo "Research multi-agent systems"     # offline, no API key
  ANTHROPIC_API_KEY=... python main.py "Research topic X and produce a structured summary"
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

from agents.executor import ExecutorAgent
from agents.planner import PlannerAgent
from agents.retriever import RetrieverAgent
from core.llm import AnthropicClient, DemoLLM, HuggingFaceClient, LLMClient
from core.orchestrator import Config, Orchestrator
from core.tools import ToolRegistry, canned_http_get, default_http_get, make_wikipedia_search_tool


def build_system(llm: LLMClient, http_get=default_http_get, config: Config | None = None) -> Orchestrator:
    retriever = RetrieverAgent(llm, ToolRegistry([make_wikipedia_search_tool(http_get)]))
    executor = ExecutorAgent(llm)                      # no tools: least privilege
    agents = {"retriever": retriever, "executor": executor}
    planner = PlannerAgent(llm, known_agents=set(agents))
    return Orchestrator(planner, agents, synthesizer=ExecutorAgent(llm), config=config)


def main(argv=None) -> int:
    load_dotenv()

    p = argparse.ArgumentParser(description="Multi-agent research & task automation")
    p.add_argument("goal")
    p.add_argument("--demo", action="store_true", help="offline: scripted LLM + canned search results")
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(name)s: %(message)s")
    if a.demo:
        llm = DemoLLM()
    else:
        provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
        if provider == "anthropic":
            llm = AnthropicClient()
        elif provider == "huggingface":
            if not os.environ.get("HF_TOKEN"):
                p.error("HF_TOKEN is missing; add your Hugging Face Inference Providers token to .env")
            llm = HuggingFaceClient()
        else:
            p.error("LLM_PROVIDER must be 'anthropic' or 'huggingface'")
    http = canned_http_get if a.demo else default_http_get
    orch = build_system(llm, http, Config(max_steps=a.max_steps))

    report = orch.run(a.goal)
    s = report.state
    print(f"\nStatus: {report.status} ({report.reason})")
    print(f"Steps run: {s.total_steps} | replans: {s.replans} | tokens: {s.tokens_used}")
    if s.assumptions:
        print("Assumptions:", "; ".join(s.assumptions))
    print("\n--- Step log ---")
    for h in s.history:
        print(h)
    print("\n--- Final answer ---")
    print(report.answer)
    return 0 if report.status == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
