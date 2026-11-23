# Multi-Agent Research and Task Automation System

A modular, object-oriented multi-agent pipeline (Planner, Retriever, Executor) that decomposes a goal
into a validated plan and executes it through a central orchestrator, with tool calling, guardrails,
and failure handling.

## Run it

Install the dependencies and add your Hugging Face token to the local `.env` file
in this directory. Create a fine-grained token with **Make calls to Inference Providers**
permission at <https://huggingface.co/settings/tokens/new?ownUserPermissions=inference.serverless.write&tokenType=fineGrained>.
The `.env` file is ignored by Git so your token is not committed.

```powershell
python -m pip install -r requirements.txt
# In .env, set HF_TOKEN to your Hugging Face token
python main.py "Research topic X and produce a structured summary"
```

```bash
# Offline demo: scripted LLM + canned search results (no API key, no network)
python main.py --demo "Research multi-agent systems" -v

# Anthropic alternative: set LLM_PROVIDER=anthropic and ANTHROPIC_API_KEY in .env
export ANTHROPIC_API_KEY=...
export AGENT_MODEL=claude-sonnet-5-5      # optional
python main.py "Research topic X and produce a structured summary"

# Tests (all deterministic, mock LLM and mock HTTP)
pip install pytest && python -m pytest -q
```

## Layout

```
main.py                  CLI + build_system() wiring
core/state.py            AgentResult(status, output, error, metadata), PlanStep, SharedState
core/llm.py              LLMClient interface; Anthropic/Hugging Face clients, ScriptedLLM (tests), DemoLLM (offline)
core/tools.py            Tool, ToolRegistry (allow-list + arg validation), HTTP retries/backoff, Wikipedia tool
core/validation.py       Plan validation: JSON (fence-tolerant), fields, known agents, deps, cycle check
core/orchestrator.py     Control loop, guardrails, retry / replan / abort, final synthesis
agents/base.py           BaseAgent (abstract run()) + the tool-calling loop with a hard iteration cap
agents/planner.py        Goal -> JSON plan; re-prompts with the exact validation error
agents/retriever.py      Tool-using agent; returns evidence with provenance (source + snippet)
agents/executor.py       Gets only its step + dependency outputs; no tools (least privilege)
tests/                   36 tests: validation, tools, agents, orchestrator, end-to-end
```

## How it maps to the resume line / interview notes

| Claim | Where |
|---|---|
| Modular OOP agent classes | `agents/base.py` (abstract `BaseAgent`), three concrete subclasses |
| Defined interfaces / inter-agent communication | `AgentResult` + `SharedState`; agents never call each other |
| Tool-calling logic | `BaseAgent.run_with_tools`, `ToolRegistry.execute` |
| Problem decomposition / ambiguity | Planner emits assumptions + success criterion per step |
| Control-flow design | `Orchestrator.run`: DAG order, retry vs replan vs abort |
| Guardrails | `Config`: max steps, retries/step, replans, timeout, token budget; per-agent `max_iters` |

## Things to fill in / know before the interview

- **Own control flow vs framework:** this version uses plain Python (no LangChain etc.). Say that.
- **"Hardest bug":** the fence-stripping in `core/validation.py::extract_json` is a real example
  (models wrap JSON in markdown fences); `tests/test_validation.py` is the regression test.
- **Not implemented (your "what I'd improve" list):** parallel execution of independent steps,
  persistent state (SQS/DynamoDB), tracing/correlation IDs, a Critic agent, human approval, caching.
- `DemoLLM` is a rule-based stub for offline runs; it does not reason. Real behavior needs a provider token.
