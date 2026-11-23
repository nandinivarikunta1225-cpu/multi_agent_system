import json
from agents.executor import ExecutorAgent
from agents.planner import PlannerAgent
from agents.retriever import RetrieverAgent
from core.llm import LLMResponse, ScriptedLLM, ToolCall
from core.state import Status
from core.tools import Tool, ToolRegistry

GOOD_PLAN = json.dumps({"steps": [{"id": "s1", "description": "x", "agent": "executor"}]})


def echo_registry():
    return ToolRegistry([Tool("search", "d",
                              {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
                              lambda q: {"results": [{"title": q, "snippet": "s", "source": "http://src"}]})])


def tool_call(name="search", args=None, id="c1"):
    return LLMResponse(stop_reason="tool_use", tool_calls=[ToolCall(id, name, args or {"q": "x"})])


# ----------------------------------------------------------------- planner
def test_planner_valid_first_try():
    r = PlannerAgent(ScriptedLLM([GOOD_PLAN]), {"executor"}).run("goal")
    assert r.status == Status.SUCCESS and r.metadata["attempts"] == 1


def test_planner_reprompts_with_exact_error_then_succeeds():
    llm = ScriptedLLM(["garbage", GOOD_PLAN])
    r = PlannerAgent(llm, {"executor"}).run("goal")
    assert r.status == Status.SUCCESS and r.metadata["attempts"] == 2
    assert "not valid JSON" in llm.calls[1]["messages"][-1]["content"]


def test_planner_gives_up_after_bounded_retries():
    llm = ScriptedLLM(["bad"] * 3)
    r = PlannerAgent(llm, {"executor"}, max_plan_retries=2).run("goal")
    assert r.status == Status.FAILED and len(llm.calls) == 3


# ----------------------------------------------------------------- tool loop
def test_retriever_runs_tool_and_records_provenance():
    llm = ScriptedLLM([tool_call(), "Evidence: x (http://src)"])
    r = RetrieverAgent(llm, echo_registry()).run("find x")
    assert r.status == Status.SUCCESS
    assert r.metadata["evidence"] == [{"source": "http://src", "snippet": "s"}]
    # tool result was appended and fed back to the model
    assert llm.calls[1]["messages"][-1]["content"][0]["type"] == "tool_result"


def test_hallucinated_tool_error_is_fed_back_not_raised():
    llm = ScriptedLLM([tool_call("made_up"), "ok, no tool"])
    r = RetrieverAgent(llm, echo_registry()).run("t")
    assert r.status == Status.SUCCESS
    fed_back = llm.calls[1]["messages"][-1]["content"][0]["content"]
    assert "unknown tool" in fed_back


def test_iteration_cap_prevents_infinite_loop():
    llm = ScriptedLLM([tool_call(id=f"c{i}") for i in range(10)])
    r = RetrieverAgent(llm, echo_registry(), max_iters=3).run("t")
    assert r.status == Status.FAILED and "max iterations" in r.error and len(llm.calls) == 3


def test_executor_sees_only_dependency_outputs():
    llm = ScriptedLLM(["done"])
    ExecutorAgent(llm).run("write", {"dependencies": {"s1": "FACTS"}})
    prompt = llm.calls[0]["messages"][0]["content"]
    assert "FACTS" in prompt and llm.calls[0]["tools"] is None
