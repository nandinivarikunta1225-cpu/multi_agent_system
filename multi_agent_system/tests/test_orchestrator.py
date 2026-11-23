import json
from agents.base import BaseAgent
from core.llm import ScriptedLLM
from core.orchestrator import Config, Orchestrator
from agents.planner import PlannerAgent
from core.state import AgentResult, Status


class StubAgent(BaseAgent):
    """Stub agent with scripted results; records calls so ordering can be asserted."""
    def __init__(self, name, results, log):
        super().__init__(llm=None, system_prompt="")
        self.name, self._results, self.log = name, list(results), log

    def run(self, task, context=None):
        self.log.append((self.name, task, context))
        r = self._results.pop(0) if len(self._results) > 1 else self._results[0]
        return r() if callable(r) else r


def ok(text="out"):
    return AgentResult(Status.SUCCESS, output=text, metadata={"tokens": 10})


def bad(err="boom"):
    return AgentResult(Status.FAILED, error=err)


def plan_json(*steps):
    return json.dumps({"assumptions": [], "steps": [
        {"id": i, "description": f"task {i}", "agent": a, "depends_on": d} for i, a, d in steps]})


def build(plans, agents, **cfg):
    planner = PlannerAgent(ScriptedLLM(plans), set(agents))
    synth = StubAgent("synth", [ok("FINAL")], [])
    return Orchestrator(planner, agents, synth, Config(**cfg)), synth


def test_executes_in_dependency_order_and_passes_only_dependency_outputs():
    log = []
    agents = {"a": StubAgent("a", [ok("A-out")], log), "b": StubAgent("b", [ok("B-out")], log)}
    orch, _ = build([plan_json(("s2", "b", ["s1"]), ("s1", "a", []))], agents)
    rep = orch.run("goal")
    assert rep.status == "success" and rep.answer == "FINAL"
    assert [c[0] for c in log] == ["a", "b"]                      # s1 before s2 despite plan order
    assert log[1][2]["dependencies"] == {"s1": "A-out"}


def test_retry_then_success():
    log = []
    agents = {"a": StubAgent("a", [bad(), ok()], log)}
    rep = build([plan_json(("s1", "a", []))], agents)[0].run("g")
    assert rep.status == "success" and len(log) == 2


def test_empty_output_counts_as_failure():
    agents = {"a": StubAgent("a", [AgentResult(Status.SUCCESS, output="  ")], [])}
    rep = build([plan_json(("s1", "a", []))], agents, max_retries_per_step=1, max_replans=0)[0].run("g")
    assert rep.status == "failed"


def test_replan_keeps_completed_results_and_finishes():
    log = []
    agents = {"a": StubAgent("a", [ok("first")], log),
              "b": StubAgent("b", [bad("api down")], log),
              "c": StubAgent("c", [ok("recovered")], log)}
    plans = [plan_json(("s1", "a", []), ("s2", "b", ["s1"])),
             plan_json(("r1", "c", ["s1"]))]          # replan: new id, depends on completed s1
    orch, _ = build(plans, agents, max_retries_per_step=0)
    # replanner may only use agents known to planner: include c in set via agents dict
    rep = orch.run("goal")
    assert rep.status == "success" and rep.state.replans == 1
    assert set(rep.state.results) == {"s1", "r1"}
    assert [c[0] for c in log].count("a") == 1                   # completed step not redone


def test_abort_with_partial_results_when_replans_exhausted():
    agents = {"a": StubAgent("a", [ok("first")], []), "b": StubAgent("b", [bad("down")], [])}
    orch, _ = build([plan_json(("s1", "a", []), ("s2", "b", ["s1"]))], agents,
                    max_retries_per_step=0, max_replans=0)
    rep = orch.run("goal")
    assert rep.status == "partial" and "s2" in rep.reason and rep.answer == "FINAL"


def test_max_steps_guardrail():
    agents = {"a": StubAgent("a", [ok()], [])}
    plan = plan_json(("s1", "a", []), ("s2", "a", ["s1"]), ("s3", "a", ["s2"]))
    rep = build([plan], agents, max_steps=2)[0].run("g")
    assert rep.status == "partial" and "max steps" in rep.reason and rep.state.total_steps == 2


def test_token_budget_guardrail():
    agents = {"a": StubAgent("a", [ok()], [])}            # each step uses 10 tokens
    plan = plan_json(("s1", "a", []), ("s2", "a", ["s1"]))
    rep = build([plan], agents, token_budget=10)[0].run("g")
    assert rep.status == "partial" and "token" in rep.reason


def test_planner_failure_is_clean_failure():
    agents = {"a": StubAgent("a", [ok()], [])}
    rep = build(["bad", "bad", "bad"], agents)[0].run("g")
    assert rep.status == "failed" and "valid plan" in rep.reason


def test_agent_exception_does_not_crash_loop():
    def explode():
        raise RuntimeError("kaboom")
    agents = {"a": StubAgent("a", [explode], [])}
    rep = build([plan_json(("s1", "a", []))], agents, max_retries_per_step=0, max_replans=0)[0].run("g")
    assert rep.status == "failed"
