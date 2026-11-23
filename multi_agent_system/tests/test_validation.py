import json
import pytest
from core.validation import PlanValidationError, validate_plan, extract_json

AGENTS = {"retriever", "executor"}


def plan(steps):
    return json.dumps({"assumptions": ["a"], "steps": steps})


def step(i, deps=(), agent="executor"):
    return {"id": i, "description": f"do {i}", "agent": agent, "depends_on": list(deps)}


def test_valid_plan():
    _, steps = validate_plan(plan([step("s1", agent="retriever"), step("s2", ["s1"])]), AGENTS)
    assert [s.id for s in steps] == ["s1", "s2"]


def test_markdown_fenced_json_is_accepted():
    raw = "```json\n" + plan([step("s1")]) + "\n```"
    assert validate_plan(raw, AGENTS)[1][0].id == "s1"


def test_commentary_around_json_is_accepted():
    raw = "Sure! Here is the plan:\n" + plan([step("s1")]) + "\nHope it helps."
    assert extract_json(raw)["steps"][0]["id"] == "s1"


@pytest.mark.parametrize("raw,msg", [
    ("not json", "not valid JSON"),
    (json.dumps({"steps": []}), "non-empty"),
    (plan([{"id": "s1", "agent": "executor"}]), "description"),
    (plan([step("s1", agent="wizard")]), "unknown agent"),
    (plan([step("s1", ["nope"])]), "unknown step"),
    (plan([step("s1"), step("s1")]), "duplicate"),
    (plan([step("a", ["b"]), step("b", ["a"])]), "cycle"),
])
def test_invalid_plans(raw, msg):
    with pytest.raises(PlanValidationError, match=msg):
        validate_plan(raw, AGENTS)


def test_replan_may_depend_on_completed_steps_but_not_reuse_ids():
    ok = plan([step("r1", ["s1"])])
    assert validate_plan(ok, AGENTS, existing_ids={"s1"})[1][0].id == "r1"
    with pytest.raises(PlanValidationError, match="already used"):
        validate_plan(plan([step("s1")]), AGENTS, existing_ids={"s1"})
