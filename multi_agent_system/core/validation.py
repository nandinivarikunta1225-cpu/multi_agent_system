"""Plan validation: JSON parsing, required fields, known agents, real deps, no cycles."""
from __future__ import annotations

import json
import re

from .state import PlanStep


class PlanValidationError(ValueError):
    pass


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def extract_json(text: str) -> dict:
    """Models often wrap JSON in markdown fences or add commentary; handle both."""
    cleaned = _FENCE.sub("", text.strip()).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                pass
        raise PlanValidationError("plan is not valid JSON")


def topological_order(steps: list[PlanStep]) -> list[str]:
    """Kahn's algorithm. Raises PlanValidationError on cycles."""
    deps = {s.id: set(s.depends_on) for s in steps}
    order: list[str] = []
    ready = [i for i, d in deps.items() if not d]
    while ready:
        n = ready.pop(0)
        order.append(n)
        for i, d in deps.items():
            if n in d:
                d.discard(n)
                if not d and i not in order and i not in ready:
                    ready.append(i)
    if len(order) != len(steps):
        raise PlanValidationError("plan has a dependency cycle")
    return order


def validate_plan(raw: str, known_agents: set[str],
                  existing_ids: set[str] | None = None) -> tuple[list[str], list[PlanStep]]:
    """Returns (assumptions, steps). existing_ids = already-completed step ids (for replans)."""
    existing_ids = existing_ids or set()
    data = extract_json(raw)
    if not isinstance(data, dict) or not isinstance(data.get("steps"), list) or not data["steps"]:
        raise PlanValidationError("plan must be an object with a non-empty 'steps' list")

    steps: list[PlanStep] = []
    for i, s in enumerate(data["steps"]):
        if not isinstance(s, dict):
            raise PlanValidationError(f"step {i} is not an object")
        for key in ("id", "description", "agent"):
            if not isinstance(s.get(key), str) or not s[key].strip():
                raise PlanValidationError(f"step {i} is missing required field '{key}'")
        deps = s.get("depends_on", [])
        if not isinstance(deps, list) or not all(isinstance(d, str) for d in deps):
            raise PlanValidationError(f"step '{s['id']}': depends_on must be a list of step ids")
        steps.append(PlanStep(
            id=s["id"], description=s["description"], agent=s["agent"], depends_on=deps,
            expected_output=str(s.get("expected_output", "")),
            success_criterion=str(s.get("success_criterion", "")),
        ))

    ids = [s.id for s in steps]
    if len(set(ids)) != len(ids):
        raise PlanValidationError("duplicate step ids")
    clash = set(ids) & existing_ids
    if clash:
        raise PlanValidationError(f"step ids already used by completed steps: {sorted(clash)}")
    for s in steps:
        if s.agent not in known_agents:
            raise PlanValidationError(f"step '{s.id}' uses unknown agent '{s.agent}' (known: {sorted(known_agents)})")
        for d in s.depends_on:
            if d not in ids and d not in existing_ids:
                raise PlanValidationError(f"step '{s.id}' depends on unknown step '{d}'")

    # cycle check only over new steps (deps on completed steps are already satisfied)
    topological_order([PlanStep(s.id, s.description, s.agent, [d for d in s.depends_on if d in ids]) for s in steps])
    assumptions = [str(a) for a in data.get("assumptions", [])] if isinstance(data.get("assumptions", []), list) else []
    return assumptions, steps
