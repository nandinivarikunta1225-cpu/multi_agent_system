"""Shared state and the uniform message contract between components."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Status(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    NEEDS_RETRY = "needs_retry"


@dataclass
class AgentResult:
    """Every agent returns this same typed result (the 'defined interface')."""
    status: Status
    output: str = ""
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)  # tokens, latency, evidence...


@dataclass
class PlanStep:
    id: str
    description: str
    agent: str
    depends_on: list[str] = field(default_factory=list)
    expected_output: str = ""
    success_criterion: str = ""


@dataclass
class SharedState:
    """Agents never call each other: they read/write this via the orchestrator."""
    goal: str
    assumptions: list[str] = field(default_factory=list)
    plan: list[PlanStep] = field(default_factory=list)
    results: dict[str, AgentResult] = field(default_factory=dict)   # step_id -> result
    failed: dict[str, str] = field(default_factory=dict)            # step_id -> error
    history: list[dict[str, Any]] = field(default_factory=list)     # per-step log
    total_steps: int = 0
    replans: int = 0
    tokens_used: int = 0
