"""The agent's own audit trail -- rendered live, and the reason a judge can trust it."""

from __future__ import annotations

import time
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class TraceKind(StrEnum):
    THOUGHT = "thought"
    TOOL_CALL = "tool_call"
    OBSERVATION = "observation"
    DECISION = "decision"
    ERROR = "error"


class TraceStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    kind: TraceKind
    title: str = Field(
        description="Short line for the UI, e.g. 'Asking Cosmos about the reversing path'."
    )
    detail: str = Field(default="")
    tool: str | None = None
    args: dict[str, object] = Field(default_factory=dict)
    result_summary: str = Field(default="")
    duration_ms: int = Field(default=0, ge=0)
    mono: float = Field(default_factory=time.monotonic, description="Seconds since process start.")


class InvestigationTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_id: str = Field(min_length=1)
    steps: list[TraceStep] = Field(default_factory=list)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    wall_clock_ms: int = Field(default=0, ge=0)
    iterations: int = Field(default=0, ge=0)

    def add(self, step: TraceStep) -> TraceStep:
        self.steps.append(step)
        return step

    @property
    def last_thought(self) -> str:
        for step in reversed(self.steps):
            if step.kind is TraceKind.THOUGHT:
                return step.detail or step.title
        return ""

    def tools_used(self) -> list[str]:
        seen: list[str] = []
        for step in self.steps:
            if step.tool and step.tool not in seen:
                seen.append(step.tool)
        return seen

    def to_loglines(self) -> list[str]:
        return [f"[{s.index:>2}] {s.kind.value:<10} {s.title}" for s in self.steps]
