"""The action grammar the agent thinks in.

Deliberately structured JSON rather than provider tool-calling: the same loop then runs
against Token Factory, a local model, or the scripted planner, and the trace records exactly
what the model asked for. Every action is a closed model with ``extra="forbid"`` so a
hallucinated argument fails loudly at the boundary instead of being silently dropped --
which is also what keeps ``vigil eval`` honest about what the agent actually did.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from vigil.models.risk import (
    HazardClass,
    Likelihood,
    Mitigation,
    RiskAssessment,
    RuleCitation,
)


class _Action(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str


class _WindowAction(_Action):
    t0_s: float = Field(ge=0.0, le=86_400.0)
    t1_s: float = Field(ge=0.0, le=86_400.0)

    @model_validator(mode="after")
    def _ascending(self) -> Self:
        if self.t1_s <= self.t0_s:
            raise ValueError(f"window must ascend: t0_s={self.t0_s} t1_s={self.t1_s}")
        return self


class SurveyAction(_WindowAction):
    """Broad read of a window -- the agent asking to look at a different slice in full."""

    tool: Literal["survey"] = "survey"


class RelookAction(_WindowAction):
    """One narrow question about one narrow window. This is the investigative move."""

    tool: Literal["relook"] = "relook"
    question: str = Field(min_length=8, max_length=600)


class PolicyAction(_Action):
    tool: Literal["policy_search"] = "policy_search"
    query: str = Field(min_length=3, max_length=300)
    hazard_type: str = Field(
        default="",
        max_length=60,
        description="Taxonomy term (struck_by, slip_trip, ...) when known; broadens recall.",
    )


class FlagAction(_Action):
    """A finding. ``rule_ids`` are resolved against the rulebook by the loop, never by the
    model, so a citation the agent invented cannot reach a report."""

    tool: Literal["flag"] = "flag"
    hazard_id: str = Field(min_length=1, max_length=40)
    title: str = Field(min_length=4, max_length=120)
    severity: int = Field(ge=1, le=5)
    likelihood: Likelihood
    hazard_class: HazardClass | None = Field(
        default=None,
        description=(
            "Mechanism of harm, one of the seven. Optional because a finding may be about an "
            "exposure the taxonomy has no word for; an invented fifth category is worse than None."
        ),
    )
    predicted_event: str = Field(min_length=8, max_length=600)
    lead_time_s: float | None = Field(default=None, ge=0.0, le=3600.0)
    entities_involved: list[str] = Field(default_factory=list, max_length=8)
    reasoning: str = Field(default="", max_length=2000)
    rule_ids: list[str] = Field(default_factory=list, max_length=6)
    mitigations: list[Mitigation] = Field(default_factory=list, max_length=5)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    def to_assessment(
        self,
        violated_rules: list[RuleCitation] | None = None,
        *,
        clip_window: tuple[float, float] | None = None,
    ) -> RiskAssessment:
        """Build the finding.

        ``violated_rules`` is injected by the toolkit rather than taken from the model, which
        is what keeps an invented citation out of a filed report. ``clip_window`` is likewise
        resolved from the footage actually read, not from what the model remembered.
        """
        return RiskAssessment(
            hazard_id=self.hazard_id,
            title=self.title,
            severity=self.severity,
            likelihood=self.likelihood,
            hazard_class=self.hazard_class,
            predicted_event=self.predicted_event,
            lead_time_s=self.lead_time_s,
            entities_involved=list(self.entities_involved),
            clip_window=clip_window,
            reasoning=self.reasoning,
            violated_rules=list(violated_rules or []),
            mitigations=list(self.mitigations),
            confidence=self.confidence,
        )


class ClearAction(_Action):
    """Something actively ruled out. Silence and 'checked and it is fine' are different
    statements, and only one of them is auditable."""

    tool: Literal["clear"] = "clear"
    subject: str = Field(min_length=3, max_length=120)
    reason: str = Field(min_length=8, max_length=600)


class FinishAction(_Action):
    tool: Literal["finish"] = "finish"
    summary: str = Field(min_length=8, max_length=1200)


Action = Annotated[
    SurveyAction | RelookAction | PolicyAction | FlagAction | ClearAction | FinishAction,
    Field(discriminator="tool"),
]

ACTION_ADAPTER: TypeAdapter[Action] = TypeAdapter(Action)


class Turn(BaseModel):
    """One model reply: the reasoning, plus exactly one action to execute."""

    model_config = ConfigDict(extra="forbid")

    thought: str = Field(min_length=1, max_length=1200)
    action: Action


TOOL_NAMES: tuple[str, ...] = (
    "survey",
    "relook",
    "policy_search",
    "flag",
    "clear",
    "finish",
)
