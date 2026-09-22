"""Risk vocabulary: the agent's output, as opposed to the camera's."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class HypothesisStatus(StrEnum):
    OPEN = "open"
    SUPPORTED = "supported"
    REJECTED = "rejected"


class Disposition(StrEnum):
    """What should happen to this finding."""

    MONITOR = "monitor"
    ADVISE = "advise"
    INTERVENE_NOW = "intervene_now"
    STOP_WORK = "stop_work"


class Likelihood(StrEnum):
    REMOTE = "remote"
    POSSIBLE = "possible"
    LIKELY = "likely"
    IMMINENT = "imminent"


class HazardClass(StrEnum):
    """The injury mechanisms the rulebook is indexed by.

    A shared vocabulary between three layers that otherwise have nothing to say to each other:
    the authored scenarios, the policy corpus, and the benchmark that scores a finding against
    ground truth. It exists so that "which rule governs this hazard" is a lookup rather than a
    vocabulary contest -- without it, a rider-on-the-tines rule surfaces for a struck-by
    pedestrian simply because both mention forklifts.
    """

    STRUCK_BY = "struck_by"
    STRUCK_BY_REVERSING_VEHICLE = "struck_by_reversing_vehicle"
    STRUCK_BY_FALLING_OBJECT = "struck_by_falling_object"
    WORKER_RIDING_ON_FORKS = "worker_riding_on_forks"
    SLIP_TRIP = "slip_trip"
    FALL_FROM_HEIGHT = "fall_from_height"
    CAUGHT_BETWEEN = "caught_between"


_RANK = {
    Likelihood.REMOTE: 1,
    Likelihood.POSSIBLE: 2,
    Likelihood.LIKELY: 3,
    Likelihood.IMMINENT: 4,
}


class RuleCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_id: str = Field(min_length=1)
    source: str = Field(description="Code of practice or site rulebook the rule came from.")
    text: str = Field(description="The rule, quoted or tightly paraphrased.")
    url: str | None = None
    relevance: float = Field(default=0.5, ge=0.0, le=1.0)


class Mitigation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str = Field(
        description="Imperative, concrete: 'Bank out the reversing alarm audible by now.'"
    )
    horizon: str = Field(default="immediate", description="immediate | end_of_shift | 7d | 30d")
    owner_role: str = Field(default="floor_supervisor")
    rationale: str = Field(default="")


class Hypothesis(BaseModel):
    """A falsifiable claim the agent then goes looking for in the footage."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    statement: str
    status: HypothesisStatus = HypothesisStatus.OPEN
    evidence: list[str] = Field(
        default_factory=list, description="Observations that moved the needle."
    )
    asked_of_perception: list[str] = Field(
        default_factory=list,
        description="Questions the agent sent back to the vision model to test this.",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def unresolved(self) -> bool:
        return self.status is HypothesisStatus.OPEN


class RiskAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hazard_id: str = Field(min_length=1)
    title: str
    severity: int = Field(ge=1, le=5, description="1 negligible .. 5 fatality plausible.")
    likelihood: Likelihood
    hazard_class: HazardClass | None = Field(
        default=None,
        description=(
            "Mechanism of harm, from the same seven the rulebook and the footage are indexed by. "
            "It is what lets two runs of one clip be compared and what the benchmark matches on; "
            "None when the finding's prose is the only clue to the mechanism."
        ),
    )
    predicted_event: str = Field(description="What happens if nothing changes, in one sentence.")
    lead_time_s: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Seconds of warning: how far ahead of the predicted event the finding was raised. "
            "The metric that separates anticipation from recording."
        ),
    )
    entities_involved: list[str] = Field(default_factory=list)
    clip_window: tuple[float, float] | None = Field(
        default=None, description="t0,t1 seconds in the source video."
    )
    reasoning: str = Field(
        default="", description="Why the agent believes this, referencing observations."
    )
    violated_rules: list[RuleCitation] = Field(default_factory=list)
    mitigations: list[Mitigation] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _window_ordered(self) -> RiskAssessment:
        if self.clip_window and self.clip_window[1] < self.clip_window[0]:
            raise ValueError("clip_window must be (start, end) in ascending order")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def risk_score(self) -> int:
        """5x4 matrix collapsed to 1..20 so two sites can compare notes."""
        return self.severity * _RANK[self.likelihood]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def disposition(self) -> Disposition:
        imminent_major = self.likelihood is Likelihood.IMMINENT and self.severity >= 4
        if self.risk_score >= 16 or imminent_major:
            return Disposition.STOP_WORK
        if self.risk_score >= 10:
            return Disposition.INTERVENE_NOW
        if self.risk_score >= 4:
            return Disposition.ADVISE
        return Disposition.MONITOR

    @property
    def actionable(self) -> bool:
        return self.disposition is not Disposition.MONITOR


class RiskLedger(BaseModel):
    """Everything the agent concluded about one video."""

    model_config = ConfigDict(extra="forbid")

    video_id: str = Field(min_length=1)
    assessments: list[RiskAssessment] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    summary: str = Field(default="")
    cleared: list[str] = Field(
        default_factory=list, description="Hazards actively ruled out -- not silence."
    )

    @property
    def top_risk(self) -> RiskAssessment | None:
        return max(self.assessments, key=lambda a: a.risk_score, default=None)

    @property
    def open_hazards(self) -> list[RiskAssessment]:
        return [a for a in self.assessments if a.actionable]

    @property
    def max_lead_time_s(self) -> float | None:
        values = [a.lead_time_s for a in self.assessments if a.lead_time_s is not None]
        return max(values) if values else None
