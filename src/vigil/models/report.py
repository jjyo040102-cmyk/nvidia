"""The deliverable a safety manager actually files."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from vigil.models.risk import Mitigation, RiskAssessment, RuleCitation
from vigil.models.trace import InvestigationTrace


class TimelineEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t_s: float = Field(ge=0.0)
    event: str
    observed_by: str = Field(default="perception", description="perception | agent | operator")


class ModelProvenance(BaseModel):
    """Judges ask 'what ran, where, for how much'. This answers without a re-run."""

    model_config = ConfigDict(extra="forbid")

    perception_backend: str
    vision_model: str = Field(
        description="The model that read the frames. The backend name alone does not say."
    )
    reasoning_model: str
    inference_provider: str = "Nebius Token Factory"
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    wall_clock_ms: int = Field(default=0, ge=0)
    video_hours_analysed: float = Field(default=0.0, ge=0.0)


class IncidentReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    headline: str
    narrative: str = Field(description="Plain-language account a human can act on.")
    timeline: list[TimelineEntry] = Field(default_factory=list)
    assessments: list[RiskAssessment] = Field(default_factory=list)
    citations: list[RuleCitation] = Field(default_factory=list)
    actions: list[Mitigation] = Field(default_factory=list)
    provenance: ModelProvenance
    trace: InvestigationTrace | None = None
    disclaimer: str = (
        "Decision support only. Vigil surfaces anticipated physical risk for a competent human to "
        "confirm; it does not replace a trained safety officer or emergency procedure."
    )

    @property
    def worst(self) -> RiskAssessment | None:
        return max(self.assessments, key=lambda a: a.risk_score, default=None)

    @property
    def lead_time_s(self) -> float | None:
        values = [a.lead_time_s for a in self.assessments if a.lead_time_s is not None]
        return max(values) if values else None
