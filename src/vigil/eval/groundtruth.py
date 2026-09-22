"""What the clip actually does, written down before any model was allowed to look at it.

The ground truth is a sidecar the synthesiser produces from the authored scenario, so it is
independent of every backend: turning it into a pydantic model here is what stops a typo in a
``.groundtruth.json`` becoming a benchmark number. A file that does not validate is refused out
right, because a harness that silently scores against half a truth table is worse than one that
refuses to run.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from vigil.models.risk import HazardClass

SUFFIX = ".groundtruth.json"


class TrueHazard(BaseModel):
    """One event in the footage: when it becomes real, when it lands, and how hard."""

    model_config = ConfigDict(extra="forbid")

    type: HazardClass
    t_start: float = Field(ge=0.0, description="The hazard exists from here.")
    t_impact: float = Field(ge=0.0, description="Contact, or the fall, or the slip.")
    anticipatable_s: float = Field(
        gt=0.0, description="How much warning the footage contains, if you are looking."
    )
    severity: int = Field(ge=1, le=5)
    entities: list[str] = Field(default_factory=list)
    description: str

    @model_validator(mode="after")
    def _consistent(self) -> TrueHazard:
        if self.t_impact <= self.t_start:
            raise ValueError(f"{self.type}: t_impact must come after t_start")
        gap = self.t_impact - self.t_start
        if abs(gap - self.anticipatable_s) > 0.01:
            raise ValueError(
                f"{self.type}: anticipatable_s {self.anticipatable_s} disagrees with "
                f"t_impact - t_start = {gap:.2f}"
            )
        return self


class VideoTruth(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clip_id: str = Field(min_length=1)
    title: str = ""
    path: str
    duration_s: float = Field(gt=0.0)
    fps: float = Field(gt=0.0)
    resolution: tuple[int, int] | None = None
    synthetic: bool = True
    camera_label: str = ""
    scene: dict[str, str] = Field(default_factory=dict)
    hazards: list[TrueHazard] = Field(default_factory=list)
    notes: str = ""

    @property
    def is_control(self) -> bool:
        """A clip authored to be safe. Any finding on one of these is a false positive."""
        return not self.hazards


def load_truth(clip_dir: Path) -> dict[str, VideoTruth]:
    """Every sidecar in a clip directory, keyed by clip id.

    A directory with no sidecars returns an empty mapping rather than raising: which clips have
    truth is the caller's business, and ``vigil eval`` reports the ones it could not score
    instead of dying on them.
    """
    out: dict[str, VideoTruth] = {}
    for path in sorted(Path(clip_dir).glob(f"*{SUFFIX}")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            truth = VideoTruth.model_validate(raw)
        except (ValidationError, json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"unreadable ground truth in {path.name}: {exc}") from exc
        if truth.clip_id != path.name.removesuffix(SUFFIX):
            raise ValueError(
                f"{path.name} describes {truth.clip_id!r}; sidecars must match their clip"
            )
        out[truth.clip_id] = truth
    return out
