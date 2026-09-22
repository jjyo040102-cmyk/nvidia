"""Physical-world scene representation produced by the perception layer.

The same schema is serialized to JSON Schema and handed to the vision model, so every
field must be describable from pixels alone. Judgements that need site knowledge
(law, policy, responsibility) live in :mod:`vigil.models.risk`.

Validation is deliberately asymmetric. Unknown extra keys are ignored, because vision models
volunteer fields nobody asked for and rejecting a good read over a stray "colour" would throw
away real signal. But the structure that carries meaning -- declared entities, an ordered time
window, conflicts that point at things which exist -- is enforced and raises rather than being
silently repaired. Bounded, *recorded* repair happens in :func:`vigil.perception.base.repair_scene`
before validation, and lands in ``repair_notes`` so the pipeline always knows when it guessed.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AreaType(StrEnum):
    PRODUCTION_FLOOR = "production_floor"
    WAREHOUSE = "warehouse"
    CONSTRUCTION_SITE = "construction_site"
    LOADING_DOCK = "loading_dock"
    VEHICLE_YARD = "vehicle_yard"
    PUBLIC_SPACE = "public_space"
    RESIDENTIAL = "residential"
    LABORATORY = "laboratory"
    OTHER = "other"


class Lighting(StrEnum):
    BRIGHT_UNIFORM = "bright_uniform"
    TYPICAL_INDOOR = "typical_indoor"
    LOW = "low"
    BACKLIT_OR_GLARE = "backlit_or_glare"
    NIGHT_ARTIFICIAL = "night_artificial"
    UNKNOWN = "unknown"


class SurfaceCondition(StrEnum):
    DRY_CLEAR = "dry_clear"
    WET = "wet"
    CLUTTERED = "cluttered"
    UNEVEN = "uneven"
    OBSTRUCTED_EGRESS = "obstructed_egress"
    UNKNOWN = "unknown"


class Kinematics(StrEnum):
    STATIONARY = "stationary"
    WALKING = "walking"
    RUNNING = "running"
    DRIVING = "driving"
    REVERSING = "reversing"
    LIFTING = "lifting"
    FALLING = "falling"
    CARRYING = "carrying_load"
    UNKNOWN = "unknown"


class Role(StrEnum):
    VULNERABLE_PARTY = "vulnerable_party"
    POWERED_MACHINE = "powered_machine"
    OPERATOR = "operator"
    BYSTANDER = "bystander"
    UNKNOWN = "unknown"


def normalise_ref(value: object) -> str:
    """One handle format, shared by entities and the conflicts that point at them.

    A vision model writes "Pedestrian A" as often as "P1". If the two sides normalise
    differently -- spaces kept in one, dropped in the other -- the scene validator concludes
    the conflict names something undeclared and the whole read is refused, which is the exact
    failure this schema is supposed to absorb.
    """
    return str(value).strip().upper().replace(" ", "")


class Entity(BaseModel):
    """A person, vehicle or object visible in the clip."""

    model_config = ConfigDict(extra="ignore")

    ref: str = Field(
        min_length=1, max_length=16, description="Short handle used by relations, e.g. 'P1', 'V1'."
    )
    category: str = Field(description="What it is: 'worker', 'forklift', 'pedestrian', 'crate'.")
    role: Role = Role.UNKNOWN
    location: str = Field(
        description="Where in frame, in words: 'centre-left, on the machine lane'."
    )
    kinematics: Kinematics = Kinematics.UNKNOWN
    speed_mps: float | None = Field(
        default=None, ge=0.0, le=40.0, description="Estimate in metres/second."
    )
    ppe_missing: list[str] = Field(
        default_factory=list, description="e.g. 'hi_vis', 'hard_hat', 'harness'."
    )
    attention: str = Field(default="unknown", description="What the entity appears occupied with.")

    @field_validator("ref", mode="before")
    @classmethod
    def _upper_ref(cls, v: object) -> object:
        return normalise_ref(v) if isinstance(v, str) else v

    @field_validator("ppe_missing")
    @classmethod
    def _strip_ppe(cls, v: list[str]) -> list[str]:
        return [item.strip() for item in v if item and item.strip()]


class Conflict(BaseModel):
    """An interaction the vision model can already see coming."""

    model_config = ConfigDict(extra="ignore")

    participants: list[str] = Field(min_length=2, max_length=4, description="Entity refs involved.")
    kind: str = Field(
        description=(
            "Mechanism of harm, one of: struck_by | struck_by_reversing_vehicle | "
            "struck_by_falling_object | worker_riding_on_forks | slip_trip | fall_from_height | "
            "caught_between. Name what would injure, not what the object is: a person on the forks "
            "is worker_riding_on_forks even though the truck is also moving."
        )
    )
    time_to_event_s: float | None = Field(
        default=None,
        ge=0.0,
        le=120.0,
        description=(
            "Seconds from the end of the read window to contact if nothing changes; "
            "0.0 when contact already happens inside the window; None when not estimable."
        ),
    )
    distance_m: float | None = Field(default=None, ge=0.0, le=500.0)
    severity_hint: int = Field(ge=1, le=5, description="1 negligible injury .. 5 life-threatening.")
    rationale: str = Field(description="The physical argument: momentum, sightline, escape route.")

    @field_validator("participants")
    @classmethod
    def _distinct(cls, v: list[str]) -> list[str]:
        norm = sorted({normalise_ref(p) for p in v if str(p).strip()})
        if len(norm) < 2:
            raise ValueError("a conflict needs two distinct participants")
        return norm


class VisibilityLimit(BaseModel):
    """Where the camera -- or the actors -- physically cannot see."""

    model_config = ConfigDict(extra="ignore")

    occluder: str = Field(
        description="What blocks the view: stacked racks, a pillar, a truck body."
    )
    hidden_zone: str = Field(description="Which area is unobservable, and to whom.")
    camera_blind: bool = Field(
        default=False, description="True when the surveillance view itself misses this."
    )


class KeyMoment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    t_s: float = Field(ge=0.0, le=86_400.0)
    description: str = Field(description="What changes at this instant, in one sentence.")


class SceneUnderstanding(BaseModel):
    """Structured physical-common-sense read of one clip."""

    model_config = ConfigDict(extra="ignore")

    clip_id: str = Field(min_length=1)
    t0_s: float = Field(ge=0.0, le=86_400.0)
    t1_s: float = Field(ge=0.0, le=86_400.0)
    area_type: AreaType = AreaType.OTHER
    lighting: Lighting = Lighting.UNKNOWN
    surface: SurfaceCondition = SurfaceCondition.UNKNOWN
    congestion: int = Field(
        default=0, ge=0, le=1000, description="Approximate count of moving entities."
    )
    entities: list[Entity] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    visibility_limits: list[VisibilityLimit] = Field(default_factory=list)
    key_moments: list[KeyMoment] = Field(default_factory=list)
    dynamics_narrative: str = Field(
        default="",
        description="Cause-and-effect chain in prose: what moves, toward what, and why it matters.",
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    repair_notes: list[str] = Field(
        default_factory=list,
        description="Where the parser had to guess. Non-empty means the model reply was malformed.",
    )

    @model_validator(mode="after")
    def _consistent(self) -> SceneUnderstanding:
        if self.t1_s < self.t0_s:
            raise ValueError(f"t1_s ({self.t1_s}) precedes t0_s ({self.t0_s})")
        declared = {e.ref for e in self.entities}
        for conflict in self.conflicts:
            undeclared = sorted(set(conflict.participants) - declared)
            if undeclared:
                raise ValueError(f"conflict references undeclared entities: {undeclared}")
        return self

    @property
    def duration_s(self) -> float:
        return self.t1_s - self.t0_s

    @property
    def entities_by_ref(self) -> dict[str, Entity]:
        return {e.ref: e for e in self.entities}

    @property
    def vulnerable_parties(self) -> list[Entity]:
        return [
            e
            for e in self.entities
            if e.role is Role.VULNERABLE_PARTY
            or e.kinematics in {Kinematics.WALKING, Kinematics.RUNNING}
        ]

    @property
    def machines(self) -> list[Entity]:
        return [e for e in self.entities if e.role is Role.POWERED_MACHINE]

    @property
    def most_urgent(self) -> Conflict | None:
        """Highest severity per unit of remaining time; unknown-time conflicts rank last."""
        if not self.conflicts:
            return None
        urgency = sorted(
            self.conflicts,
            key=lambda c: (
                c.time_to_event_s is None,
                (c.time_to_event_s or 1e9) - 2.0 * c.severity_hint,
            ),
        )
        return urgency[0]

    def entity(self, ref: str) -> Entity | None:
        return self.entities_by_ref.get(normalise_ref(ref))
