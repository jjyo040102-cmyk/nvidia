"""Perception contract: the agent's eyes.

Two operations, and the difference between them is the whole product.

``survey`` reads a clip broadly. ``ask`` answers one narrow question about one narrow
window -- it is how the agent goes *back* to look again at the thing it is suspicious
of, instead of accepting a single first pass over the footage.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from vigil.models.scene import Entity, Role, SceneUnderstanding, normalise_ref

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

_FALSY = frozenset({"", "false", "no", "none", "null", "0"})
"""What a model writes when it means *not a hazard*."""


class Clip(BaseModel):
    """A slice of source video the perception layer may look at."""

    model_config = ConfigDict(extra="forbid")

    video_id: str = Field(min_length=1)
    path: str = Field(min_length=1, description="Filesystem path to the source video.")
    t0_s: float = Field(ge=0.0)
    t1_s: float = Field(ge=0.0)
    camera_label: str = Field(default="cam-1")

    @model_validator(mode="after")
    def _ascending(self) -> Clip:
        if self.t1_s <= self.t0_s:
            raise ValueError(f"clip window must ascend, got {self.t0_s}..{self.t1_s}s")
        return self

    @property
    def duration_s(self) -> float:
        return self.t1_s - self.t0_s

    @property
    def id(self) -> str:
        return f"{self.video_id}@{self.t0_s:.1f}-{self.t1_s:.1f}"


class Usage(BaseModel):
    """Token accounting for one model call, so budgets can be enforced honestly."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)
    model: str = "mock"

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class PerceptionAnswer(BaseModel):
    """Response to a single targeted question about a clip."""

    model_config = ConfigDict(extra="forbid")

    question: str
    answer: str
    hazard_present: bool
    confidence: float = Field(ge=0.0, le=1.0)
    observed_at_s: float | None = Field(
        default=None, ge=0.0, description="Absolute time in the source video."
    )
    entities_mentioned: list[str] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)

    @property
    def verdict(self) -> str:
        return "HAZARD" if self.hazard_present else "CLEAR"


class PerceptionError(RuntimeError):
    """Raised when a backend cannot produce a usable read. Callers decide the fallback."""


class PerceptionBackend(ABC):
    """Implementations must never raise on merely *inconclusive* input -- only on failure."""

    name: str = "abstract"
    supports_video: bool = False

    @abstractmethod
    async def survey(self, clip: Clip) -> tuple[SceneUnderstanding, Usage]: ...

    @abstractmethod
    async def ask(self, clip: Clip, question: str) -> PerceptionAnswer: ...

    async def model_id(self) -> str:
        """What the report should say read the footage.

        The backend name alone is not provenance: "nebius" records which API was called, not
        which model looked at the frames, and the model is the thing a reviewer asks about.
        """
        return self.name

    async def aclose(self) -> None:
        return None


# --------------------------------------------------------------------------- parsing


def extract_json(raw: str) -> dict[str, Any]:
    """Pull a JSON object out of a model reply.

    Models wrap JSON in prose or fences often enough that a strict ``json.loads`` on the
    whole body fails the majority of otherwise-good calls.
    """
    text = raw.strip()
    if not text:
        raise PerceptionError("empty model response")
    fenced = _JSON_FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise PerceptionError(f"no JSON object in response: {text[:200]!r}") from None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise PerceptionError(f"malformed JSON from model: {exc}") from None
    if not isinstance(parsed, dict):
        raise PerceptionError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed


def repair_scene(payload: dict[str, Any]) -> tuple[SceneUnderstanding, list[str]]:
    """Coerce a model payload into a valid scene, recording every guess we made.

    Vision models invent entity handles ("Pedestrian A") in conflicts that they never
    declared. Dropping those conflicts would hide the most dangerous thing in the frame,
    and raising would throw the whole read away, so we declare the missing entity and
    say that we did.
    """
    notes: list[str] = []
    data = json.loads(json.dumps(payload, default=str))

    entities = data.get("entities")
    if not isinstance(entities, list):
        entities = []
        notes.append("entities was not a list; replaced with []")
    data["entities"] = entities

    declared = {
        normalise_ref(e.get("ref")) for e in entities if isinstance(e, dict) and e.get("ref")
    }

    conflicts = data.get("conflicts")
    if not isinstance(conflicts, list):
        conflicts = []
        notes.append("conflicts was not a list; replaced with []")

    salvageable: list[Any] = []
    for conflict in conflicts:
        if not isinstance(conflict, dict):
            notes.append("conflict dropped: it was not an object")
            continue
        participants = conflict.get("participants")
        if not isinstance(participants, list):
            # A sentence where a list should be. Keeping it would fail the whole read; the
            # conflict is dropped and the note says so, so the rest of the frame survives.
            notes.append(
                f"conflict dropped: participants was {type(participants).__name__}, not a list"
            )
            continue
        for raw_ref in participants:
            ref = normalise_ref(raw_ref)
            if ref and ref not in declared:
                entities.append(
                    {
                        "ref": ref,
                        "category": "unspecified",
                        "role": Role.UNKNOWN.value,
                        "location": "inferred from a conflict the model reported",
                    }
                )
                declared.add(ref)
                notes.append(f"auto-declared entity {ref!r} referenced by a conflict")
        salvageable.append(conflict)
    data["conflicts"] = salvageable

    for key, fallback in (("t0_s", 0.0), ("clip_id", "unknown")):
        if key not in data:
            data[key] = fallback
            notes.append(f"{key} missing; defaulted to {fallback!r}")
    if "t1_s" not in data:
        data["t1_s"] = data["t0_s"]
        notes.append("t1_s missing; equalled to t0_s")

    data["repair_notes"] = notes
    try:
        scene = SceneUnderstanding.model_validate(data)
    except ValidationError as exc:
        raise PerceptionError(_describe_validation(exc)) from exc
    return scene, notes


def _describe_validation(exc: ValidationError) -> str:
    parts = [
        f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}" for e in exc.errors()[:6]
    ]
    return "scene failed validation -> " + "; ".join(parts)


def scene_json_schema() -> str:
    """The exact shape we demand from the vision model, as prompt text.

    ``$defs`` stays in. Dropping it to save a few hundred prompt tokens left the model holding
    ``{"items": {"$ref": "#/$defs/Entity"}}`` with no definition anywhere in its context, which
    is a schema that specifies nothing about the one field the whole product parses.
    """
    return json.dumps(SceneUnderstanding.model_json_schema(), separators=(",", ":"), sort_keys=True)


# -------------------------------------------------------------------------------- answering


def clamp_unit(value: object, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(cast("float", value))))
    except (TypeError, ValueError):
        return default


def as_flag(value: object) -> bool:
    """Read a boolean out of a field a language model may have written any way.

    ``"false"`` is truthy in Python, and a vision model that means "nothing is wrong here" and
    writes it as a string would otherwise become a HAZARD verdict -- a false alarm on a system
    whose whole job is to be trusted when it says stop.
    """
    if isinstance(value, str):
        return value.strip().lower() not in _FALSY
    return bool(value)


def answer_from_payload(
    clip: Clip, question: str, payload: dict[str, Any], *, usage: Usage
) -> PerceptionAnswer:
    """One parser for every backend's answer, so the paths cannot drift apart.

    They had: the hosted one clamped ``observed_at_s`` into the window it was asked about and
    the local one ignored the field entirely while also never asking the model for
    ``hazard_present``, which made every re-look on the edge path come back CLEAR by
    construction. A follow-up that cannot report a hazard is worse than no follow-up, because
    it reads like evidence.

    ``observed_at_s`` is clamped rather than trusted: a model that answers with a moment outside
    the window it was shown is wrong about *when*, and an unclamped value would put a sighting
    somewhere the footage does not cover.
    """
    observed = payload.get("observed_at_s")
    try:
        moment = None if observed is None else float(cast("float", observed))
    except (TypeError, ValueError):
        moment = None
    return PerceptionAnswer(
        question=question,
        answer=str(payload.get("answer", "")).strip() or "no answer returned",
        hazard_present=as_flag(payload.get("hazard_present")),
        confidence=clamp_unit(payload.get("confidence"), 0.4),
        observed_at_s=None if moment is None else max(clip.t0_s, min(clip.t1_s, moment)),
        entities_mentioned=[
            str(e) for e in payload.get("entities_mentioned", []) if str(e).strip()
        ][:8],
        usage=usage,
    )


def make_entity(
    ref: str, category: str, *, role: Role = Role.UNKNOWN, location: str = ""
) -> Entity:
    return Entity(ref=ref, category=category, role=role, location=location or "not specified")
