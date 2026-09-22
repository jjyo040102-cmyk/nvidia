"""Scripted perception for authored clips -- the backend that lets Vigil run with no key.

This is an oracle, not a model: for the eight authored clips it derives a scene straight from
the scenario definition, so the UI, the agent loop and the end-to-end tests all have something
coherent to chew on before a Nebius key exists. It is therefore **not** evidence of accuracy.
``vigil eval`` refuses to run against it unless you pass ``--allow-mock``, and every report
carries the backend name so a mock number can never be mistaken for a measurement.
"""

from __future__ import annotations

import asyncio
import re
from enum import Enum
from pathlib import Path
from typing import TypeVar

from vigil.config import Settings
from vigil.models.scene import (
    AreaType,
    Conflict,
    Entity,
    KeyMoment,
    Kinematics,
    Lighting,
    Role,
    SceneUnderstanding,
    SurfaceCondition,
    VisibilityLimit,
    normalise_ref,
)
from vigil.perception.base import Clip, PerceptionAnswer, PerceptionBackend, PerceptionError, Usage
from vigil.video.scenarios import all_scenarios
from vigil.video.synth import WIDTH, Scenario, Track

EnumT = TypeVar("EnumT", bound=Enum)

_ROLE_MAP = {
    "vulnerable_party": Role.VULNERABLE_PARTY,
    "powered_machine": Role.POWERED_MACHINE,
    "reverse": Role.POWERED_MACHINE,
    "reach": Role.VULNERABLE_PARTY,
    "bystander": Role.BYSTANDER,
    "operator": Role.OPERATOR,
}

_ACTOR_KINDS = {"person", "forklift", "truck"}


def _entity_tracks(scenario: Scenario) -> list[Track]:
    """Tracks the camera would genuinely see as things: movers, plus whatever a hazard names.

    A spill and a ladder are not movers, but "P1 trips *onto SP1*" is a conflict between two
    declared entities, and the scene schema refuses a conflict that references an undeclared
    one. Excluding hazard objects would silently delete slips and falls from the dataset.
    """
    named = {ref for hazard in scenario.hazards for ref in hazard.entities}
    return [track for track in scenario.tracks if track.kind in _ACTOR_KINDS or track.ref in named]


def _by_id() -> dict[str, Scenario]:
    return {scenario.id: scenario for scenario in all_scenarios()}


def find_scenario(clip: Clip) -> Scenario | None:
    """Authored clips are keyed by id, or by the mp4 filename when the id is generic."""
    table = _by_id()
    for key in (clip.video_id, Path(clip.path).stem):
        if key in table:
            return table[key]
    return None


def _kinematics(track_kind: str, speed: float, role: str) -> Kinematics:
    if speed < 0.06:
        return Kinematics.STATIONARY
    if track_kind == "person":
        return Kinematics.RUNNING if speed > 1.9 else Kinematics.WALKING
    return Kinematics.REVERSING if "reverse" in role else Kinematics.DRIVING


def _where(x: float, y: float) -> str:
    """Site coordinates to the words a camera operator would use."""
    sx = WIDTH / 2 + 470 * x / max(0.35, y)
    lateral = (
        "far left"
        if sx < WIDTH * 0.22
        else "left"
        if sx < WIDTH * 0.42
        else "centre"
        if sx < WIDTH * 0.58
        else "right"
        if sx < WIDTH * 0.78
        else "far right"
    )
    depth = "near the camera" if y < 3.4 else "mid-frame" if y < 5.4 else "deep in the bay"
    return f"{lateral}, {depth}, about {y:.1f} m out"


def _attention(track: Track) -> str:
    if track.carries == "rider":
        return "carrying a person on the elevated tines"
    if track.carries:
        return "occupied with a carried load"
    return "not determinable from these frames"


def scene_from_scenario(scenario: Scenario, clip: Clip) -> SceneUnderstanding:
    mid = (clip.t0_s + clip.t1_s) / 2
    entities: list[Entity] = []
    for track in _entity_tracks(scenario):
        if track.path[0][0] > clip.t1_s:
            continue  # not on screen yet when this window ends
        x, y, _ = track.state(mid)
        speed = track.speed_mps(mid)
        role = _ROLE_MAP.get(track.role, Role.UNKNOWN)
        entities.append(
            Entity(
                ref=track.ref,
                category=track.category or track.kind,
                role=role,
                location=_where(x, y),
                kinematics=_kinematics(track.kind, speed, track.role),
                speed_mps=round(speed, 2) if speed > 0.02 else None,
                ppe_missing=list(track.ppe_missing),
                attention=_attention(track),
            )
        )

    conflicts: list[Conflict] = []
    moments: list[KeyMoment] = []
    for hazard in scenario.hazards:
        if hazard.t_impact < clip.t0_s or hazard.t_start > clip.t1_s:
            continue
        participants = [
            p for p in hazard.entities if any(normalise_ref(p) == e.ref for e in entities)
        ]
        if len(participants) < 2:
            participants = [e.ref for e in entities[:2]]
        if len(participants) >= 2:
            conflicts.append(
                Conflict(
                    participants=participants,
                    kind=hazard.type,
                    time_to_event_s=max(0.0, round(hazard.t_impact - clip.t1_s, 2)),
                    severity_hint=hazard.severity,
                    rationale=hazard.description,
                )
            )
        moments.append(
            KeyMoment(
                t_s=hazard.t_start,
                description=f"{hazard.type} becomes anticipatable: {hazard.description}",
            )
        )
        moments.append(
            KeyMoment(
                t_s=hazard.t_impact, description="Contact would occur here if nothing changed."
            )
        )

    limits: list[VisibilityLimit] = []
    for track in scenario.tracks:
        if track.kind == "pillar":
            x, y, _ = track.state(mid)
            limits.append(
                VisibilityLimit(
                    occluder=f"structural pillar {track.ref}",
                    hidden_zone=(
                        f"the crossing point behind the pillar at {_where(x, y)}, "
                        "invisible to the operator until roughly 1.5 m"
                    ),
                    camera_blind=False,
                )
            )
        elif track.kind == "rack":
            x, y, _ = track.state(mid)
            limits.append(
                VisibilityLimit(
                    occluder=f"storage rack {track.ref}",
                    hidden_zone=f"aisle mouth at {_where(x, y)}",
                    camera_blind=y > 6.0,
                )
            )

    narrative = _narrative(scenario, clip, entities)
    return SceneUnderstanding(
        clip_id=clip.id,
        t0_s=clip.t0_s,
        t1_s=clip.t1_s,
        area_type=_enum(AreaType, scenario.area_type, AreaType.OTHER),
        lighting=_enum(Lighting, scenario.lighting, Lighting.UNKNOWN),
        surface=_enum(SurfaceCondition, scenario.surface, SurfaceCondition.UNKNOWN),
        congestion=sum(1 for e in entities if e.kinematics is not Kinematics.STATIONARY),
        entities=entities,
        conflicts=conflicts,
        visibility_limits=limits,
        key_moments=sorted(moments, key=lambda m: m.t_s),
        dynamics_narrative=narrative,
        confidence=0.85 if scenario.tracks else 0.2,
    )


def _enum(cls: type[EnumT], value: str, default: EnumT) -> EnumT:
    try:
        return cls(value)
    except ValueError:
        return default


def _narrative(scenario: Scenario, clip: Clip, entities: list[Entity]) -> str:
    movers = [e for e in entities if e.kinematics is not Kinematics.STATIONARY]
    if not movers:
        return "Nothing is in motion inside this window."
    hazard = next((h for h in scenario.hazards if h.t_start <= clip.t1_s), None)
    if hazard is None:
        return (
            f"{', '.join(e.ref for e in movers)} are moving on separate routes with no shared "
            "point in the window; nothing converges."
        )
    return (
        f"{', '.join(hazard.entities)} are converging. {hazard.description} "
        "The window in which that could still have been prevented opened at "
        f"{hazard.t_start:.1f}s."
    )


class MockPerception(PerceptionBackend):
    name = "mock"
    supports_video = False

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def model_id(self) -> str:
        return "scripted reads from the authored scenarios (not a model)"

    async def _pause(self) -> None:
        if self.settings.mock_latency_ms:
            await asyncio.sleep(self.settings.mock_latency_ms / 1000)

    async def survey(self, clip: Clip) -> tuple[SceneUnderstanding, Usage]:
        await self._pause()
        scenario = find_scenario(clip)
        if scenario is None:
            raise PerceptionError(
                f"mock has no fixture for {clip.video_id!r}. Use an authored clip "
                f"({', '.join(sorted(_by_id()))}) or set VIGIL_PERCEPTION_BACKEND=nebius."
            )
        scene = scene_from_scenario(scenario, clip)
        return scene, Usage(model="mock", latency_ms=self.settings.mock_latency_ms)

    async def ask(self, clip: Clip, question: str) -> PerceptionAnswer:
        await self._pause()
        scenario = find_scenario(clip)
        if scenario is None:
            raise PerceptionError(f"mock has no fixture for {clip.video_id!r}")
        return answer_from_scenario(scenario, clip, question)


def answer_from_scenario(scenario: Scenario, clip: Clip, question: str) -> PerceptionAnswer:
    """Answers a follow-up from the authored truth, matching on the question's vocabulary."""
    low = question.lower()
    for hazard in scenario.hazards:
        mentions_hazard = any(
            token in low
            for token in re.split(r"[^a-z0-9]+", hazard.type.replace("_", " "))
            if len(token) > 3
        ) or any(ref.lower() in low for ref in hazard.entities)
        if not mentions_hazard:
            continue
        window_hits = clip.t0_s - 1.0 <= hazard.t_impact <= clip.t1_s + 1.0
        if not window_hits and not clip.t0_s <= hazard.t_start <= clip.t1_s:
            continue
        return PerceptionAnswer(
            question=question,
            answer=hazard.description,
            hazard_present=True,
            confidence=0.86,
            observed_at_s=min(max(hazard.t_start, clip.t0_s), clip.t1_s),
            entities_mentioned=list(hazard.entities),
            usage=Usage(model="mock"),
        )
    refs = [t.ref for t in scenario.tracks if t.kind in _ACTOR_KINDS]
    return PerceptionAnswer(
        question=question,
        answer=(
            "Within this window the routes stay separated and no converging pair is visible. "
            f"Entities present: {', '.join(refs) if refs else 'none'}."
        ),
        hazard_present=False,
        confidence=0.72,
        observed_at_s=None,
        entities_mentioned=refs[:4],
        usage=Usage(model="mock"),
    )
