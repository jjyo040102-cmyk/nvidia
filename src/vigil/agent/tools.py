"""The agent's hands: what each action actually does.

Two properties are enforced here rather than requested in the prompt, because a prompt is a
hope and this file is a gate.

**Citations must have been seen.** A ``flag`` can only cite rule ids that a
``policy_search`` already returned in this investigation. A model that invents a plausible
29 CFR subsection produces a report that looks authoritative and is wrong, which is the single
most damaging failure this product can have, so the id is dropped and the drop is logged.

**Surveys are remembered.** Re-reading the same window costs nothing twice, and the observation
says so, which stops a model with a token ceiling from looping on itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from vigil.agent.actions import (
    Action,
    ClearAction,
    FlagAction,
    PolicyAction,
    RelookAction,
    SurveyAction,
)
from vigil.config import Settings
from vigil.models.risk import HazardClass, RiskAssessment, RuleCitation
from vigil.models.scene import SceneUnderstanding, normalise_ref
from vigil.perception.base import Clip, PerceptionAnswer, PerceptionBackend, PerceptionError, Usage
from vigil.policy.kb import Rulebook, Scored
from vigil.video.source import VideoSource

MAX_RULES_SHOWN = 5


@dataclass(frozen=True)
class Observation:
    """What the agent is told happened, plus the objects the loop needs for the ledger.

    ``usage`` is None for anything served from memory: a cached read spent no tokens, and the
    budget guard has to be able to tell the difference or a run looks expensive when it was
    just repetitive.
    """

    tool: str
    summary: str
    body: str
    ok: bool = True
    scene: SceneUnderstanding | None = None
    answer: PerceptionAnswer | None = None
    rules: tuple[Scored, ...] = ()
    assessment: RiskAssessment | None = None
    usage: Usage | None = None


def describe_scene(scene: SceneUnderstanding) -> str:
    """A readable rendering of a structured read.

    Raw JSON is both longer and harder for a model to reason over than eight lines of plain
    observation, and this is the form the transcript keeps.
    """
    lines = [
        f"Window {scene.t0_s:.1f}-{scene.t1_s:.1f}s | {scene.area_type.value} | "
        f"{scene.lighting.value} | surface {scene.surface.value} | "
        f"confidence {scene.confidence:.2f}"
    ]
    for e in scene.entities:
        bits = [f"{e.ref} {e.category} ({e.role.value})", e.location]
        if e.speed_mps is not None:
            bits.append(f"{e.speed_mps:.2f} m/s, {e.kinematics.value}")
        else:
            bits.append(e.kinematics.value)
        if e.ppe_missing:
            bits.append("missing " + "/".join(e.ppe_missing))
        if e.attention:
            bits.append(e.attention)
        lines.append("- " + "; ".join(bits))
    for c in scene.conflicts:
        tte = (
            "not estimable" if c.time_to_event_s is None else f"{c.time_to_event_s:.1f}s to contact"
        )
        lines.append(
            f"- CONFLICT {'+'.join(c.participants)}: {c.kind}, {tte}, "
            f"severity {c.severity_hint}/5. {c.rationale}"
        )
    for v in scene.visibility_limits:
        blind = " (camera blind)" if v.camera_blind else ""
        lines.append(f"- NOT VISIBLE: {v.occluder} hides {v.hidden_zone}{blind}")
    for m in scene.key_moments[:6]:
        lines.append(f"- AT {m.t_s:.1f}s: {m.description}")
    if scene.dynamics_narrative:
        lines.append(f"- NARRATIVE: {scene.dynamics_narrative}")
    for note in scene.repair_notes:
        lines.append(f"- PARSER NOTE: {note}")
    return "\n".join(lines)


def _clip_key(clip: Clip) -> tuple[float, float]:
    return (round(clip.t0_s, 2), round(clip.t1_s, 2))


class ToolKit:
    def __init__(
        self,
        *,
        perception: PerceptionBackend,
        rulebook: Rulebook,
        video: VideoSource,
        settings: Settings,
    ) -> None:
        self.perception = perception
        self.rulebook = rulebook
        self.video = video
        self.settings = settings
        self._scenes: dict[tuple[float, float], SceneUnderstanding] = {}
        self._answers: dict[tuple[float, float, str], PerceptionAnswer] = {}
        # rule id -> the score it was retrieved at. Doubles as the citation allow-list: a
        # flag can only cite ids that appear here, so this dict is the run's evidence trail.
        self.surfaced: dict[str, float] = {}
        self.errors: list[str] = []

    @property
    def scenes(self) -> list[SceneUnderstanding]:
        return list(self._scenes.values())

    def _window_for(self, entities: list[str]) -> tuple[float, float] | None:
        """Where in the footage this finding was seen, resolved from the reads themselves.

        The window that gave the *longest* warning about this pair, not the narrowest one that
        mentions them. The model is not trusted with it because the timeline overlay and the
        benchmark's lead-time metric both hang off this number, and a reply is a poor place to
        verify one.

        Narrowest was the first choice here, and it quietly broke the report: a bullet that read
        "seen in window 5.2-7.4s, warning 2.9s ahead" named an end time from one read and a lead
        from another, so the two numbers in the same line pointed at different contacts. Taking
        the longest-warning window keeps ``window end + lead`` at one instant, which is the only
        reading a safety officer can check.
        """
        wanted = {normalise_ref(e) for e in entities if str(e).strip()}
        if not wanted:
            return None
        hits = [
            scene
            for scene in self._scenes.values()
            if any(wanted <= set(conflict.participants) for conflict in scene.conflicts)
        ]
        if not hits:
            return None

        def warning_s(scene: SceneUnderstanding) -> float:
            return max(
                (
                    conflict.time_to_event_s or 0.0
                    for conflict in scene.conflicts
                    if wanted <= set(conflict.participants)
                ),
                default=0.0,
            )

        best = min(hits, key=lambda s: (-warning_s(s), s.t1_s - s.t0_s, s.t0_s))
        return (best.t0_s, best.t1_s)

    async def execute(self, action: Action) -> Observation:
        if isinstance(action, SurveyAction):
            return await self._survey(action.t0_s, action.t1_s)
        if isinstance(action, RelookAction):
            return await self._relook(action)
        if isinstance(action, PolicyAction):
            return self._policy(action)
        if isinstance(action, FlagAction):
            return self._flag(action)
        if isinstance(action, ClearAction):
            return Observation(
                tool="clear",
                summary=f"Ruled out: {action.subject}",
                body=(
                    f"Noted. {action.subject} is recorded as checked and cleared, with your reason."
                ),
            )
        raise TypeError(f"action {type(action).__name__} is executed by the loop, not the toolkit")

    async def _survey(self, t0_s: float, t1_s: float) -> Observation:
        try:
            clip = self.video.clip(t0_s, t1_s)
        except ValueError as exc:
            return Observation(tool="survey", summary="Window unusable", body=str(exc), ok=False)
        key = _clip_key(clip)
        cached = self._scenes.get(key)
        if cached is not None:
            return Observation(
                tool="survey",
                summary=f"Re-read {clip.t0_s:.1f}-{clip.t1_s:.1f}s from memory",
                body="You already read this window; no second call was made. Here it is again:\n"
                + describe_scene(cached),
                scene=cached,
            )
        try:
            scene, usage = await self.perception.survey(clip)
        except PerceptionError as exc:
            self.errors.append(f"survey: {exc}")
            return Observation(
                tool="survey",
                summary="Perception failed",
                body=f"The vision layer could not read that window: {exc}",
                ok=False,
            )
        self._scenes[key] = scene
        return Observation(
            tool="survey",
            summary=f"Read {clip.t0_s:.1f}-{clip.t1_s:.1f}s",
            body=describe_scene(scene),
            scene=scene,
            usage=usage,
        )

    async def _relook(self, action: RelookAction) -> Observation:
        try:
            clip = self.video.clip(action.t0_s, action.t1_s)
        except ValueError as exc:
            return Observation(tool="relook", summary="Window unusable", body=str(exc), ok=False)
        key = (*_clip_key(clip), action.question.strip().lower())
        cached = self._answers.get(key)
        if cached is not None:
            return self._answer_observation(cached, repeat=True)
        try:
            answer = await self.perception.ask(clip, action.question)
        except PerceptionError as exc:
            self.errors.append(f"relook: {exc}")
            return Observation(
                tool="relook",
                summary="Perception failed",
                body=f"The vision layer could not answer that: {exc}",
                ok=False,
            )
        self._answers[key] = answer
        return self._answer_observation(answer, repeat=False)

    @staticmethod
    def _answer_observation(answer: PerceptionAnswer, *, repeat: bool) -> Observation:
        head = "You already asked this; no second call was made." if repeat else ""
        body = "\n".join(
            part
            for part in (
                head,
                answer.answer,
                f"hazard_present={str(answer.hazard_present).lower()} "
                f"confidence={answer.confidence:.2f} "
                + (
                    f"observed_at={answer.observed_at_s:.1f}s"
                    if answer.observed_at_s is not None
                    else "observed_at=n/a"
                ),
                f"entities mentioned: {', '.join(answer.entities_mentioned)}"
                if answer.entities_mentioned
                else "",
            )
            if part
        )
        return Observation(
            tool="relook",
            summary=(
                f"Answered ({'yes' if answer.hazard_present else 'no'} hazard, "
                f"{answer.confidence:.2f})"
            ),
            body=body,
            answer=answer,
            usage=None if repeat else answer.usage,
        )

    def _policy(self, action: PolicyAction) -> Observation:
        found = (
            self.rulebook.for_hazard(action.hazard_type, k=MAX_RULES_SHOWN, note=action.query)
            if action.hazard_type
            else self.rulebook.search(action.query, k=MAX_RULES_SHOWN)
        )
        if not found:
            return Observation(
                tool="policy_search",
                summary="No rule matched",
                body=(
                    "Nothing in the rulebook matches that wording. Try the hazard type "
                    "(struck_by, slip_trip, fall_from_height, caught_between) or plainer terms."
                ),
            )
        for scored in found:
            rule_id = scored.rule.id
            self.surfaced[rule_id] = max(self.surfaced.get(rule_id, 0.0), scored.score)
        lines = [
            f"[{s.rule.id}] {s.rule.title} -- {s.rule.source} "
            f"(relevance {s.score:.2f}, severity floor {s.rule.severity_floor})\n  {s.rule.text}"
            for s in found
        ]
        return Observation(
            tool="policy_search",
            summary=f"{len(found)} rule(s), best: {found[0].rule.title}",
            body="Cite these by id:\n" + "\n".join(lines),
            rules=tuple(found),
        )

    def _flag(self, action: FlagAction) -> Observation:
        citations: list[RuleCitation] = []
        floors: list[int] = []
        rejected: list[str] = []
        cited_hazards: set[HazardClass] = set()
        for raw in action.rule_ids:
            rule_id = raw.strip()
            rule = self.rulebook.get(rule_id)
            if rule is None:
                rejected.append(rule_id)
                continue
            if rule_id not in self.surfaced:
                rejected.append(f"{rule_id} (never retrieved this run)")
                continue
            citations.append(
                RuleCitation(
                    rule_id=rule.id,
                    source=rule.source,
                    text=rule.text,
                    url=rule.url,
                    relevance=self.surfaced[rule_id],
                )
            )
            cited_hazards.update(rule.hazards)
            floors.append(rule.severity_floor)
        assessment = action.to_assessment(
            citations, clip_window=self._window_for(action.entities_involved)
        )
        if assessment.hazard_class is None and len(cited_hazards) == 1:
            # The rule the agent was actually shown governs exactly one mechanism. That names
            # the finding better than the agent's own vocabulary, and it is retrieved evidence
            # rather than a guess -- so the taxonomy column is rarely empty in a real report.
            assessment = assessment.model_copy(update={"hazard_class": next(iter(cited_hazards))})

        notes: list[str] = []
        if rejected:
            notes.append("Ignored citation(s) you were not shown: " + ", ".join(rejected))
        if floors:
            floor = max(floors)
            if floor > assessment.severity:
                notes.append(
                    f"Severity raised {assessment.severity} -> {floor} by the floor on "
                    f"{', '.join(c.rule_id for c in citations)}."
                )
                assessment = assessment.model_copy(update={"severity": floor})
        body = (
            f"Recorded {assessment.hazard_id}: {assessment.title} -- risk "
            f"{assessment.risk_score}/20, {assessment.disposition.value}."
        )
        if notes:
            body += "\n" + "\n".join(notes)
        return Observation(
            tool="flag",
            summary=(
                f"{assessment.hazard_id} {assessment.disposition.value} "
                f"(risk {assessment.risk_score})"
            ),
            body=body,
            assessment=assessment,
        )


def scene_payload(scene: SceneUnderstanding) -> dict[str, Any]:
    """Structured form kept for the UI and the benchmark; the transcript uses the digest."""
    payload: dict[str, Any] = json.loads(scene.model_dump_json())
    return payload


__all__ = [
    "MAX_RULES_SHOWN",
    "Observation",
    "ToolKit",
    "describe_scene",
    "scene_payload",
]
