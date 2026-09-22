"""Who chooses the agent's next move.

Two implementations behind one tiny interface, and the difference between them is a honesty
statement rather than an engineering one: :class:`NemotronReasoner` is a language model
planning over the transcript, :class:`ScriptedReasoner` is a deterministic controller over the
same structured state. The scripted one lets the loop, the trace, the ledger, the UI and the
benchmark all run with no API key, and it is labelled as what it is everywhere it appears --
``vigil eval`` refuses to report a mock run as a measurement.

A turn is always exactly one action. Batching actions would save tokens and lose the property
that matters here: every step is taken with the previous step's evidence in hand.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from vigil.agent.actions import (
    ClearAction,
    FinishAction,
    FlagAction,
    PolicyAction,
    RelookAction,
    Turn,
)
from vigil.agent.prompts import (
    AGENT_SYSTEM,
    ASKING,
    BRIEFING_HEADER,
    BRIEFING_STATE,
    REPAIR_INSTRUCTION,
)
from vigil.agent.tools import Observation
from vigil.config import Settings
from vigil.models.risk import HazardClass, Likelihood, Mitigation, RiskAssessment
from vigil.models.scene import SceneUnderstanding
from vigil.nebius import NebiusClient, NebiusError
from vigil.perception.base import PerceptionAnswer, PerceptionError, Usage, extract_json
from vigil.policy.kb import Scored
from vigil.video.source import VideoSource

# A scene names the mechanism of harm; the rulebook is indexed by the same seven classes, so
# the mapping is identity. The three legacy words are kept because a vision model reaching for
# "trip_onto" is describing a real hazard, and dropping it would be worse than accepting a
# synonym. Anything unmapped falls through as free text, which still gets lexical retrieval.
_TAXONOMY = {
    "struck_by": HazardClass.STRUCK_BY,
    "struck_by_reversing_vehicle": HazardClass.STRUCK_BY_REVERSING_VEHICLE,
    "struck_by_falling_object": HazardClass.STRUCK_BY_FALLING_OBJECT,
    "worker_riding_on_forks": HazardClass.WORKER_RIDING_ON_FORKS,
    "slip_trip": HazardClass.SLIP_TRIP,
    "fall_from_height": HazardClass.FALL_FROM_HEIGHT,
    "caught_between": HazardClass.CAUGHT_BETWEEN,
    "trip_onto": HazardClass.SLIP_TRIP,
    "fall_onto": HazardClass.FALL_FROM_HEIGHT,
    "collapse_onto": HazardClass.STRUCK_BY_FALLING_OBJECT,
}


class ReasonerError(RuntimeError):
    """The planner could not produce a usable action, even after one repair attempt."""


@dataclass(frozen=True)
class Briefing:
    """Everything the planner is allowed to know before choosing the next action.

    Rebuilt from scratch each turn rather than accumulated as a message list: the transcript is
    a pure function of the world state, so a turn can be replayed, diffed or cached, and a
    repair attempt cannot corrupt the conversation.
    """

    video: VideoSource
    turn: int
    max_turns: int
    tokens_used: int
    token_budget: int
    log: tuple[Observation, ...] = ()
    scenes: tuple[SceneUnderstanding, ...] = ()
    findings: tuple[RiskAssessment, ...] = ()
    cleared: tuple[str, ...] = ()

    def render(self) -> str:
        """The user-side prompt: header, then the evidence in the order it arrived."""
        parts = [
            BRIEFING_HEADER.format(
                video_id=self.video.video_id,
                camera=self.video.camera_label,
                duration=self.video.duration_s,
                n_clips=len(self.scenes),
                site=self.video.site or "not stated",
                area=self.video.area_type or "not stated",
                lighting=self.video.lighting or "not stated",
                surface=self.video.surface or "not stated",
            )
        ]
        for index, obs in enumerate(self.log, start=1):
            parts.append(f"Observation {index} from {obs.tool}:\n{obs.body}\n")
        parts.append(
            BRIEFING_STATE.format(
                turn=self.turn,
                max_turns=self.max_turns,
                tokens=self.tokens_used,
                budget=self.token_budget,
                flags=len(self.findings),
                clears=len(self.cleared),
            )
        )
        parts.append(ASKING)
        return "\n".join(parts)


class Reasoner(ABC):
    """One method, one action. See the module docstring for why there are two backends."""

    name = "abstract"

    @abstractmethod
    async def plan(self, briefing: Briefing) -> Turn: ...

    @property
    @abstractmethod
    def usage(self) -> Usage:
        """Tokens and latency this planner has spent so far, for the budget guard."""

    async def model_id(self) -> str:
        return self.name

    async def aclose(self) -> None:
        return None


# --------------------------------------------------------------------------- scripted


@dataclass
class _Lead:
    """One thing worth investigating, worked through test -> rule -> decision.

    Two clocks, and keeping them apart is the whole point of this dataclass:

    * ``lead_time_s`` is the largest advance notice any window gave -- a property of *our*
      sampling, and the number the product is judged on.
    * ``contact_s`` is the smallest time-to-contact any window reported -- a property of the
      *world*, and what urgency is judged from.

    One field for both was the bug: reading the warning clock as if it were the countdown meant
    that sampling a clip more finely produced a longer warning and therefore a *lower* risk
    score. Two findings about the same collision disagreed because of where the window edges
    fell, which is not a thing a safety report is allowed to do.
    """

    key: str
    hazard_type: str
    mitigation_key: str
    participants: tuple[str, ...]
    rationale: str
    severity: int
    lead_time_s: float | None
    window: tuple[float, float]
    confidence: float
    contact_s: float | None = None
    ppe: bool = False
    tested: bool = False
    searched: bool = False
    flagged: bool = False
    answer: PerceptionAnswer | None = None
    rules: tuple[Scored, ...] = ()


_PPE_LEADS: dict[str, tuple[str, str]] = {
    "hi_vis": (
        "struck_by",
        "high visibility personal protective equipment is not worn by a worker who "
        "shares the route with a moving vehicle",
    ),
    "hard_hat": (
        "struck_by_falling_object",
        "head protective equipment is not worn where loads are handled overhead",
    ),
    "harness": (
        "fall_from_height",
        "fall arrest protective equipment is not worn while the worker is elevated above the deck",
    ),
    "boots": ("ppe", "protective footwear is not worn where loads and trucks move"),
    "gloves": ("ppe", "protective gloves are not worn at a machine with nip points"),
}

_MITIGATIONS: dict[str, tuple[str, str]] = {
    HazardClass.STRUCK_BY: (
        "Hold the machine and re-establish separation before either party moves again.",
        "floor_supervisor",
    ),
    HazardClass.STRUCK_BY_REVERSING_VEHICLE: (
        "Stop the reversing manoeuvre until a banksman covers the blind quarter.",
        "dock_lead",
    ),
    HazardClass.STRUCK_BY_FALLING_OBJECT: (
        "Bar or restack the load before anyone passes beneath it.",
        "warehouse_lead",
    ),
    HazardClass.WORKER_RIDING_ON_FORKS: (
        "Stop the truck, lower the tines, and bring the rider down before travel resumes.",
        "floor_supervisor",
    ),
    HazardClass.SLIP_TRIP: (
        "Cordon the area, clear the surface, then fix whatever produced it.",
        "housekeeping_lead",
    ),
    HazardClass.FALL_FROM_HEIGHT: (
        "Stop the task until a guardrail or a certified platform is in place.",
        "site_supervisor",
    ),
    HazardClass.CAUGHT_BETWEEN: (
        "Isolate and lock out before anyone reaches into the machine.",
        "maintenance_lead",
    ),
    "ppe": (
        "Supply and require the missing equipment before the task continues.",
        "site_supervisor",
    ),
}
_GENERIC_MITIGATION = (
    "Stop the task until the hazard is controlled and re-assessed.",
    "site_supervisor",
)


_HEADLINE: dict[str, str] = {
    HazardClass.STRUCK_BY: "Struck-by between converging routes",
    HazardClass.STRUCK_BY_REVERSING_VEHICLE: "Pedestrian in the path of a reversing vehicle",
    HazardClass.STRUCK_BY_FALLING_OBJECT: "Anyone below an unsecured overhead load",
    HazardClass.WORKER_RIDING_ON_FORKS: "Worker carried on elevated tines",
    HazardClass.SLIP_TRIP: "Slip or trip on an uncontrolled surface",
    HazardClass.FALL_FROM_HEIGHT: "Fall from height with nothing to hold the person",
    HazardClass.CAUGHT_BETWEEN: "Body part going into a machine's nip point",
    "ppe": "Required protective equipment absent",
}


def _headline(lead: _Lead) -> str:
    """The finding's label: the mechanism of harm, then who is exposed.

    Not the first sentence of the physical account -- ``predicted_event`` carries that prose in
    full, and having both fields quote the same sentence made a report read as if it stuttered.
    The label is what a supervisor scans; the account is what they read.
    """
    label = (
        _HEADLINE.get(lead.mitigation_key) or _HEADLINE.get(lead.hazard_type) or "Hazard exposure"
    )
    return f"{label}: {' and '.join(lead.participants)}"


def _as_hazard_class(value: str) -> HazardClass | None:
    """The taxonomy has seven words; a lead may carry one, or a synonym, or a PPE label."""
    try:
        return HazardClass(value)
    except ValueError:
        return None


def _likelihood(lead: _Lead) -> Likelihood:
    """How fixed the outcome is, judged from the closest observation of the countdown.

    The one place the two clocks are allowed to meet, and it reads ``contact_s`` rather than
    ``lead_time_s`` on purpose. How far ahead *we* noticed is a claim about Vigil, so it belongs
    in the report and the benchmark; how close the parties got is a claim about the world, so it
    belongs in the score. Feeding the warning clock in here made a collision caught at 2.9 s
    rank below the same collision caught at 1.9 s, so sampling the footage more finely *lowered*
    the reported risk, and two runs of one clip disagreed about how dangerous the same seconds
    were depending on where the window edges fell. Reading the minimum is monotone in the safe
    direction: denser coverage can only bring an observation closer to the contact.
    """
    if lead.contact_s is None:
        # Nothing to count down: a standing exposure, such as protective equipment not worn.
        return Likelihood.POSSIBLE if lead.confidence >= 0.6 else Likelihood.REMOTE
    if lead.contact_s <= 1.0:
        # A window that already contains the contact, or one that ends on its heels.
        return Likelihood.IMMINENT
    if lead.contact_s <= 6.0:
        return Likelihood.LIKELY
    if lead.contact_s <= 12.0:
        return Likelihood.POSSIBLE
    return Likelihood.REMOTE


class ScriptedReasoner(Reasoner):
    """Deterministic controller: test each lead, look up its rule, then decide.

    A fixed priority order, not a learned policy. It reads only what the loop hands it --
    conflicts and protective-equipment gaps from the structured scene, plus the perception
    answers it asked for -- so it never sees ground truth, and it passes through the same
    citation gate as a language model: it may only cite what its own search returned.
    """

    name = "scripted"

    def __init__(self) -> None:
        self._leads: dict[str, _Lead] = {}
        self._order: list[str] = []
        self._cleared: set[str] = set()
        self._pending: tuple[str, str] | None = None
        self._seq = 0

    @property
    def usage(self) -> Usage:
        return Usage()

    async def model_id(self) -> str:
        return "scripted-controller (rule-based, not a language model)"

    async def plan(self, briefing: Briefing) -> Turn:
        self._collect(briefing)
        self._harvest(briefing)

        lead = self._next_lead()
        if lead is not None:
            if not lead.tested:
                self._pending = (lead.key, "relook")
                lead.tested = True
                return Turn(
                    thought=(
                        f"{_headline(lead)} Testing the timing before I commit to a severity."
                    ),
                    action=self._test(lead, briefing.video),
                )
            if not lead.searched:
                self._pending = (lead.key, "policy_search")
                lead.searched = True
                return Turn(
                    thought="Held up by the re-sample. Checking which rule this breaches.",
                    action=PolicyAction(query=lead.rationale[:280], hazard_type=lead.hazard_type),
                )
            if not lead.flagged:
                lead.flagged = True
                self._seq += 1
                return Turn(
                    thought=(
                        f"Two parties, one shared point, {self._lead_time_phrase(lead)}. "
                        "That is a finding, not a description."
                        if not lead.ppe
                        else "The equipment is missing and the rule is explicit. Recording it."
                    ),
                    action=self._flag(lead),
                )

        uncleared = next(
            (
                scene
                for scene in briefing.scenes
                if not scene.conflicts and _scene_key(scene) not in self._cleared
            ),
            None,
        )
        if uncleared is not None:
            self._cleared.add(_scene_key(uncleared))
            return Turn(
                thought="This window has no converging pair; saying so is evidence, "
                "silence is not.",
                action=ClearAction(
                    subject=f"Route separation across {uncleared.t0_s:.1f}-{uncleared.t1_s:.1f}s",
                    reason=(uncleared.dynamics_narrative or "No pair shares a point in time here.")[
                        :580
                    ],
                ),
            )

        return Turn(
            thought="Every lead is resolved and further looking cannot change the recommendation.",
            action=FinishAction(summary=_closing(briefing)),
        )

    # ------------------------------------------------------------------ state
    def _collect(self, briefing: Briefing) -> None:
        """Adopt new leads from whatever the loop has just read.

        Conflicts are collected across all windows before protective-equipment gaps, so a live
        trajectory is always worked out ahead of a missing item of kit.
        """
        for scene in briefing.scenes:
            for conflict in scene.conflicts:
                # One key for both the merge and the mitigation: two windows that describe the
                # same mechanism in different words must produce one lead, not two findings.
                hazard_type = _TAXONOMY.get(conflict.kind, "")
                kind = hazard_type or conflict.kind
                self._add(
                    _Lead(
                        key=f"{kind}:{'-'.join(conflict.participants)}",
                        hazard_type=hazard_type,
                        mitigation_key=kind,
                        participants=tuple(conflict.participants),
                        rationale=conflict.rationale,
                        severity=conflict.severity_hint,
                        lead_time_s=conflict.time_to_event_s,
                        contact_s=conflict.time_to_event_s,
                        window=(scene.t0_s, scene.t1_s),
                        confidence=scene.confidence,
                    )
                )
        for scene in briefing.scenes:
            for entity in scene.entities:
                for item in entity.ppe_missing:
                    hazard_type, why = _PPE_LEADS.get(
                        item.lower(), ("ppe", f"{item} protective equipment is not being worn")
                    )
                    self._add(
                        _Lead(
                            key=f"ppe:{entity.ref}:{item.lower()}",
                            hazard_type=hazard_type,
                            mitigation_key="ppe",
                            participants=(entity.ref,),
                            rationale=why,
                            severity=2,
                            lead_time_s=None,
                            window=(scene.t0_s, scene.t1_s),
                            confidence=scene.confidence,
                            ppe=True,
                        )
                    )

    def _add(self, lead: _Lead) -> None:
        """Merge repeated sightings of one lead, keeping both clocks in their own direction.

        The same pair appears in every window that covers them. A later window has less time
        left in it by definition, so the two clocks move apart and each keeps the sighting that
        answers its own question: the largest warning for "how far ahead of the event did we
        first see this", the smallest remaining time for "how close did it actually get". The
        second one is what makes a window that already contains the contact *raise* the assessed
        risk instead of, as before, erasing the warning the earlier window had given.
        """
        existing = self._leads.get(lead.key)
        if existing is None:
            self._leads[lead.key] = lead
            self._order.append(lead.key)
            return
        if existing.lead_time_s is None or (
            lead.lead_time_s is not None and lead.lead_time_s > existing.lead_time_s
        ):
            existing.lead_time_s = lead.lead_time_s
            existing.window = lead.window
        if lead.contact_s is not None and (
            existing.contact_s is None or lead.contact_s < existing.contact_s
        ):
            existing.contact_s = lead.contact_s
        existing.severity = max(existing.severity, lead.severity)
        existing.confidence = max(existing.confidence, lead.confidence)

    def _harvest(self, briefing: Briefing) -> None:
        """Fold the newest observation into the lead that asked for it."""
        if self._pending is None or not briefing.log:
            return
        key, expected = self._pending
        self._pending = None
        last = briefing.log[-1]
        if last.tool != expected:
            return
        lead = self._leads.get(key)
        if lead is None:
            return
        if expected == "relook":
            lead.answer = last.answer
        else:
            lead.rules = last.rules

    def _next_lead(self) -> _Lead | None:
        for key in self._order:
            lead = self._leads[key]
            if not (lead.tested and lead.searched and lead.flagged):
                return lead
        return None

    # ------------------------------------------------------------------ actions
    def _test(self, lead: _Lead, video: VideoSource) -> RelookAction:
        """One narrow question, aimed at the moment that decides it.

        For a trajectory conflict that is the instant the prediction puts the contact at, not
        the instant the evidence was: time-to-contact is measured from the end of the window
        that produced it, so a follow-up that stays inside that window cannot settle it.
        """
        names = " and ".join(lead.participants)
        if lead.ppe:
            question = (
                f"Is {names} wearing {lead.rationale.split(' is ')[0]} at any point "
                "in this window? Say where you can see them best."
            )
            return RelookAction(t0_s=lead.window[0], t1_s=lead.window[1], question=question)

        question = (
            f"Do the routes of {names} reach the same point while both are there? If they do, give "
            "the time of contact and the first thing that would have had to change."
        )
        if lead.lead_time_s is None:
            return RelookAction(t0_s=lead.window[0], t1_s=lead.window[1], question=question)

        t0, t1 = lead.window
        contact = t1 + lead.lead_time_s
        start = max(0.0, contact - 2.5)
        end = min(video.duration_s, contact + 1.5)
        if end <= start:
            start, end = t0, t1
        return RelookAction(t0_s=round(start, 2), t1_s=round(end, 2), question=question)

    def _lead_time_phrase(self, lead: _Lead) -> str:
        return (
            f"{lead.lead_time_s:.1f}s of warning"
            if lead.lead_time_s is not None
            else "no estimable time to contact"
        )

    def _flag(self, lead: _Lead) -> FlagAction:
        # The top two, not the whole page: a finding that cites five rules cites none of them
        # convincingly, and retrieval over a few dozen rules is noisy at the tail.
        citations = [scored.rule.id for scored in lead.rules[:2]]
        floor = max((scored.rule.severity_floor for scored in lead.rules[:2]), default=1)
        severity = min(5, max(lead.severity, floor))
        answer_conf = lead.answer.confidence if lead.answer and lead.answer.hazard_present else 0.0
        confidence = round(
            min(0.92, max(lead.confidence, answer_conf) * (0.85 if citations else 0.7)), 2
        )
        lead_time = lead.lead_time_s
        action_text, owner = _MITIGATIONS.get(lead.mitigation_key, _GENERIC_MITIGATION)
        mitigations = [
            Mitigation(
                action=action_text,
                horizon="immediate",
                owner_role=owner,
                rationale=lead.rationale[:300],
            )
        ]
        if citations:
            mitigations.append(
                Mitigation(
                    action="Record the breach against "
                    + ", ".join(citations)
                    + " and re-audit this route.",
                    horizon="7d",
                    owner_role="safety_officer",
                    rationale="Findings that never reach the audit trail repeat.",
                )
            )
        return FlagAction(
            hazard_id=f"H{self._seq}",
            title=_headline(lead),
            severity=severity,
            likelihood=_likelihood(lead),
            hazard_class=_as_hazard_class(lead.hazard_type),
            predicted_event=lead.rationale.strip()[:600] or "Contact between the parties named.",
            lead_time_s=lead_time,
            entities_involved=list(lead.participants)[:8],
            reasoning=(
                f"Perception reported the interaction; a targeted re-sample of "
                f"{lead.window[0]:.1f}-{lead.window[1]:.1f}s was asked, and "
                + (
                    f"it held the hazard at confidence {answer_conf:.2f}. "
                    if answer_conf
                    else "it added no further hazard signal. "
                )
                + (
                    f"Scored against {', '.join(citations)}."
                    if citations
                    else "No rule was retrieved, so it is recorded uncited and at "
                    "reduced confidence."
                )
            )[:2000],
            rule_ids=citations[:6],
            mitigations=mitigations[:5],
            confidence=confidence,
        )


def _scene_key(scene: SceneUnderstanding) -> str:
    return f"{scene.t0_s:.1f}-{scene.t1_s:.1f}"


def _closing(briefing: Briefing) -> str:
    flags = ", ".join(a.hazard_id for a in briefing.findings) or "none"
    return (
        f"{len(briefing.scenes)} window(s) read across {briefing.video.duration_s:.1f}s of "
        f"{briefing.video.camera_label} footage. Findings: {flags}. "
        f"{len(briefing.cleared)} line(s) of inquiry checked and cleared."
    )


# --------------------------------------------------------------------------- nemotron


class NemotronReasoner(Reasoner):
    """The real planner: a Token Factory text model choosing one action per turn."""

    name = "nebius"

    def __init__(self, settings: Settings, client: NebiusClient | None = None) -> None:
        self.settings = settings
        self._own_client = client is None
        self.client = client or NebiusClient(settings)
        self._model: str | None = None
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._latency_ms = 0

    @property
    def usage(self) -> Usage:
        """Built per call rather than accumulated in place, so nothing mutates a shared model."""
        return Usage(
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            latency_ms=self._latency_ms,
            model=self._model or "unresolved",
        )

    async def model_id(self) -> str:
        if self._model is None:
            try:
                self._model = await self.client.pick("reasoning")
            except NebiusError as exc:
                raise ReasonerError(str(exc)) from exc
        return self._model

    async def plan(self, briefing: Briefing) -> Turn:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": briefing.render()},
        ]
        raw = await self._complete(messages)
        try:
            return _parse_turn(raw)
        except (PerceptionError, ValidationError, ReasonerError) as first:
            # One bounded repair turn. A second failure is a real problem and should surface
            # with both replies attached rather than being retried into silence.
            messages.append({"role": "assistant", "content": raw[:4000]})
            messages.append(
                {"role": "user", "content": REPAIR_INSTRUCTION.format(error=str(first)[:400])}
            )
            retry = await self._complete(messages)
            try:
                return _parse_turn(retry)
            except (PerceptionError, ValidationError, ReasonerError) as second:
                raise ReasonerError(
                    "reply was not a usable action after one repair attempt. "
                    f"First: {str(first)[:200]}. Second: {str(second)[:200]}. "
                    f"Reply was {retry[:400]!r}"
                ) from second

    async def _complete(self, messages: list[dict[str, Any]]) -> str:
        try:
            result = await self.client.chat(
                messages, model=await self.model_id(), kind="reasoning", json_mode=True
            )
        except NebiusError as exc:
            raise ReasonerError(str(exc)) from exc
        self._prompt_tokens += result.prompt_tokens
        self._completion_tokens += result.completion_tokens
        self._latency_ms += result.latency_ms
        return result.text

    async def aclose(self) -> None:
        if self._own_client:
            await self.client.aclose()


def _parse_turn(raw: str) -> Turn:
    """Validate one reply into a Turn, tolerating the two shape errors models actually make."""
    payload = extract_json(raw)
    if "action" not in payload and "tool" in payload:
        # Flattened replies are the common mistake: the model ran out of patience for the
        # wrapper. Re-wrapping costs nothing, where re-asking costs a turn and the budget.
        thought = str(payload.pop("thought", ""))[:1200]
        payload = {"thought": thought, "action": payload}
    if not str(payload.get("thought", "")).strip():
        payload["thought"] = "(the model supplied no reasoning for this move)"
    if not isinstance(payload.get("action"), dict):
        raise ReasonerError(
            f"'action' was {type(payload.get('action')).__name__}, expected an object"
        )
    return Turn.model_validate(payload)


def build_reasoner(settings: Settings, client: NebiusClient | None = None) -> Reasoner:
    """Mirror of :func:`vigil.perception.registry.build_perception` for the thinking layer."""
    backend = settings.resolved_reasoning
    if backend in {"mock", "scripted"}:
        return ScriptedReasoner()
    if backend == "nebius":
        return NemotronReasoner(settings, client)
    raise ReasonerError(
        f"unknown reasoning backend {backend!r}; use mock, nebius or auto. cosmos is a "
        "perception backend: it reads pixels, it does not plan."
    )


__all__ = [
    "Briefing",
    "NemotronReasoner",
    "Reasoner",
    "ReasonerError",
    "ScriptedReasoner",
    "build_reasoner",
]
