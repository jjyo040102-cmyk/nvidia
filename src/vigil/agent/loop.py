"""The investigation itself: read, doubt, go back, decide, and account for every step.

The loop is deliberately small. It opens the footage, then hands over: the reasoner picks each
action, the toolkit executes it, and the result is appended to the transcript the next turn is
planned from. Nothing here decides what the agent should be suspicious of -- if this file
contained hazard-detection rules, the "agent" would be a for-loop with a JSON habit.

Two things it does own, because they are gates rather than judgements:

* **Budgets.** The token ceiling is checked before every planning call, and the client refuses
  any call that would cross the USD ceiling, so a runaway loop cannot spend a trial credit.
* **The audit trail.** Every thought, call, observation and decision is written to the trace
  with its duration, and the trace ships inside the report. A judge reading a report can see
  what was looked at, in which order, and what it cost.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from vigil.agent.actions import Action, ClearAction, FinishAction, SurveyAction
from vigil.agent.reasoner import Briefing, Reasoner, ReasonerError, build_reasoner
from vigil.agent.tools import Observation, ToolKit
from vigil.config import Settings
from vigil.models.report import IncidentReport, ModelProvenance, TimelineEntry
from vigil.models.risk import (
    Hypothesis,
    HypothesisStatus,
    Mitigation,
    RiskAssessment,
    RiskLedger,
    RuleCitation,
)
from vigil.models.scene import SceneUnderstanding
from vigil.models.trace import InvestigationTrace, TraceKind, TraceStep
from vigil.nebius import BudgetExceeded, NebiusClient, Spend
from vigil.perception.base import PerceptionAnswer, PerceptionBackend, Usage
from vigil.perception.registry import build_perception
from vigil.policy.kb import Rulebook, load_rulebook
from vigil.video.source import VideoSource

STOPPED_NO_BUDGET = "the token ceiling was reached before the agent called finish"
STOPPED_NO_TURNS = "the turn limit was reached before the agent called finish"
MAX_TIMELINE_ENTRIES = 40


@dataclass(frozen=True)
class Outcome:
    """Everything one investigation produced, including what a report does not carry."""

    ledger: RiskLedger
    trace: InvestigationTrace
    report: IncidentReport
    scenes: tuple[SceneUnderstanding, ...] = ()
    answers: tuple[PerceptionAnswer, ...] = ()
    spend: dict[str, Any] | None = None
    stopped: str = ""

    @property
    def findings(self) -> tuple[RiskAssessment, ...]:
        return tuple(self.ledger.assessments)

    @property
    def lead_time_s(self) -> float | None:
        return self.ledger.max_lead_time_s


class Investigator:
    """One video, one investigation, one report."""

    def __init__(
        self,
        *,
        video: VideoSource,
        kit: ToolKit,
        reasoner: Reasoner,
        settings: Settings,
        spend: Spend | None = None,
        generated_at: datetime | None = None,
    ) -> None:
        self.video = video
        self.kit = kit
        self.reasoner = reasoner
        self.settings = settings
        self.spend = spend or Spend()
        self.generated_at = generated_at or datetime.now(UTC)
        self._vision_prompt = 0
        self._vision_completion = 0
        self._vision_latency = 0

    # ------------------------------------------------------------------ public
    async def run(self) -> Outcome:
        started = time.perf_counter()
        trace = InvestigationTrace(video_id=self.video.video_id)
        log: list[Observation] = []
        findings: dict[str, RiskAssessment] = {}
        hypotheses: dict[str, Hypothesis] = {}
        cleared: list[str] = []
        summary = ""
        stopped = ""

        windows = self.video.scan(
            clip_seconds=self.settings.clip_seconds,
            max_clips=self.settings.max_clips_per_video,
            overlap=self.settings.clip_overlap,
        )
        _record(
            trace,
            TraceKind.THOUGHT,
            f"Opening {self.video.video_id}",
            detail=(
                f"Reading {len(windows)} overlapping window(s) that cover the whole "
                "recording before forming any opinion, so the first judgement is "
                "made on all of the footage."
            ),
        )
        for clip in windows:
            if self._over_budget():
                stopped = STOPPED_NO_BUDGET
                _record(
                    trace,
                    TraceKind.ERROR,
                    "Opening scan stopped by the token budget",
                    detail=stopped,
                )
                break
            try:
                obs = await self._act(trace, log, SurveyAction(t0_s=clip.t0_s, t1_s=clip.t1_s))
            except BudgetExceeded as exc:
                stopped = f"spend ceiling reached during the opening scan: {exc}"
                _record(
                    trace, TraceKind.ERROR, "Stopped by the spend ceiling", detail=stopped[:2000]
                )
                break
            if not obs.ok:
                stopped = "perception could not read the footage"

        iterations = 0
        for turn_index in range(1, self.settings.max_iterations + 1):
            if self._over_budget():
                stopped = STOPPED_NO_BUDGET
                _record(trace, TraceKind.ERROR, "Stopped by the token budget", detail=stopped)
                break
            # Counted after the guard, because the report publishes this as "N planning
            # turn(s)": a turn the budget stopped before the planner was asked never happened.
            iterations = turn_index
            try:
                turn = await self.reasoner.plan(self._briefing(turn_index, log, findings, cleared))
            except BudgetExceeded as exc:
                stopped = f"spend ceiling reached: {exc}"
                _record(
                    trace, TraceKind.ERROR, "Stopped by the spend ceiling", detail=stopped[:2000]
                )
                break
            except ReasonerError as exc:
                stopped = f"the planner produced no usable action: {exc}"
                _record(trace, TraceKind.ERROR, "Planning failed", detail=stopped[:2000])
                break

            _record(trace, TraceKind.THOUGHT, _clause(turn.thought), detail=turn.thought)
            action = turn.action
            if isinstance(action, FinishAction):
                summary = action.summary
                _record(trace, TraceKind.DECISION, "Closed the investigation", detail=summary)
                break

            try:
                obs = await self._act(trace, log, action)
            except BudgetExceeded as exc:
                stopped = f"spend ceiling reached: {exc}"
                _record(
                    trace, TraceKind.ERROR, "Stopped by the spend ceiling", detail=stopped[:2000]
                )
                break

            if obs.assessment is not None:
                prior = findings.get(obs.assessment.hazard_id)
                if prior is None or obs.assessment.risk_score > prior.risk_score:
                    findings[obs.assessment.hazard_id] = obs.assessment
                _record(
                    trace,
                    TraceKind.DECISION,
                    f"Flagged {obs.assessment.hazard_id}",
                    detail=obs.body,
                )
            if isinstance(action, ClearAction):
                cleared.append(action.subject)
                _record(
                    trace, TraceKind.DECISION, f"Cleared: {action.subject}", detail=action.reason
                )
            if obs.answer is not None and obs.answer.question not in hypotheses:
                hypotheses[obs.answer.question] = _hypothesis(len(hypotheses) + 1, obs.answer)
        else:
            stopped = stopped or STOPPED_NO_TURNS

        ledger = RiskLedger(
            video_id=self.video.video_id,
            assessments=_ranked(findings),
            hypotheses=list(hypotheses.values()),
            summary=summary or _closing_fallback(findings, cleared),
            cleared=cleared,
        )
        total = self._usage()
        trace.iterations = iterations
        trace.prompt_tokens = total.prompt_tokens
        trace.completion_tokens = total.completion_tokens
        trace.wall_clock_ms = int((time.perf_counter() - started) * 1000)

        answers = tuple(obs.answer for obs in log if obs.answer is not None)
        report = await self._build_report(ledger, trace, answers, stopped)
        return Outcome(
            ledger=ledger,
            trace=trace,
            report=report,
            scenes=tuple(self.kit.scenes),
            answers=answers,
            spend=self.spend.as_dict(),
            stopped=stopped,
        )

    # ------------------------------------------------------------------ accounting
    @property
    def tokens_spent(self) -> int:
        return self._usage().total_tokens

    def _usage(self) -> Usage:
        """Both layers' tokens in one place: the count shown to the model in the transcript and
        the count published in the report's provenance come from here.
        """
        reasoner = self.reasoner.usage
        return Usage(
            prompt_tokens=self._vision_prompt + reasoner.prompt_tokens,
            completion_tokens=self._vision_completion + reasoner.completion_tokens,
            latency_ms=self._vision_latency + reasoner.latency_ms,
            model=reasoner.model,
        )

    def _over_budget(self) -> bool:
        return self.tokens_spent >= self.settings.token_budget

    async def _act(
        self, trace: InvestigationTrace, log: list[Observation], action: Action
    ) -> Observation:
        args = action.model_dump(mode="json", exclude={"tool"})
        _record(
            trace,
            TraceKind.TOOL_CALL,
            f"{action.tool}: {_args_line(args)}",
            tool=action.tool,
            args=args,
        )
        started = time.perf_counter()
        obs = await self.kit.execute(action)
        _record(
            trace,
            TraceKind.OBSERVATION if obs.ok else TraceKind.ERROR,
            obs.summary,
            tool=action.tool,
            detail=obs.body,
            duration_ms=int((time.perf_counter() - started) * 1000),
            result_summary=obs.summary,
        )
        log.append(obs)
        if obs.usage is not None:
            self._vision_prompt += obs.usage.prompt_tokens
            self._vision_completion += obs.usage.completion_tokens
            self._vision_latency += obs.usage.latency_ms
        return obs

    def _briefing(
        self,
        turn_index: int,
        log: list[Observation],
        findings: dict[str, RiskAssessment],
        cleared: list[str],
    ) -> Briefing:
        return Briefing(
            video=self.video,
            turn=turn_index,
            max_turns=self.settings.max_iterations,
            tokens_used=self.tokens_spent,
            token_budget=self.settings.token_budget,
            log=tuple(log),
            scenes=tuple(self.kit.scenes),
            findings=tuple(findings.values()),
            cleared=tuple(cleared),
        )

    # ------------------------------------------------------------------ report
    async def _build_report(
        self,
        ledger: RiskLedger,
        trace: InvestigationTrace,
        answers: tuple[PerceptionAnswer, ...],
        stopped: str,
    ) -> IncidentReport:
        reasoning_model = await self.reasoner.model_id()
        backend = self.kit.perception.name
        vision_model = await self.kit.perception.model_id()
        total = self._usage()
        worst = ledger.top_risk
        if worst is None:
            headline = f"No finding met the reporting bar on {self.video.camera_label}"
        else:
            ahead = (
                "" if worst.lead_time_s is None else f", {worst.lead_time_s:.1f}s ahead of contact"
            )
            verdict = f"{worst.disposition.value.replace('_', ' ')} ({worst.risk_score}/20)"
            headline = f"{worst.title} -- {verdict}{ahead}"
        return IncidentReport(
            report_id=_report_id(self.video.video_id, self.generated_at),
            video_id=self.video.video_id,
            generated_at=self.generated_at,
            headline=headline,
            narrative=_narrative(
                self.video, ledger, self.kit.scenes, stopped, backend, reasoning_model
            ),
            timeline=_timeline(ledger, self.kit.scenes, answers),
            assessments=ledger.assessments,
            citations=_citations(ledger.assessments),
            actions=_actions(ledger.assessments),
            provenance=ModelProvenance(
                perception_backend=backend,
                vision_model=vision_model,
                reasoning_model=reasoning_model,
                prompt_tokens=total.prompt_tokens,
                completion_tokens=total.completion_tokens,
                wall_clock_ms=trace.wall_clock_ms,
                video_hours_analysed=round(self.video.duration_s / 3600.0, 8),
            ),
            trace=trace,
        )


# --------------------------------------------------------------------------- helpers


def _record(trace: InvestigationTrace, kind: TraceKind, title: str, **kwargs: Any) -> TraceStep:
    step = TraceStep(index=len(trace.steps), kind=kind, title=title[:200], **kwargs)
    trace.add(step)
    return step


def _clause(text: str) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= 90 else clean[:87].rstrip() + "..."


def _args_line(args: dict[str, Any]) -> str:
    parts = []
    for key, value in args.items():
        parts.append(
            f"{key}={value}" if isinstance(value, (int, float)) else f"{key}={str(value)[:40]}"
        )
        if len(parts) == 2:
            break
    return " ".join(parts) or "no arguments"


def _hypothesis(index: int, answer: PerceptionAnswer) -> Hypothesis:
    """The agent's own questions, kept as a falsifiable record rather than a chat log."""
    if answer.hazard_present and answer.confidence >= 0.5:
        status = HypothesisStatus.SUPPORTED
    elif not answer.hazard_present and answer.confidence >= 0.5:
        status = HypothesisStatus.REJECTED
    else:
        status = HypothesisStatus.OPEN
    return Hypothesis(
        id=f"Q{index}",
        statement=answer.question,
        status=status,
        evidence=[answer.answer],
        asked_of_perception=[answer.question],
    )


def _ranked(findings: dict[str, RiskAssessment]) -> list[RiskAssessment]:
    return sorted(findings.values(), key=lambda a: (-a.risk_score, a.hazard_id))


def _closing_fallback(findings: dict[str, RiskAssessment], cleared: list[str]) -> str:
    flags = ", ".join(sorted(findings)) or "none"
    return (
        f"The investigation stopped before the agent called finish. Findings held: {flags}; "
        f"lines of inquiry checked and cleared: {len(cleared)}."
    )


def _report_id(video_id: str, when: datetime) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", video_id).strip("-.") or "video"
    return f"VIG-{slug}-{when:%Y%m%dT%H%M%SZ}"


def _timeline(
    ledger: RiskLedger,
    scenes: list[SceneUnderstanding],
    answers: tuple[PerceptionAnswer, ...],
) -> list[TimelineEntry]:
    entries: dict[tuple[float, str], TimelineEntry] = {}

    def add(t_s: float, event: str, who: str) -> None:
        if not event.strip():
            return
        entries.setdefault(
            (round(max(0.0, t_s), 2), event),
            TimelineEntry(t_s=max(0.0, t_s), event=event, observed_by=who),
        )

    for scene in scenes:
        for moment in scene.key_moments:
            add(moment.t_s, moment.description, "perception")
        if not scene.key_moments:
            # An all-clear window still has to appear on the timeline: "looked here, saw
            # nothing" is the evidence a control clip is judged on, and an empty timeline
            # reads as a broken integration rather than a quiet bay.
            seen = f"{len(scene.conflicts)} conflict(s) noted" if scene.conflicts else "no conflict"
            add(scene.t0_s, f"Watched {scene.t0_s:.1f}-{scene.t1_s:.1f}s: {seen}", "perception")
    for answer in answers:
        if answer.observed_at_s is not None:
            add(answer.observed_at_s, f"Targeted re-look settled it: {answer.answer}", "agent")
    for finding in ledger.assessments:
        if finding.clip_window is not None:
            add(
                finding.clip_window[0],
                f"Vigil flagged {finding.hazard_id}: {finding.title}",
                "agent",
            )
    ordered = sorted(entries.values(), key=lambda e: (e.t_s, e.observed_by != "perception"))
    return ordered[:MAX_TIMELINE_ENTRIES]


def _citations(assessments: list[RiskAssessment]) -> list[RuleCitation]:
    best: dict[str, RuleCitation] = {}
    for assessment in assessments:
        for citation in assessment.violated_rules:
            prior = best.get(citation.rule_id)
            if prior is None or citation.relevance > prior.relevance:
                best[citation.rule_id] = citation
    return sorted(best.values(), key=lambda c: (-c.relevance, c.rule_id))


def _actions(assessments: list[RiskAssessment]) -> list[Mitigation]:
    """Immediate first, and never the same instruction twice."""
    seen: set[str] = set()
    ordered: list[Mitigation] = []
    for horizon in ("immediate", "end_of_shift", "7d", "30d"):
        for assessment in assessments:
            for mitigation in assessment.mitigations:
                key = mitigation.action.strip().lower()
                if mitigation.horizon == horizon and key not in seen:
                    seen.add(key)
                    ordered.append(mitigation)
    return ordered


def _narrative(
    video: VideoSource,
    ledger: RiskLedger,
    scenes: list[SceneUnderstanding],
    stopped: str,
    backend: str,
    reasoning_model: str,
) -> str:
    lines: list[str] = []
    worst = ledger.top_risk
    if worst is None:
        lines.append(
            f"Vigil read {video.duration_s:.1f}s from {video.camera_label} and found "
            "nothing that met the reporting bar. That is a conclusion, not an "
            "absence of looking: the lines of inquiry it checked are listed below."
        )
    else:
        ahead = (
            "with no lead time estimable"
            if worst.lead_time_s is None
            else f"{worst.lead_time_s:.1f}s ahead of contact"
        )
        lines.append(
            f"Vigil read {video.duration_s:.1f}s from {video.camera_label} and stopped "
            f"on {len(ledger.assessments)} finding(s). The highest is {worst.hazard_id}, "
            f"{worst.title}: {worst.disposition.value.replace('_', ' ')} at risk "
            f"{worst.risk_score}/20, {ahead}."
        )
    for assessment in ledger.assessments:
        window = (
            f"{assessment.clip_window[0]:.1f}-{assessment.clip_window[1]:.1f}s"
            if assessment.clip_window
            else "window unresolved"
        )
        warning = (
            "no warning time"
            if assessment.lead_time_s is None
            else f"{assessment.lead_time_s:.1f}s of warning"
        )
        title = (
            assessment.title
            if assessment.title.endswith((".", "!", "?"))
            else f"{assessment.title}."
        )
        wording = assessment.disposition.value.replace("_", " ")
        lines.append(
            f"- {assessment.hazard_id} at {window}, {warning}: {title} "
            f"Risk {assessment.risk_score}/20 (severity {assessment.severity}/5, "
            f"{assessment.likelihood.value}), {wording}. {assessment.reasoning}"
        )
    if ledger.hypotheses:
        tested = (
            ", ".join(h.id for h in ledger.hypotheses if h.status is HypothesisStatus.SUPPORTED)
            or "none confirmed"
        )
        lines.append(
            f"- Went back to the footage {len(ledger.hypotheses)} time(s) with a "
            f"specific question; confirmed: {tested}."
        )
    if ledger.cleared:
        lines.append("- Checked and cleared: " + "; ".join(ledger.cleared) + ".")
    blind = list(dict.fromkeys(v.hidden_zone for s in scenes for v in s.visibility_limits))
    if blind:
        lines.append("- Not observable from this camera: " + "; ".join(blind[:3]) + ".")
    if stopped:
        lines.append(f"The investigation ended early: {stopped}.")
    if backend == "mock" or reasoning_model.startswith("scripted"):
        lines.append(
            "Scripted run: the deterministic backends were used, so these are "
            "pipeline outputs and not accuracy measurements. Re-run with a Nebius "
            "key for a real number."
        )
    return "\n".join(lines)


async def investigate(
    video: VideoSource,
    settings: Settings,
    *,
    perception: PerceptionBackend | None = None,
    reasoner: Reasoner | None = None,
    rulebook: Rulebook | None = None,
    client: NebiusClient | None = None,
    spend: Spend | None = None,
) -> Outcome:
    """Wire the collaborators and run one video.

    The CLI, the benchmark and the tests all start here. Hand-rolled wiring in three places is
    three chances for the thing that got benchmarked to differ from the thing on stage.
    Anything the caller did not supply is built *and* closed here; shared parts are left open.
    """
    owns_perception = perception is None
    owns_reasoner = reasoner is None
    backend = perception or build_perception(settings, client)
    brain = reasoner or build_reasoner(settings, client)
    kit = ToolKit(
        perception=backend,
        rulebook=rulebook or load_rulebook(settings.policy_dir),
        video=video,
        settings=settings,
    )
    try:
        investigator = Investigator(
            video=video, kit=kit, reasoner=brain, settings=settings, spend=spend
        )
        return await investigator.run()
    finally:
        if owns_reasoner:
            await brain.aclose()
        if owns_perception:
            await backend.aclose()


__all__ = [
    "STOPPED_NO_BUDGET",
    "STOPPED_NO_TURNS",
    "Investigator",
    "Outcome",
    "investigate",
]
