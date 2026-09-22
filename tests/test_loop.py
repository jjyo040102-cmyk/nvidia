"""The loop: what it must never do, and what it must always record.

Two things are asserted here over and over. First, the gates: a token ceiling, a spend ceiling and
a turn limit each have to stop the investigation *and say so in the report*, because an agent that
stops silently writes a confident report about half the footage. Second, the audit trail: every
thought, call, observation and decision has to be in the trace with the numbers attached, because
that trail is the only reason a safety finding from a language model is usable at all.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

import pytest

from vigil.agent.actions import (
    Action,
    ClearAction,
    FinishAction,
    FlagAction,
    RelookAction,
    SurveyAction,
    Turn,
)
from vigil.agent.loop import (
    MAX_TIMELINE_ENTRIES,
    STOPPED_NO_BUDGET,
    STOPPED_NO_TURNS,
    Investigator,
    _args_line,
    _clause,
    _hypothesis,
    _report_id,
    _timeline,
    investigate,
)
from vigil.agent.reasoner import Briefing, Reasoner, ReasonerError
from vigil.agent.tools import ToolKit
from vigil.config import Settings
from vigil.models.risk import HypothesisStatus, Likelihood, RiskAssessment, RiskLedger
from vigil.models.scene import (
    AreaType,
    Entity,
    KeyMoment,
    Lighting,
    Role,
    SceneUnderstanding,
    SurfaceCondition,
    VisibilityLimit,
)
from vigil.models.trace import TraceKind
from vigil.nebius import BudgetExceeded
from vigil.perception.base import (
    Clip,
    PerceptionAnswer,
    PerceptionBackend,
    PerceptionError,
    Usage,
)
from vigil.perception.mock import MockPerception
from vigil.policy.kb import Rulebook
from vigil.video.source import VideoSource

from .helpers import make_assessment

NOW = datetime(2026, 5, 4, 9, 8, 7, tzinfo=UTC)


# --------------------------------------------------------------------------- stubs


class StubReasoner(Reasoner):
    """Plays back a fixed list of actions and remembers every briefing it was shown."""

    name = "stub"

    def __init__(self, *actions: Action, usage: Usage | None = None) -> None:
        self.queue = list(actions)
        self.briefings: list[Briefing] = []
        self._usage = usage or Usage(prompt_tokens=40, completion_tokens=10, model="stub/one")

    @property
    def usage(self) -> Usage:
        return self._usage

    async def model_id(self) -> str:
        return "stub/one"

    async def plan(self, briefing: Briefing) -> Turn:
        self.briefings.append(briefing)
        if self.queue:
            action = self.queue.pop(0)
        else:
            action = FinishAction(summary="Nothing further to establish.")
        return Turn(thought=f"turn {len(self.briefings)}", action=action)


class ExplodingReasoner(StubReasoner):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    async def plan(self, briefing: Briefing) -> Turn:
        self.briefings.append(briefing)
        raise self.error


class DeadPerception(PerceptionBackend):
    """Fails the way a networked vision model does, on both operations."""

    name = "dead"

    async def survey(self, clip: Clip) -> tuple[SceneUnderstanding, Usage]:
        raise PerceptionError("vision layer is down")

    async def ask(self, clip: Clip, question: str) -> PerceptionAnswer:
        raise PerceptionError("vision layer is down")


def kit_for(
    video: VideoSource,
    rulebook: Rulebook,
    settings: Settings,
    perception: PerceptionBackend | None = None,
) -> ToolKit:
    return ToolKit(
        perception=perception or MockPerception(settings),
        rulebook=rulebook,
        video=video,
        settings=settings,
    )


def run(
    video: VideoSource,
    rulebook: Rulebook,
    settings: Settings,
    reasoner: Reasoner,
    perception: PerceptionBackend | None = None,
) -> Investigator:
    kit = kit_for(video, rulebook, settings, perception)
    return Investigator(
        video=video,
        kit=kit,
        reasoner=reasoner,
        settings=settings,
        generated_at=NOW,
    )


# --------------------------------------------------------------------------- scripted end to end


async def test_a_hazard_clip_comes_back_flagged_measured_and_traced(
    blind_corner: VideoSource, rulebook: Rulebook, tmp_settings: Settings
) -> None:
    outcome = await investigate(blind_corner, tmp_settings, rulebook=rulebook)
    report = outcome.report

    assert outcome.findings, "the scripted controller must find the authored collision"
    worst = outcome.findings[0]
    assert worst.risk_score == max(a.risk_score for a in outcome.findings)
    assert outcome.lead_time_s is not None
    assert outcome.lead_time_s > 0.0, "a warning two seconds out is the product"

    assert report.video_id == blind_corner.video_id
    assert report.report_id.startswith(f"VIG-{blind_corner.video_id}-")
    assert report.headline
    assert report.assessments == list(outcome.findings)
    assert report.provenance.perception_backend == "mock"
    assert "scripted" in report.provenance.reasoning_model
    assert report.provenance.video_hours_analysed == pytest.approx(blind_corner.duration_s / 3600)
    assert "Scripted run" in report.narrative, "a mock run has to label itself as one"

    assert outcome.trace.video_id == blind_corner.video_id
    assert [step.index for step in outcome.trace.steps] == list(range(len(outcome.trace.steps)))
    kinds = {step.kind for step in outcome.trace.steps}
    assert {TraceKind.THOUGHT, TraceKind.TOOL_CALL, TraceKind.OBSERVATION} <= kinds
    assert outcome.trace.iterations >= 1
    assert outcome.spend is not None
    assert outcome.spend["live_calls"] == 0, "a mock run must not touch the network"
    assert outcome.stopped == ""
    assert outcome.scenes
    assert outcome.answers


async def test_every_trace_step_is_addressable_and_no_step_is_empty(
    blind_corner: VideoSource, rulebook: Rulebook, tmp_settings: Settings
) -> None:
    outcome = await investigate(blind_corner, tmp_settings, rulebook=rulebook)
    for step in outcome.trace.steps:
        assert step.title.strip()
        assert len(step.title) <= 200
        assert step.kind in TraceKind
        assert step.duration_ms >= 0
    tool_calls = [s for s in outcome.trace.steps if s.kind is TraceKind.TOOL_CALL]
    assert all(s.tool for s in tool_calls)


async def test_a_control_clip_ends_with_a_conclusion_not_an_empty_file(
    control_videos: list[VideoSource], rulebook: Rulebook, tmp_settings: Settings
) -> None:
    outcome = await investigate(control_videos[0], tmp_settings, rulebook=rulebook)
    assert outcome.findings == ()
    assert outcome.report.headline.startswith("No finding met the reporting bar")
    assert "nothing that met the reporting bar" in outcome.report.narrative
    assert outcome.ledger.summary, "an all-clear still has to be a written decision"
    assert outcome.report.timeline, "what it watched is still evidence"


async def test_the_report_gathers_citations_once_and_puts_the_urgent_first(
    hazard_videos: list[VideoSource], rulebook: Rulebook, tmp_settings: Settings
) -> None:
    outcomes = [
        await investigate(video, tmp_settings, rulebook=rulebook) for video in hazard_videos[:3]
    ]
    for outcome in outcomes:
        report = outcome.report
        ids = [c.rule_id for c in report.citations]
        assert len(ids) == len(set(ids)), "the same standard twice reads as padding, not sourcing"
        assert report.citations == sorted(report.citations, key=lambda c: (-c.relevance, c.rule_id))
        horizons = [m.horizon for m in report.actions]
        keys = [m.action.strip().lower() for m in report.actions]
        assert len(keys) == len(set(keys))
        assert horizons == sorted(horizons, key=["immediate", "end_of_shift", "7d", "30d"].index)
        for assessment in report.assessments:
            assert assessment.hazard_id in report.narrative


# --------------------------------------------------------------------------- the gates


async def test_a_token_ceiling_reached_first_stops_both_loops_and_says_where(
    blind_corner: VideoSource, rulebook: Rulebook, settings_for: Callable[..., Settings]
) -> None:
    brain = StubReasoner(usage=Usage(prompt_tokens=900, completion_tokens=200, model="stub/one"))
    tight = settings_for(token_budget=1000)
    outcome = await run(blind_corner, rulebook, tight, brain).run()
    assert outcome.stopped == STOPPED_NO_BUDGET
    assert brain.briefings == [], "the guard runs before the planner is asked"
    errors = [s for s in outcome.trace.steps if s.kind is TraceKind.ERROR]
    assert errors
    assert "token" in errors[0].title.lower()
    assert outcome.trace.iterations == 0
    assert "ended early" in outcome.report.narrative


async def test_a_spend_ceiling_hit_while_reading_the_footage_is_not_a_crash(
    blind_corner: VideoSource, rulebook: Rulebook, tmp_settings: Settings
) -> None:
    class Broke(MockPerception):
        async def survey(self, clip: Clip) -> tuple[SceneUnderstanding, Usage]:
            raise BudgetExceeded("this call would cross the ceiling")

    outcome = await run(
        blind_corner, rulebook, tmp_settings, StubReasoner(), Broke(tmp_settings)
    ).run()
    assert outcome.stopped.startswith("spend ceiling reached during the opening scan")
    assert outcome.findings == ()


async def test_a_spend_ceiling_hit_while_planning_stops_the_loop(
    blind_corner: VideoSource, rulebook: Rulebook, tmp_settings: Settings
) -> None:
    brain = ExplodingReasoner(BudgetExceeded("over the ceiling"))
    outcome = await run(blind_corner, rulebook, tmp_settings, brain).run()
    assert outcome.stopped.startswith("spend ceiling reached:")
    assert len(brain.briefings) == 1


async def test_a_spend_ceiling_hit_while_re_looking_is_recorded_not_swallowed(
    blind_corner: VideoSource, rulebook: Rulebook, tmp_settings: Settings
) -> None:
    """The opening scan is the cheap part; the follow-ups are what run a trial credit away.

    Without the trace step the report reads as an agent that looked twice and concluded, when
    the second look never happened.
    """

    class OutOfCredit(MockPerception):
        async def ask(self, clip: Clip, question: str) -> PerceptionAnswer:
            raise BudgetExceeded("the next re-look would cross the ceiling")

    brain = StubReasoner(
        RelookAction(t0_s=4.0, t1_s=8.0, question="Does the worker step into the aisle?")
    )
    outcome = await run(
        blind_corner, rulebook, tmp_settings, brain, OutOfCredit(tmp_settings)
    ).run()
    assert outcome.stopped.startswith("spend ceiling reached:")
    assert "cross the ceiling" in outcome.stopped
    errors = [s for s in outcome.trace.steps if s.kind is TraceKind.ERROR]
    assert errors, "the stop belongs in the audit trail a reviewer reads"
    assert "Stopped by the spend ceiling" in errors[0].title
    assert "ended early" in outcome.report.narrative


async def test_a_planner_that_cannot_produce_an_action_ends_the_run_loudly(
    blind_corner: VideoSource, rulebook: Rulebook, tmp_settings: Settings
) -> None:
    outcome = await run(
        blind_corner, rulebook, tmp_settings, ExplodingReasoner(ReasonerError("no JSON"))
    ).run()
    assert "no usable action" in outcome.stopped
    titles = [s.title for s in outcome.trace.steps]
    assert "Planning failed" in titles
    assert outcome.findings == ()


async def test_running_out_of_turns_is_recorded_as_incomplete(
    blind_corner: VideoSource, rulebook: Rulebook, settings_for: Callable[..., Settings]
) -> None:
    brain = StubReasoner(
        SurveyAction(t0_s=0.0, t1_s=4.0),
        SurveyAction(t0_s=0.0, t1_s=4.0),
    )
    outcome = await run(blind_corner, rulebook, settings_for(max_iterations=2), brain).run()
    assert outcome.stopped == STOPPED_NO_TURNS
    assert "stopped before the agent called finish" in outcome.ledger.summary
    assert outcome.trace.iterations == 2


async def test_perception_that_cannot_read_the_footage_yields_no_findings_not_an_exception(
    blind_corner: VideoSource, rulebook: Rulebook, tmp_settings: Settings
) -> None:
    outcome = await run(
        blind_corner, rulebook, tmp_settings, StubReasoner(), DeadPerception()
    ).run()
    assert outcome.stopped == "perception could not read the footage"
    assert outcome.findings == ()
    assert outcome.report.timeline == []
    assert outcome.trace.steps, "a failed investigation still has to be auditable"


# --------------------------------------------------------------------------- bookkeeping


def flag(hazard_id: str, severity: int, likelihood: Likelihood) -> FlagAction:
    return FlagAction(
        hazard_id=hazard_id,
        title=f"Authored {hazard_id}",
        severity=severity,
        likelihood=likelihood,
        predicted_event="Contact at the crossing.",
        entities_involved=["P1", "FL1"],
    )


async def test_flagging_the_same_hazard_twice_keeps_the_worse_reading(
    blind_corner: VideoSource, rulebook: Rulebook, tmp_settings: Settings
) -> None:
    brain = StubReasoner(
        SurveyAction(t0_s=0.0, t1_s=4.0),
        flag("H1", 2, Likelihood.POSSIBLE),
        flag("H1", 5, Likelihood.IMMINENT),
        flag("H1", 1, Likelihood.REMOTE),
        FinishAction(summary="Established."),
    )
    outcome = await run(blind_corner, rulebook, tmp_settings, brain).run()
    assert len(outcome.findings) == 1
    assert outcome.findings[0].severity == 5
    assert "H1" in outcome.report.narrative


async def test_findings_are_ordered_by_risk_not_by_the_order_they_arrived(
    blind_corner: VideoSource, rulebook: Rulebook, tmp_settings: Settings
) -> None:
    brain = StubReasoner(
        flag("H2", 3, Likelihood.LIKELY),
        flag("H1", 5, Likelihood.IMMINENT),
        FinishAction(summary="Nothing further to establish."),
    )
    outcome = await run(blind_corner, rulebook, tmp_settings, brain).run()
    assert [a.hazard_id for a in outcome.findings] == ["H1", "H2"]


async def test_a_repeated_question_becomes_one_hypothesis_even_when_asked_twice(
    blind_corner: VideoSource, rulebook: Rulebook, tmp_settings: Settings
) -> None:
    question = "Do the routes of P1 and FL1 reach the same point while both are there?"
    brain = StubReasoner(
        SurveyAction(t0_s=0.0, t1_s=4.0),
        RelookAction(t0_s=4.0, t1_s=8.0, question=question),
        RelookAction(t0_s=4.0, t1_s=8.0, question=question),
        FinishAction(summary="Nothing further to establish."),
    )
    outcome = await run(blind_corner, rulebook, tmp_settings, brain).run()
    assert len(outcome.ledger.hypotheses) == 1
    hypothesis = outcome.ledger.hypotheses[0]
    assert hypothesis.id == "Q1"
    assert hypothesis.status is HypothesisStatus.SUPPORTED
    assert question in hypothesis.statement


async def test_a_clear_shows_up_as_a_cleared_line_of_inquiry(
    blind_corner: VideoSource, rulebook: Rulebook, tmp_settings: Settings
) -> None:
    brain = StubReasoner(
        ClearAction(subject="The gate", reason="Closed and bolted."),
        FinishAction(summary="Nothing further to establish."),
    )
    outcome = await run(blind_corner, rulebook, tmp_settings, brain).run()
    assert outcome.ledger.cleared == ["The gate"]
    assert "Checked and cleared: The gate" in outcome.report.narrative


async def test_the_second_hand_is_not_double_counted_in_the_token_provenance(
    blind_corner: VideoSource, rulebook: Rulebook, tmp_settings: Settings
) -> None:
    """Vision tokens and planner tokens both land in the report, and neither is lost.

    ``MockPerception`` returns ``Usage(model="mock")`` with no token counts, so a costed
    stand-in is needed to exercise the accumulation path -- which is the path the budget
    guard and the published provenance both rest on.
    """

    class Costed(MockPerception):
        async def survey(self, clip: Clip) -> tuple[SceneUnderstanding, Usage]:
            scene, _ = await super().survey(clip)
            return scene, Usage(model="costed", prompt_tokens=120, completion_tokens=30)

        async def ask(self, clip: Clip, question: str) -> PerceptionAnswer:
            answer = await super().ask(clip, question)
            return answer.model_copy(
                update={"usage": Usage(model="costed", prompt_tokens=70, completion_tokens=12)}
            )

    brain = StubReasoner(
        SurveyAction(t0_s=0.0, t1_s=4.0),
        RelookAction(t0_s=4.0, t1_s=8.0, question="Who is in the lane?"),
        FinishAction(summary="Nothing further to establish."),
    )
    investigator = run(blind_corner, rulebook, tmp_settings, brain, Costed(tmp_settings))
    outcome = await investigator.run()
    windows = len(investigator.kit.scenes)
    assert investigator._vision_prompt == windows * 120 + 70, "the re-read cost nothing"
    assert investigator._vision_completion == windows * 30 + 12
    total = outcome.report.provenance
    assert total.prompt_tokens == investigator._vision_prompt + brain.usage.prompt_tokens
    assert (
        total.completion_tokens == investigator._vision_completion + brain.usage.completion_tokens
    )
    assert outcome.trace.prompt_tokens == total.prompt_tokens


async def test_investigate_closes_only_the_backends_it_built(
    blind_corner: VideoSource,
    rulebook: Rulebook,
    tmp_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A borrowed backend belongs to its caller; a built one must not outlive the run."""

    class Counting(MockPerception):
        closed = 0

        async def aclose(self) -> None:
            Counting.closed += 1

    borrowed = Counting(tmp_settings)
    await investigate(blind_corner, tmp_settings, perception=borrowed, rulebook=rulebook)
    assert Counting.closed == 0

    monkeypatch.setattr(
        "vigil.agent.loop.build_perception",
        lambda settings, client=None: Counting(settings),
    )
    await investigate(blind_corner, tmp_settings, rulebook=rulebook)
    assert Counting.closed == 1, "the backend this call built was left open"


# --------------------------------------------------------------------------- helpers


def test_a_thought_is_shortened_at_a_word_not_mid_number() -> None:
    assert _clause("one  two   three") == "one two three"
    long = "x" * 200
    assert _clause(long) == "x" * 87 + "..."
    assert len(_clause(long)) == 90


def test_the_trace_shows_two_arguments_at_most() -> None:
    line = _args_line({"t0_s": 1.5, "t1_s": 4.0, "question": "why" * 40})
    assert line.startswith("t0_s=1.5 t1_s=4.0")
    assert "question" not in line
    assert _args_line({}) == "no arguments"
    assert _args_line({"question": "q"}) == "question=q"


@pytest.mark.parametrize(
    ("video_id", "expected"),
    [
        ("blind_corner_struck_by", "VIG-blind_corner_struck_by-20260504T090807Z"),
        ("bay 3 / north-mound!", "VIG-bay-3-north-mound-20260504T090807Z"),
        ("///", "VIG-video-20260504T090807Z"),
    ],
)
def test_a_report_id_is_a_filename_and_a_timestamp(video_id: str, expected: str) -> None:
    got = _report_id(video_id, NOW)
    assert got == expected
    assert re.fullmatch(r"VIG-[\w.-]+-\d{8}T\d{6}Z", got)


def _moment(t_s: float, text: str) -> KeyMoment:
    return KeyMoment(t_s=t_s, description=text)


def _scene(t0: float, t1: float, moments: Sequence[KeyMoment]) -> SceneUnderstanding:
    return SceneUnderstanding(
        clip_id=f"v@{t0}-{t1}",
        t0_s=t0,
        t1_s=t1,
        area_type=AreaType.WAREHOUSE,
        lighting=Lighting.TYPICAL_INDOOR,
        surface=SurfaceCondition.DRY_CLEAR,
        entities=[
            Entity(ref="P1", category="worker", role=Role.VULNERABLE_PARTY, location="aisle")
        ],
        key_moments=list(moments),
        visibility_limits=[
            VisibilityLimit(occluder="pillar", hidden_zone="the corner", camera_blind=False)
        ],
        dynamics_narrative="One worker, walking.",
    )


def _ledger(*assessments: RiskAssessment) -> RiskLedger:
    return RiskLedger(
        video_id="v",
        assessments=list(assessments),
        hypotheses=[],
        summary="",
        cleared=[],
    )


def test_the_timeline_is_chronological_and_capped() -> None:
    scene = _scene(0.0, 4.0, [_moment(float(i), f"moment {i}") for i in range(60)])
    entries = _timeline(_ledger(), [scene], ())
    assert len(entries) == MAX_TIMELINE_ENTRIES
    assert [e.t_s for e in entries] == sorted(e.t_s for e in entries)
    assert all(e.observed_by == "perception" for e in entries)


def test_a_finding_is_placed_on_the_timeline_at_the_window_it_was_seen_in() -> None:
    assessment = make_assessment(clip_window=(2.0, 6.0))
    answer = PerceptionAnswer(
        question="why", answer="yes", hazard_present=True, confidence=0.9, observed_at_s=2.0
    )
    entries = _timeline(_ledger(assessment), [], (answer,))
    assert [e.t_s for e in entries] == [2.0, 2.0], "the flag lands on the window, not on now"
    assert [e.event for e in entries] == [
        "Targeted re-look settled it: yes",
        "Vigil flagged H1: Struck-by between converging routes: P1 and FL1",
    ], "the evidence is listed before the conclusion drawn from it"


def test_perception_and_agent_agreement_on_one_instant_keeps_perception_first() -> None:
    scene = _scene(0.0, 4.0, [_moment(2.9, "the truck begins to back")])
    assessment = make_assessment(clip_window=(2.9, 4.0))
    entries = _timeline(_ledger(assessment), [scene], ())
    assert [(e.t_s, e.observed_by) for e in entries] == [(2.9, "perception"), (2.9, "agent")]


def test_an_empty_moment_never_reaches_the_timeline() -> None:
    scene = _scene(0.0, 4.0, [_moment(1.0, "   ")])
    assert _timeline(_ledger(), [scene], ()) == []


def test_hypothesis_status_follows_the_answer_not_the_asking() -> None:
    def answer(present: bool, confidence: float) -> PerceptionAnswer:
        return PerceptionAnswer(
            question="Is the guard in place?",
            answer="checked",
            hazard_present=present,
            confidence=confidence,
        )

    assert _hypothesis(1, answer(True, 0.9)).status is HypothesisStatus.SUPPORTED
    assert _hypothesis(2, answer(False, 0.9)).status is HypothesisStatus.REJECTED
    assert _hypothesis(3, answer(True, 0.2)).status is HypothesisStatus.OPEN
