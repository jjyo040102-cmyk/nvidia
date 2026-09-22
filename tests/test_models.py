"""The contracts every layer shares.

These models are the seams between perception, the agent, the report and the UI. A validator
that silently accepts a bad scene is worse than one that raises: the bad data reaches a report
that a safety manager acts on.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vigil.models.report import IncidentReport, ModelProvenance, TimelineEntry
from vigil.models.risk import (
    Disposition,
    HazardClass,
    Hypothesis,
    HypothesisStatus,
    Likelihood,
    Mitigation,
    RiskAssessment,
    RiskLedger,
    RuleCitation,
)
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
)
from vigil.models.trace import InvestigationTrace, TraceKind, TraceStep

from .helpers import make_assessment


def test_risk_score_is_the_documented_five_by_four_matrix() -> None:
    assert make_assessment(severity=5, likelihood=Likelihood.IMMINENT).risk_score == 20
    assert make_assessment(severity=5, likelihood=Likelihood.LIKELY).risk_score == 15
    assert make_assessment(severity=3, likelihood=Likelihood.POSSIBLE).risk_score == 6
    assert make_assessment(severity=1, likelihood=Likelihood.REMOTE).risk_score == 1


@pytest.mark.parametrize(
    ("severity", "likelihood", "expected"),
    [
        (5, Likelihood.IMMINENT, Disposition.STOP_WORK),
        (4, Likelihood.IMMINENT, Disposition.STOP_WORK),
        (3, Likelihood.IMMINENT, Disposition.INTERVENE_NOW),  # 12: urgent, but not a stop-the-line
        (5, Likelihood.LIKELY, Disposition.INTERVENE_NOW),
        (4, Likelihood.LIKELY, Disposition.INTERVENE_NOW),
        (3, Likelihood.POSSIBLE, Disposition.ADVISE),
        (1, Likelihood.POSSIBLE, Disposition.MONITOR),
        (1, Likelihood.REMOTE, Disposition.MONITOR),
    ],
)
def test_disposition_bands(severity: int, likelihood: Likelihood, expected: Disposition) -> None:
    assessment = make_assessment(severity=severity, likelihood=likelihood)
    assert assessment.disposition is expected
    assert assessment.actionable is (expected is not Disposition.MONITOR)


def test_warning_time_is_reported_but_never_rewritten_by_the_score() -> None:
    """The invariant the whole scoring design rests on.

    Two findings about the same collision must agree on risk whatever the sampling grid looked
    like. ``lead_time_s`` is a claim about Vigil's own reaction; feeding it into the score made
    a better-read clip look less dangerous.
    """
    early = make_assessment(lead_time_s=9.0)
    late = make_assessment(lead_time_s=0.3)
    assert early.risk_score == late.risk_score
    assert early.disposition is late.disposition


def test_severity_and_likelihood_are_bounded() -> None:
    with pytest.raises(ValidationError):
        make_assessment(severity=6)
    with pytest.raises(ValidationError):
        make_assessment(severity=0)
    with pytest.raises(ValidationError):
        make_assessment(confidence=1.4)
    with pytest.raises(ValidationError):
        make_assessment(lead_time_s=-0.1)


def test_a_window_cannot_run_backwards() -> None:
    with pytest.raises(ValidationError, match="ascending"):
        make_assessment(clip_window=(8.0, 4.0))
    assert make_assessment(clip_window=(4.0, 8.0)).clip_window == (4.0, 8.0)


def test_invented_fields_are_refused() -> None:
    """``extra="forbid"`` is what stops a hallucinated key becoming a silently ignored one."""
    with pytest.raises(ValidationError):
        RiskAssessment.model_validate({**make_assessment().model_dump(), "fault": "operator"})


def test_citation_and_mitigation_defaults() -> None:
    citation = RuleCitation(rule_id="osha-x", source="29 CFR", text="Keep the path clear.")
    assert citation.relevance == 0.5
    assert citation.url is None
    mitigation = Mitigation(action="Bank out the reversing alarm.")
    assert mitigation.horizon == "immediate"
    assert mitigation.owner_role == "floor_supervisor"
    with pytest.raises(ValidationError):
        RuleCitation(rule_id="", source="s", text="t")


def test_hazard_classes_are_the_shared_vocabulary() -> None:
    assert HazardClass.WORKER_RIDING_ON_FORKS.value == "worker_riding_on_forks"
    assert len(set(HazardClass)) == 7


def test_hypothesis_reports_itself_unresolved() -> None:
    open_one = Hypothesis(id="Q1", statement="Do the routes meet?")
    assert open_one.unresolved is True
    assert (
        Hypothesis(
            id="Q2", statement="Do the routes meet?", status=HypothesisStatus.SUPPORTED
        ).unresolved
        is False
    )


def test_ledger_ranks_and_summarises() -> None:
    top = make_assessment(hazard_id="H1", severity=5, likelihood=Likelihood.IMMINENT)
    mid = make_assessment(
        hazard_id="H2", severity=2, likelihood=Likelihood.POSSIBLE, lead_time_s=11.0
    )
    quiet = make_assessment(
        hazard_id="H3", severity=1, likelihood=Likelihood.REMOTE, lead_time_s=None
    )
    ledger = RiskLedger(video_id="v", assessments=[mid, top, quiet], cleared=["gate interlock"])
    assert ledger.top_risk is not None
    assert ledger.top_risk.hazard_id == "H1"
    assert {a.hazard_id for a in ledger.open_hazards} == {"H1", "H2"}
    assert ledger.max_lead_time_s == 11.0
    assert RiskLedger(video_id="v").top_risk is None
    assert RiskLedger(video_id="v").max_lead_time_s is None


# ------------------------------------------------------------------------ scene


def make_scene(**overrides: object) -> SceneUnderstanding:
    base: dict[str, object] = {
        "clip_id": "c1",
        "t0_s": 0.0,
        "t1_s": 4.0,
        "area_type": AreaType.WAREHOUSE,
        "lighting": Lighting.TYPICAL_INDOOR,
        "surface": SurfaceCondition.DRY_CLEAR,
        "entities": [
            Entity(
                ref="p1",
                category="worker",
                role=Role.VULNERABLE_PARTY,
                location="left",
                kinematics=Kinematics.WALKING,
            ),
            Entity(
                ref="FL1",
                category="forklift",
                role=Role.POWERED_MACHINE,
                location="right",
                kinematics=Kinematics.DRIVING,
                speed_mps=2.4,
            ),
        ],
    }
    base.update(overrides)
    return SceneUnderstanding(**base)


def test_entity_handles_are_normalised_not_guessed() -> None:
    entity = Entity(ref=" p 1 ", category="worker", location="x")
    assert entity.ref == "P1"
    assert Entity(
        ref="P1", category="w", location="x", ppe_missing=[" hi_vis ", ""]
    ).ppe_missing == ["hi_vis"]


def test_a_conflict_needs_two_distinct_parties() -> None:
    with pytest.raises(ValidationError, match="two distinct"):
        Conflict(participants=["P1", "p1"], kind="struck_by", severity_hint=4, rationale="r")
    with pytest.raises(ValidationError):
        Conflict(
            participants=["P1", "P2", "P3", "P4", "P5"],
            kind="struck_by",
            severity_hint=4,
            rationale="r",
        )


def test_time_to_event_is_bounded_and_honest_about_unknowns() -> None:
    conflict = Conflict(
        participants=["P1", "FL1"], kind="struck_by", severity_hint=4, rationale="r"
    )
    assert conflict.time_to_event_s is None
    assert (
        Conflict(
            participants=["P1", "FL1"],
            kind="struck_by",
            severity_hint=4,
            rationale="r",
            time_to_event_s=0.0,
        ).time_to_event_s
        == 0.0
    )
    with pytest.raises(ValidationError):
        Conflict(
            participants=["P1", "FL1"],
            kind="struck_by",
            severity_hint=4,
            rationale="r",
            time_to_event_s=-1,
        )


def test_a_scene_refuses_a_conflict_about_things_that_do_not_exist() -> None:
    """The one asymmetry in the schema: stray extra keys are forgiven, ghost entities are not."""
    with pytest.raises(ValidationError, match="undeclared"):
        make_scene(
            conflicts=[
                Conflict(
                    participants=["P1", "X9"], kind="struck_by", severity_hint=4, rationale="r"
                )
            ]
        )
    with pytest.raises(ValidationError, match="precedes"):
        make_scene(t0_s=5.0, t1_s=4.0)


def test_unknown_extra_keys_are_tolerated() -> None:
    scene = SceneUnderstanding.model_validate(
        {**make_scene().model_dump(), "colour": "grey", "mood": "busy"}
    )
    assert not hasattr(scene, "colour")


def test_scene_views_answer_the_questions_the_agent_asks() -> None:
    scene = make_scene(
        entities=[
            *make_scene().entities,
            Entity(ref="CR1", category="crate", role=Role.UNKNOWN, location="shelf"),
        ],
        conflicts=[
            Conflict(
                participants=["P1", "FL1"],
                kind="struck_by",
                severity_hint=4,
                rationale="r",
                time_to_event_s=3.0,
            ),
            Conflict(
                participants=["P1", "CR1"],
                kind="struck_by_falling_object",
                severity_hint=2,
                rationale="r",
            ),
        ],
    )
    assert scene.duration_s == 4.0
    assert scene.entity(" p1 ") is not None
    assert scene.entity("nope") is None
    assert {e.ref for e in scene.vulnerable_parties} == {"P1"}
    assert [m.ref for m in scene.machines] == ["FL1"]
    assert scene.most_urgent is not None
    assert set(scene.most_urgent.participants) == {"P1", "FL1"}  # the validator sorts them


def test_most_urgent_ranks_an_untimed_conflict_last() -> None:
    scene = make_scene(
        conflicts=[
            Conflict(participants=["P1", "FL1"], kind="struck_by", severity_hint=5, rationale="r"),
            Conflict(
                participants=["P1", "FL1"],
                kind="slip_trip",
                severity_hint=2,
                rationale="r",
                time_to_event_s=1.0,
            ),
        ]
    )
    assert scene.most_urgent is not None
    assert scene.most_urgent.kind == "slip_trip"
    assert make_scene().most_urgent is None


def test_visibility_limits_default_to_the_actors_not_the_camera() -> None:
    limit = VisibilityLimit(occluder="a rack", hidden_zone="the aisle mouth")
    assert limit.camera_blind is False


# ------------------------------------------------------------------------ report / trace


def test_trace_reads_back_the_way_the_ui_renders_it() -> None:
    trace = InvestigationTrace(video_id="v1")
    trace.add(TraceStep(index=0, kind=TraceKind.THOUGHT, title="thinking", detail="because"))
    trace.add(TraceStep(index=1, kind=TraceKind.TOOL_CALL, title="survey", tool="survey"))
    trace.add(TraceStep(index=2, kind=TraceKind.DECISION, title="flagged", tool="flag"))
    assert trace.last_thought == "because"
    assert trace.tools_used() == ["survey", "flag"]
    assert trace.to_loglines()[0].startswith("[ 0] thought")
    empty = InvestigationTrace(video_id="v2")
    assert empty.last_thought == ""
    assert empty.tools_used() == []


def test_trace_step_bounds() -> None:
    with pytest.raises(ValidationError):
        TraceStep(index=-1, kind=TraceKind.THOUGHT, title="x")
    with pytest.raises(ValidationError):
        TraceStep(index=0, kind=TraceKind.THOUGHT, title="x", duration_ms=-5)


def make_report(assessments: list[RiskAssessment]) -> IncidentReport:
    return IncidentReport(
        report_id="r1",
        video_id="v1",
        headline="h",
        narrative="n",
        assessments=assessments,
        provenance=ModelProvenance(
            perception_backend="mock",
            vision_model="scripted",
            reasoning_model="scripted",
        ),
        timeline=[TimelineEntry(t_s=1.0, event="saw it")],
    )


def test_report_exposes_the_worst_finding_and_the_earliest_warning() -> None:
    early = make_assessment(
        hazard_id="H1", severity=1, likelihood=Likelihood.POSSIBLE, lead_time_s=7.0
    )
    worst = make_assessment(
        hazard_id="H2", severity=5, likelihood=Likelihood.IMMINENT, lead_time_s=1.0
    )
    report = make_report([early, worst])
    assert report.worst is not None
    assert report.worst.hazard_id == "H2"
    assert report.lead_time_s == 7.0
    assert make_report([]).worst is None
    assert make_report([]).lead_time_s is None


def test_report_carries_the_disclaimer_it_ships_with() -> None:
    assert "Decision support only" in make_report([]).disclaimer


def test_key_moments_and_hypotheses_round_trip_through_json() -> None:
    scene = make_scene(key_moments=[KeyMoment(t_s=2.0, description="the load shifts")])
    assert SceneUnderstanding.model_validate_json(scene.model_dump_json()) == scene
