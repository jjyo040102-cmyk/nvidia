"""The benchmark's own tests: the arithmetic that decides which numbers reach a README.

These matter more than most, because a scoring bug does not crash anything -- it prints a
confident, wrong figure next to the product's name. So the truth used here is hand-built, the
findings are hand-built, and every definition in :mod:`vigil.eval.metrics` is checked against a
case where the wrong reading would produce a different number.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from vigil.config import Settings
from vigil.eval import (
    ClipScore,
    EvalRun,
    Scorecard,
    Skipped,
    TrueHazard,
    VideoTruth,
    evaluate,
    is_measurement,
    load_truth,
    render,
    score_clip,
    score_run,
)
from vigil.eval.metrics import LEAD_TOLERANCE_S
from vigil.models.risk import Disposition, HazardClass, Likelihood, RiskAssessment, RuleCitation

from .helpers import CLIP_DIR, make_assessment

STRUCK_BY = TrueHazard(
    type=HazardClass.STRUCK_BY,
    t_start=3.3,
    t_impact=6.9,
    anticipatable_s=3.6,
    severity=5,
    entities=["P1", "FL1"],
    description="A worker crosses the lane as the forklift reaches the pillar.",
)

FALL = TrueHazard(
    type=HazardClass.FALL_FROM_HEIGHT,
    t_start=6.2,
    t_impact=9.4,
    anticipatable_s=3.2,
    severity=4,
    entities=["P1", "L1"],
    description="An overreach past the ladder's side rails.",
)


def truth(
    clip_id: str = "blind_corner_struck_by",
    hazards: list[TrueHazard] | None = None,
    duration_s: float = 10.0,
) -> VideoTruth:
    return VideoTruth(
        clip_id=clip_id,
        title=clip_id.replace("_", " "),
        path=f"{clip_id}.mp4",
        duration_s=duration_s,
        fps=12.0,
        hazards=[STRUCK_BY] if hazards is None else hazards,
    )


def finding(
    hazard_id: str = "H1",
    *,
    window: tuple[float, float] | None = (0.0, 4.0),
    lead_time_s: float | None = 2.9,
    hazard_class: HazardClass | None = HazardClass.STRUCK_BY,
    severity: int = 5,
    **overrides: object,
) -> RiskAssessment:
    return make_assessment(
        hazard_id=hazard_id,
        clip_window=window,
        lead_time_s=lead_time_s,
        hazard_class=hazard_class,
        severity=severity,
        **overrides,
    )


# ------------------------------------------------------------------------ ground truth loading


def test_the_authored_sidecars_all_validate_and_the_controls_are_empty() -> None:
    loaded = load_truth(CLIP_DIR)
    assert len(loaded) == 8
    hazards = [t for t in loaded.values() if not t.is_control]
    assert len(hazards) == 6
    assert {h.type for t in hazards for h in t.hazards} == {
        HazardClass.STRUCK_BY,
        HazardClass.STRUCK_BY_REVERSING_VEHICLE,
        HazardClass.FALL_FROM_HEIGHT,
        HazardClass.SLIP_TRIP,
        HazardClass.WORKER_RIDING_ON_FORKS,
    }
    assert all(t.is_control for t in loaded.values() if t.clip_id.startswith("control_"))


def test_a_sidecar_that_disagrees_with_itself_is_refused(tmp_path: Path) -> None:
    """anticipatable_s is the denominator of the headline metric. A sidecar that contradicts its
    own timestamps would silently move every captured-ratio number in the run."""
    bad = {
        "clip_id": "contradiction",
        "path": "contradiction.mp4",
        "duration_s": 10.0,
        "fps": 12.0,
        "hazards": [
            {
                "type": "struck_by",
                "t_start": 3.0,
                "t_impact": 7.0,
                "anticipatable_s": 1.5,
                "severity": 5,
                "entities": ["P1"],
                "description": "x",
            }
        ],
    }
    (tmp_path / "contradiction.groundtruth.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match=r"anticipatable_s 1\.5 disagrees"):
        load_truth(tmp_path)


def test_a_sidecar_named_for_a_different_clip_is_refused(tmp_path: Path) -> None:
    payload: dict[str, Any] = {
        "clip_id": "impostor",
        "path": "impostor.mp4",
        "duration_s": 9.0,
        "fps": 12.0,
        "hazards": [],
    }
    (tmp_path / "real_name.groundtruth.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="sidecars must match their clip"):
        load_truth(tmp_path)


def test_a_hazard_type_outside_the_taxonomy_is_refused(tmp_path: Path) -> None:
    """A mechanism the rulebook cannot index is a truth row no finding could ever be paired with."""
    payload: dict[str, Any] = {
        "clip_id": "invented",
        "path": "invented.mp4",
        "duration_s": 9.0,
        "fps": 12.0,
        "hazards": [
            {
                "type": "electrocution",
                "t_start": 1.0,
                "t_impact": 4.0,
                "anticipatable_s": 3.0,
                "severity": 5,
                "entities": ["P1"],
                "description": "x",
            }
        ],
    }
    (tmp_path / "invented.groundtruth.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="electrocution"):
        load_truth(tmp_path)


def test_a_directory_of_clips_with_no_sidecars_scores_nothing_rather_than_failing(
    tmp_path: Path,
) -> None:
    assert load_truth(tmp_path) == {}


# ---------------------------------------------------------------------------- the pairing


def test_the_finding_that_names_the_mechanism_wins_the_hazard() -> None:
    """Two findings claim one hazard: the one that says which mechanism is at work is the match,
    even though the other is the more alarming number."""
    vague = finding("H9", hazard_class=None, severity=5, window=(3.0, 6.5), lead_time_s=0.4)
    precise = finding("H1", window=(0.0, 4.0), lead_time_s=2.9)
    score = score_clip(truth(), [vague, precise])
    assert [m.finding.hazard_id for m in score.matches] == ["H1"]
    assert [f.hazard_id for f in score.unpaired] == ["H9"]


def test_a_second_finding_about_the_same_event_is_unpaired_not_a_false_positive() -> None:
    """A hi-vis gap beside a live struck-by is a true observation the sidecar has no column for."""
    score = score_clip(
        truth(),
        [
            finding("H1"),
            finding("H2", hazard_class=None, severity=2, window=(0.0, 4.0), lead_time_s=None),
        ],
    )
    assert score.detected == 1
    assert len(score.unpaired) == 1
    assert score.false_positives == 0


def test_anything_filed_on_a_control_is_a_false_positive() -> None:
    empty = truth("control_housekeeping", hazards=[])
    score = score_clip(empty, [finding("H1", hazard_class=None)])
    assert score.is_control
    assert score.false_positives == 1
    assert score.detected == 0


def test_a_finding_with_no_window_can_still_be_a_detection_but_carries_no_timing() -> None:
    """The entity refs a model returns do not always line up with a surveyed window. The hazard
    was still named -- what is unknown is when the agent stood relative to it."""
    score = score_clip(truth(), [finding("H1", window=None, lead_time_s=2.0)])
    match = score.matches[0]
    assert match.overlap_s == 0.0
    assert match.available_lead_s is None
    assert match.timing_error_s is None
    assert match.fails is None, "unknown timing is unknown, not an over-claim"


def test_two_hazards_in_one_clip_are_paired_one_to_one() -> None:
    second = TrueHazard(
        type=HazardClass.SLIP_TRIP,
        t_start=1.0,
        t_impact=2.0,
        anticipatable_s=1.0,
        severity=3,
        entities=["P1"],
        description="An unmarked spill.",
    )
    score = score_clip(
        truth(hazards=[STRUCK_BY, second]),
        [
            finding("H1", window=(0.0, 4.0), lead_time_s=2.9),
            finding(
                "H2",
                hazard_class=HazardClass.SLIP_TRIP,
                severity=3,
                window=(0.0, 1.5),
                lead_time_s=0.5,
            ),
        ],
    )
    assert score.detected == 2
    assert score.missed == ()
    assert [m.truth.type for m in score.matches] == [
        HazardClass.SLIP_TRIP,
        HazardClass.STRUCK_BY,
    ], "matches come back in the order the hazards happen"


def test_a_miss_is_named_by_the_hazard_that_was_missed() -> None:
    score = score_clip(truth(hazards=[STRUCK_BY, FALL]), [finding("H1")])
    assert [m.truth.type for m in score.matches] == [HazardClass.STRUCK_BY]
    assert [h.type for h in score.missed] == [HazardClass.FALL_FROM_HEIGHT]


# ---------------------------------------------------------------------------- the lead clock


def test_the_reported_window_and_lead_have_to_agree_about_when_contact_lands() -> None:
    """window end + claimed warning is the contact time the report implies. Where that lands
    later than the truth, the report is placing the event in the wrong second."""
    match = score_clip(truth(), [finding("H1")]).matches[0]
    assert match.available_lead_s == pytest.approx(2.9)
    assert match.captured_ratio == pytest.approx(2.9 / 3.6)
    assert match.timing_error_s == pytest.approx(0.0)
    assert match.fails is None


def test_a_warning_claimed_beyond_the_footage_is_a_defect_not_a_good_score() -> None:
    """The number a marketing slide would love: 9 seconds of warning on a clip that contained
    3.6. The harness exists to make that impossible to print without a red line next to it."""
    match = score_clip(truth(), [finding("H1", window=(0.0, 4.0), lead_time_s=9.0)]).matches[0]
    assert match.over_claim_s == pytest.approx(9.0 - 2.9)
    assert match.fails is not None
    assert "claims 9.0s of warning" in match.fails
    score = score_clip(truth(), [finding("H1", window=(0.0, 4.0), lead_time_s=9.0)])
    assert len(score.failures()) == 1
    assert score.failures()[0].startswith("blind_corner_struck_by: H1")


def test_sampling_granularity_is_allowed_one_second_of_slack() -> None:
    """Frames are 1/12 s apart and windows end where they end. Demanding exact agreement would
    fail a correct system for the resolution it was sampled at."""
    within = finding("H1", window=(0.0, 4.0), lead_time_s=2.9 + LEAD_TOLERANCE_S - 0.1)
    over = finding("H1", window=(0.0, 4.0), lead_time_s=2.9 + LEAD_TOLERANCE_S + 0.1)
    assert score_clip(truth(), [within]).matches[0].fails is None
    assert score_clip(truth(), [over]).matches[0].fails is not None


def test_a_warning_raised_at_the_moment_of_contact_scores_zero_lead_not_negative() -> None:
    """The window already contains the impact: the finding is real, the anticipation is not."""
    late = finding("H1", window=(0.0, 7.5), lead_time_s=0.0)
    match = score_clip(truth(), [late]).matches[0]
    assert match.available_lead_s == 0.0
    assert match.captured_ratio == 0.0
    assert match.timing_error_s == pytest.approx(0.6)


def test_a_lead_longer_than_the_clip_contains_is_clamped_not_reported_over_a_hundred_percent() -> (
    None
):
    """An early read can leave more seconds than the hazard's own anticipation window. A ratio
    above 1 would read as "we predicted something that had not become visible yet"."""
    early = TrueHazard(
        type=HazardClass.STRUCK_BY,
        t_start=6.0,
        t_impact=7.0,
        anticipatable_s=1.0,
        severity=5,
        entities=["P1"],
        description="Late-developing conflict.",
    )
    match = score_clip(truth(hazards=[early]), [finding("H1", window=(0.0, 1.0))]).matches[0]
    assert match.available_lead_s == pytest.approx(6.0)
    assert match.captured_ratio == 1.0


# ---------------------------------------------------------------------------- the scorecard


def card(clips: list[ClipScore], rule_ids: list[str] | None = None) -> Scorecard:
    return Scorecard(clips=tuple(clips), known_rule_ids=frozenset(rule_ids or ["osha-1"]))


def test_the_scorecard_counts_hazards_not_clips() -> None:
    two = VideoTruth(
        clip_id="pair",
        path="pair.mp4",
        duration_s=12.0,
        fps=12.0,
        hazards=[STRUCK_BY, FALL],
    )
    score = score_clip(
        two,
        [
            finding("H1"),
            finding(
                "H2",
                hazard_class=HazardClass.FALL_FROM_HEIGHT,
                severity=4,
                window=(4.0, 6.5),
                lead_time_s=2.9,
            ),
        ],
    )
    board = card(
        [
            score,
            score_clip(truth(), [finding("H1")]),
            score_clip(truth("control_housekeeping", hazards=[]), []),
        ]
    )
    assert len(board.clips) == 3
    assert board.truth_hazards == 3
    assert board.detected == 3
    assert board.recall == 1.0
    assert board.mechanism_accuracy == 1.0
    assert board.severity_within_1 == 1.0
    assert len(board.hazard_clips) == 2
    assert len(board.controls) == 1


def test_a_missed_hazard_lowers_recall_and_is_attributable() -> None:
    board = card([score_clip(truth(hazards=[STRUCK_BY, FALL]), [finding("H1")])])
    assert board.detected == 1
    assert board.truth_hazards == 2
    assert board.recall == 0.5
    assert board.clips[0].missed[0].type is HazardClass.FALL_FROM_HEIGHT


def test_every_citation_in_a_reported_finding_must_resolve_in_the_rulebook() -> None:
    """The loop already drops citations it never retrieved. If one still reaches a report, this
    is the line that catches it: an id that is not in the corpus cannot be a citation."""
    cited = finding(
        "H1",
        violated_rules=[
            RuleCitation(rule_id="osha-1", source="29 CFR 1910", text="Keep clear."),
            RuleCitation(rule_id="osha-invented", source="made up", text="Nothing."),
        ],
    )
    board = card([score_clip(truth(), [cited])], rule_ids=["osha-1"])
    assert board.citations == (1, 2)
    assert board.as_dict()["citations_resolved"] == 1
    assert board.as_dict()["citations_filed"] == 2


def test_an_over_claim_is_visible_from_the_board_not_only_from_the_row() -> None:
    board = card(
        [score_clip(truth(), [finding("H1", lead_time_s=12.0)])],
    )
    assert board.max_over_claim_s == pytest.approx(9.1)
    assert len(board.over_claims) == 1
    assert board.as_dict()["integrity_failures"] == list(board.over_claims)


def test_the_board_never_divides_by_zero_on_an_empty_run() -> None:
    board = card([])
    assert board.recall == 1.0
    assert board.mean_available_lead_s is None
    assert board.mechanism_accuracy is None
    assert board.mean_captured_ratio is None
    assert board.citations == (0, 0)


def test_the_serialised_board_is_json_and_carries_every_row() -> None:
    """Whatever the renderer prints, the machine-readable run has to carry too -- the UI reads
    this dictionary, and a metric that exists only in a Markdown table cannot be audited."""
    board = score_run(
        [(truth(), [finding("H1")]), (truth("control_housekeeping", hazards=[]), [])], []
    )
    payload = json.loads(json.dumps(board.as_dict()))
    assert payload["recall"] == 1.0
    assert payload["detected"] == 1
    assert len(payload["per_clip"]) == 2
    assert payload["per_clip"][0]["missed"] == []
    assert payload["per_clip"][1]["is_control"] is True
    row = payload["per_clip"][0]["matches"][0]
    assert row["type"] == "struck_by"
    assert row["available_lead_s"] == pytest.approx(2.9)
    assert row["timing_error_s"] == pytest.approx(0.0)
    assert row["disposition"] == "stop_work"


# --------------------------------------------------------------------- what counts as measured


@pytest.mark.parametrize(
    ("perception", "reasoning", "expected"),
    [
        ("mock", "mock", False),
        ("nebius", "mock", False),
        ("mock", "nebius", False),
        ("nebius", "nebius", True),
        ("cosmos", "nebius", True),
        ("cosmos", "mock", False),
    ],
)
def test_only_a_run_that_could_have_been_wrong_on_its_own_counts_as_a_measurement(
    settings_for: Callable[..., Settings],
    perception: str,
    reasoning: str,
    expected: bool,
) -> None:
    settings = settings_for(
        nebius_api_key="k",
        perception_backend=perception,
        reasoning_backend=reasoning,
    )
    assert is_measurement(settings) is expected


def test_the_verdict_for_a_scripted_run_refuses_to_be_a_number() -> None:
    run = EvalRun(
        card=card([score_clip(truth(), [finding("H1")])]),
        skipped=(),
        measured=False,
        backends={"perception": "mock", "reasoning": "mock"},
        models=("scripted reads",),
        spend={"live_calls": 0},
        seconds=1.0,
    )
    message, code = run.verdict()
    assert code != 0
    assert message.startswith("NOT A MEASUREMENT")
    assert run.as_dict()["measured"] is False
    assert run.as_dict()["exit_code"] == code


def test_a_measured_run_with_an_over_claim_still_fails() -> None:
    run = EvalRun(
        card=card([score_clip(truth(), [finding("H1", lead_time_s=12.0)])]),
        skipped=(),
        measured=True,
        backends={"perception": "nebius", "reasoning": "nebius"},
        models=("nvidia/nemotron-3-super-120b-a12b",),
        spend={"live_calls": 4},
        seconds=2.0,
    )
    assert not run.integrity_ok
    message, code = run.verdict()
    assert code != 0
    assert "claimed more warning" in message


def test_a_clean_measured_run_is_the_only_one_that_exits_zero() -> None:
    run = EvalRun(
        card=card([score_clip(truth(), [finding("H1")])]),
        skipped=(),
        measured=True,
        backends={"perception": "nebius", "reasoning": "nebius"},
        models=("m",),
        spend={"live_calls": 4},
        seconds=2.0,
    )
    message, code = run.verdict()
    assert code == 0
    assert "1/1 hazards anticipated" in message


def test_unscored_clips_keep_a_measured_run_from_claiming_success() -> None:
    run = EvalRun(
        card=card([score_clip(truth(), [finding("H1")])]),
        skipped=(Skipped("mystery", "no ground-truth sidecar beside it"),),
        measured=True,
        backends={"perception": "nebius", "reasoning": "nebius"},
        models=("m",),
        spend={"live_calls": 4},
        seconds=2.0,
    )
    assert run.verdict()[1] != 0
    assert "no ground truth" in run.verdict()[0]


# ---------------------------------------------------------------------------- the table


def test_the_table_has_one_row_per_authored_hazard_plus_one_per_control() -> None:
    board = card(
        [
            score_clip(truth(hazards=[STRUCK_BY, FALL]), [finding("H1")]),
            score_clip(truth("control_housekeeping", hazards=[]), []),
        ]
    )
    rows = _body_rows(_run(board=board))
    assert len(rows) == 3
    assert any("fall_from_height" in line and "| no |" in line for line in rows), (
        "the missed hazard is printed as a miss, by name"
    )
    assert any("struck_by sev5" in line and "| yes |" in line for line in rows)
    assert any("nothing (control)" in line for line in rows)


def _run(
    board: Scorecard,
    *,
    measured: bool = True,
    skipped: tuple[Skipped, ...] = (),
) -> EvalRun:
    return EvalRun(
        card=board,
        skipped=skipped,
        measured=measured,
        backends={
            "perception": "nebius" if measured else "mock",
            "reasoning": "nebius" if measured else "mock",
        },
        models=("m",) if measured else ("scripted reads of the authored scenarios",),
        spend={
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "live_calls": 0,
            "cache_hits": 0,
            "est_usd": 0.0,
        },
        seconds=0.4,
    )


def _body_rows(run: EvalRun) -> list[str]:
    return [
        line
        for line in render(run).splitlines()
        if line.startswith("| ") and "clip |" not in line and "---" not in line
    ]


def test_the_table_says_what_ran_and_what_it_cost() -> None:
    text = render(_run(card([score_clip(truth(), [finding("H1")])]), measured=False))
    assert "NOT a measurement -- scripted backends" in text
    assert "Cost: 0 tokens, 0 live call(s), 0 cache hit(s), ~$0.0000, 0.4 s wall" in text
    assert "**Anticipated 1 of 1 authored hazards** (100%)" in text
    assert "Backends: mock perception / mock reasoning" in text


def test_a_discrepancy_between_two_printings_of_one_number_is_impossible() -> None:
    """The claim in this module's docstring: screen and README are the same renderer. Checked by
    asserting every figure in the table also appears in the JSON the same run wrote."""
    board = card([score_clip(truth(), [finding("H1")])])
    run = _run(board)
    text = render(run)
    payload = json.dumps(run.as_dict())
    assert '"detected": 1' in payload
    assert "Anticipated 1 of 1" in text
    assert "2.9s" in text
    assert '"available_lead_s": 2.9' in payload


# ---------------------------------------------------------------------------- the real sweep


async def test_scoring_the_whole_authored_set_on_scripted_backends_is_clean_but_unmeasured(
    settings_for: Callable[..., Settings],
) -> None:
    """End to end over the eight clips, no key and no credit: the pipeline, the matcher and the
    integrity gate all have to agree before a real run is worth anything."""
    settings = settings_for(perception_backend="mock", reasoning_backend="mock")
    seen: list[str] = []

    def progress(truth_row: VideoTruth, score: ClipScore, stopped: str) -> None:
        assert stopped == ""
        seen.append(truth_row.clip_id)
        assert isinstance(score.matches, tuple)

    run = await evaluate(settings, on_progress=progress)
    assert len(seen) == 8
    assert run.measured is False
    assert run.card.truth_hazards == 6
    assert run.card.detected == 6
    assert run.card.false_positives == 0
    assert run.card.over_claims == ()
    assert run.card.citations[0] == run.card.citations[1] > 0
    assert run.backends == {"perception": "mock", "reasoning": "mock"}
    assert any("rule-based controller" in m for m in run.models)
    assert run.verdict()[1] == 1


async def test_a_single_clip_can_be_scored_without_reading_the_rest(
    settings_for: Callable[..., Settings],
) -> None:
    settings = settings_for(perception_backend="mock", reasoning_backend="mock")
    run = await evaluate(settings, clip_ids=["blind_corner_struck_by"])
    assert len(run.card.clips) == 1
    assert run.card.clips[0].clip_id == "blind_corner_struck_by"


async def test_a_clip_with_no_sidecar_is_reported_as_unscorable(
    settings_for: Callable[..., Settings], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recording the harness cannot check is excluded and named, never scored as a zero."""
    settings = settings_for(perception_backend="mock", reasoning_backend="mock")
    real = load_truth(CLIP_DIR)
    del real["unmarked_spill_slip"]
    monkeypatch.setattr("vigil.eval.harness.load_truth", lambda _dir: real)
    run = await evaluate(settings, clip_ids=["unmarked_spill_slip", "ride_on_forks"])
    assert [s.clip_id for s in run.skipped] == ["unmarked_spill_slip"]
    assert [c.clip_id for c in run.card.clips] == ["ride_on_forks"]
    assert run.as_dict()["skipped"] == [
        {"clip_id": "unmarked_spill_slip", "reason": "no ground-truth sidecar beside it"}
    ]
    assert run.verdict()[1] != 0, "a sweep that skipped a clip is not a complete answer"


def test_a_match_built_from_a_real_report_pairs_with_its_own_hazard() -> None:
    """Guards the assumption underneath everything: that the finding's window and lead are the
    pair the report actually published, not ones the scorer invented."""
    assessment = finding("H1", window=(0.0, 4.0), lead_time_s=2.9, likelihood=Likelihood.IMMINENT)
    match = score_clip(truth(), [assessment]).matches[0]
    assert match.finding is assessment
    assert match.truth is STRUCK_BY
    assert match.finding.disposition is Disposition.STOP_WORK
