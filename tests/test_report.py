"""The Markdown report: the artefact a judge, a safety manager or a reviewer actually reads.

Everything here is about whether the document survives being read by a human. The renderer is
the last place a claim can drift from the data -- a number that formats to ``None``, a section
that silently disappears, an instruction that turns into the raw enum token -- and none of that
is visible in the JSON. One test renders a real investigation end to end so the rest can assume
the shapes below are shapes the pipeline actually produces.
"""

from __future__ import annotations

from vigil.agent.loop import investigate
from vigil.config import Settings
from vigil.models.report import IncidentReport, ModelProvenance, TimelineEntry
from vigil.models.risk import (
    Disposition,
    Likelihood,
    Mitigation,
    RuleCitation,
)
from vigil.models.trace import InvestigationTrace, TraceKind, TraceStep
from vigil.policy.kb import Rulebook
from vigil.report.markdown import render_report
from vigil.video.source import VideoSource

from .helpers import make_assessment


def provenance(**overrides: object) -> ModelProvenance:
    base: dict[str, object] = {
        "perception_backend": "mock",
        "vision_model": "scripted reads from the authored scenarios (not a model)",
        "reasoning_model": "scripted",
        "prompt_tokens": 1200,
        "completion_tokens": 300,
        "wall_clock_ms": 4200,
        "video_hours_analysed": 10.0 / 3600.0,
    }
    base.update(overrides)
    return ModelProvenance(**base)


def report(**overrides: object) -> IncidentReport:
    base: dict[str, object] = {
        "report_id": "VIG-test-20260504T090807Z",
        "video_id": "test_clip",
        "headline": "Two findings at Bay 3, one of them stop-work",
        "narrative": "Vigil watched four windows and found a converging route at the corner.",
        "provenance": provenance(),
    }
    base.update(overrides)
    return IncidentReport(**base)


def citation(rule_id: str = "osha-1910-178-pedestrians", **overrides: object) -> RuleCitation:
    base: dict[str, object] = {
        "rule_id": rule_id,
        "source": "OSHA 1910.178",
        "text": "Powered trucks shall sound the audible warning before moving.",
        "url": "https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.178",
        "relevance": 0.74,
    }
    base.update(overrides)
    return RuleCitation(**base)


def trace(*steps: TraceStep) -> InvestigationTrace:
    return InvestigationTrace(
        video_id="test_clip",
        steps=list(steps) or [TraceStep(index=0, kind=TraceKind.THOUGHT, title="Started")],
        iterations=2,
    )


def step(index: int, kind: TraceKind, title: str, **kw: object) -> TraceStep:
    return TraceStep(index=index, kind=kind, title=title, **kw)


# ----------------------------------------------------------------------------------- the document


async def test_a_real_investigation_renders_without_a_single_placeholder(
    blind_corner: VideoSource, rulebook: Rulebook, tmp_settings: Settings
) -> None:
    outcome = await investigate(blind_corner, tmp_settings, rulebook=rulebook)
    text = render_report(outcome.report)

    assert outcome.report.assessments, "this test is pointless if the scripted run finds nothing"
    assert text.startswith(f"# {outcome.report.headline}\n")
    # An f-string that lost its prefix leaves the braces on the page.
    assert "{" not in text
    assert "}" not in text
    assert "None" not in text
    for assessment in outcome.report.assessments:
        assert assessment.hazard_id in text
        assert assessment.title in text
    assert "## The rules cited" in text
    assert "## Recommended actions" in text
    assert "## Timeline" in text
    assert "## How it was investigated" in text
    assert text.rstrip().endswith(outcome.report.disclaimer.strip())


def test_the_glance_table_answers_what_ran_and_for_how_much() -> None:
    text = render_report(
        report(
            assessments=[make_assessment()],
            provenance=provenance(prompt_tokens=1000, completion_tokens=250, wall_clock_ms=9000),
        )
    )
    assert "| **Video** | test_clip |" in text
    assert "| **Report** | VIG-test-20260504T090807Z |" in text
    assert "| **Findings** | 1 |" in text
    assert "| **Highest risk** | 20/20 (5 x imminent) |" in text
    assert "| **Earliest warning** | 2.9 s before the predicted event |" in text
    assert "| **Vision backend** | mock |" in text
    assert "| **Cost** | 1250 tokens in 9.0 s |" in text
    assert "| **Footage analysed** | 0.003 h (10 s) |" in text
    assert "| **Generated** | 2026-" in text, "an undated report cannot be filed"


def test_an_all_clear_still_says_what_it_is_and_what_it_cost() -> None:
    text = render_report(
        report(headline="No finding met the reporting bar", narrative="Nothing time-critical.")
    )
    assert "| **Findings** | 0 |" in text
    assert "| **Highest risk** | none |" in text
    assert "| **Earliest warning** | no finding carried time ahead of the event |" in text
    for heading in ("## The rules cited", "## Recommended actions", "## Timeline"):
        assert heading not in text, "an empty section reads as a broken export, not a quiet bay"


# ------------------------------------------------------------------------------------ a finding


def test_a_finding_prints_its_evidence_and_stops_at_what_it_does_not_know() -> None:
    text = render_report(
        report(
            assessments=[
                make_assessment(
                    clip_window=(3.3, 6.9),
                    reasoning="P2 was already in the lane at 3.4s, so the route is used.",
                    violated_rules=[citation()],
                    mitigations=[
                        Mitigation(
                            action="Stop the truck at the corner.",
                            horizon="end_of_shift",
                            owner_role="floor_supervisor",
                            rationale="Two near-misses logged this week.",
                        )
                    ],
                )
            ]
        )
    )
    assert "## Finding 1: Struck-by between converging routes: P1 and FL1" in text
    assert "**STOP WORK until this is controlled.**" in text
    assert "- Risk score **20/20** -- severity 5/5, likelihood imminent" in text
    assert "- Seen in window 3.3-6.9 s; warning 2.9 s ahead of the predicted contact" in text
    assert "- Why it was believed: P2 was already in the lane" in text
    assert "- Rules breached: `osha-1910-178-pedestrians` (0.74 relevance)" in text
    assert "- Action (end_of_shift, floor_supervisor): Stop the truck at the corner" in text
    assert "-- Two near-misses logged this week." in text


def test_a_missing_window_and_a_missing_reason_are_stated_not_invented() -> None:
    text = render_report(report(assessments=[make_assessment(clip_window=None, lead_time_s=None)]))
    assert "not established" in text
    assert "warning none ahead" in text
    assert "Why it was believed" not in text, "no reasoning, no line"
    bare = render_report(report(assessments=[make_assessment(entities_involved=[])]))
    assert "- Who is exposed: not identified" in bare


def test_a_mitigation_without_a_rationale_does_not_print_an_empty_dash() -> None:
    text = render_report(
        report(
            assessments=[
                make_assessment(mitigations=[Mitigation(action="Clear the lane.", rationale="")])
            ]
        )
    )
    assert "- Action (immediate, floor_supervisor): Clear the lane.\n" in text
    assert "Clear the lane. --" not in text


def test_every_disposition_reaches_the_page_as_an_instruction() -> None:
    """The mapping is total over the enum, so no finding can print a raw token."""
    cases = {
        Disposition.STOP_WORK: (
            5,
            Likelihood.IMMINENT,
            "**STOP WORK until this is controlled.**",
        ),
        Disposition.INTERVENE_NOW: (
            5,
            Likelihood.LIKELY,
            "**Intervene now -- someone should act within the shift.**",
        ),
        Disposition.ADVISE: (2, Likelihood.POSSIBLE, "**Advise and monitor.**"),
        Disposition.MONITOR: (1, Likelihood.REMOTE, "**Monitor.**"),
    }
    for disposition, (severity, likelihood, line) in cases.items():
        text = render_report(
            report(assessments=[make_assessment(severity=severity, likelihood=likelihood)])
        )
        assert line in text, f"{disposition.value} did not reach the page as an instruction"
        assert f"**{disposition.value}.**" not in text, "an unmapped disposition leaked its enum"


# --------------------------------------------------------------------------------- rules, actions


def test_a_citation_shows_its_source_and_who_relied_on_it() -> None:
    rule = citation()
    assessment = make_assessment(violated_rules=[rule])
    text = render_report(report(assessments=[assessment], citations=[rule]))
    assert "## The rules cited" in text
    assert "### `osha-1910-178-pedestrians` -- OSHA 1910.178" in text
    assert "-- [source](https://www.osha.gov" in text
    assert "> Powered trucks shall sound the audible warning" in text
    assert "Retrieved at relevance 0.74. Cited by: H1" in text


def test_an_unsupported_rule_and_an_offline_rule_are_still_labelled() -> None:
    offline = citation("site-keep-lanes-clear", url=None, relevance=0.31)
    text = render_report(report(citations=[offline]))
    assert "### `site-keep-lanes-clear` -- OSHA 1910.178" in text
    assert "[source]" not in text
    assert "Cited by: no surviving finding" in text, (
        "a rule nothing cites must not look load-bearing"
    )


def test_the_urgent_instruction_is_separated_from_the_follow_ups() -> None:
    text = render_report(
        report(
            actions=[
                Mitigation(action="Bank the forklift.", horizon="immediate", owner_role="operator"),
                Mitigation(
                    action="Repaint the crossing.", horizon="30d", owner_role="site_manager"
                ),
            ]
        )
    )
    head = text.split("## Recommended actions")[1]
    assert "**Before anyone goes near this area again**" in head
    assert "**Follow-up**" in head
    assert head.index("Bank the forklift") < head.index("Repaint the crossing")
    assert "- (30d) Repaint the crossing. _(owner: site_manager)_" in head
    assert "- Bank the forklift. _(owner: operator)_" in head


def test_a_report_with_only_follow_ups_omits_the_immediate_heading() -> None:
    head = render_report(report(actions=[Mitigation(action="Add a mirror.", horizon="7d")])).split(
        "## Recommended actions"
    )[1]
    assert "**Before anyone goes near this area again**" not in head
    assert "- (7d) Add a mirror." in head


# --------------------------------------------------------------------------------- timeline, trace


def test_the_timeline_reads_as_seconds_and_witnesses() -> None:
    text = render_report(
        report(
            timeline=[
                TimelineEntry(
                    t_s=3.3, event="The worker steps off the walkway", observed_by="perception"
                ),
                TimelineEntry(t_s=6.9, event="Vigil flagged H1", observed_by="agent"),
            ]
        )
    )
    assert "| t (s) | what | seen by |" in text
    assert "| 3.3 | The worker steps off the walkway | perception |" in text
    assert "| 6.9 | Vigil flagged H1 | agent |" in text


def test_the_investigation_section_shows_judgement_and_hides_plumbing() -> None:
    text = render_report(
        report(
            trace=trace(
                step(
                    0,
                    TraceKind.THOUGHT,
                    "Read the whole bay first",
                    detail="Survey before judging.",
                ),
                step(1, TraceKind.TOOL_CALL, "survey t0_s=0.0 t1_s=4.0", tool="survey"),
                step(
                    2,
                    TraceKind.OBSERVATION,
                    "Read 0.0-4.0s",
                    tool="survey",
                    result_summary="Two entities, one pillar",
                ),
                step(3, TraceKind.DECISION, "Flagged H1"),
                step(4, TraceKind.ERROR, "Stopped by the token budget"),
            )
        )
    )
    body = text.split("## How it was investigated")[1]
    assert "2 planning turn(s), 5 logged steps, tools used: survey." in body
    assert " 0. **Read the whole bay first** -- Survey before judging." in body
    assert " 3. **Flagged H1**" in body
    assert " 4. **Stopped by the token budget**" in body
    assert "survey t0_s=0.0" not in body, "the tool call is not a decision"
    assert "Read 0.0-4.0s" not in body, "an observation is quoted only when it is a judgement"


def test_a_step_with_no_detail_prints_its_title_alone() -> None:
    body = render_report(
        report(trace=trace(step(0, TraceKind.DECISION, "Closed the investigation")))
    ).split("## How it was investigated")[1]
    assert " 0. **Closed the investigation**\n" in body
    assert "**Closed the investigation** --" not in body


def test_a_report_without_a_trace_omits_the_method_section() -> None:
    text = render_report(report(trace=InvestigationTrace(video_id="test_clip")))
    assert "## How it was investigated" not in text
    assert "# " in text, "the rest of the document still renders"


def test_findings_are_rendered_worst_first() -> None:
    minor = make_assessment(
        hazard_id="H2", title="Paint line worn", severity=2, likelihood=Likelihood.POSSIBLE
    )
    major = make_assessment(
        hazard_id="H1", title="Truck against a worker", severity=5, likelihood=Likelihood.IMMINENT
    )
    text = render_report(report(assessments=[minor, major]))
    assert "## Finding 1: Truck against a worker" in text
    assert "## Finding 2: Paint line worn" in text
