"""The report as a document a safety manager can forward, rather than a JSON payload.

Deliberately plain: no emoji, no colour, tables that survive being pasted into an email. Every
number here traces to a field the agent was forced to fill in, so the prose cannot claim
something the data does not support.
"""

from __future__ import annotations

from vigil.models.report import IncidentReport
from vigil.models.risk import RiskAssessment
from vigil.models.trace import InvestigationTrace, TraceKind

_DISPOSITION_LINE = {
    "stop_work": "STOP WORK until this is controlled",
    "intervene_now": "Intervene now -- someone should act within the shift",
    "advise": "Advise and monitor",
    "monitor": "Monitor",
}


def render_report(report: IncidentReport) -> str:
    lines: list[str] = [f"# {report.headline}", ""]
    lines += _at_a_glance(report)
    lines += ["", "## What Vigil saw", "", report.narrative.strip(), ""]
    for index, assessment in enumerate(
        sorted(report.assessments, key=lambda a: -a.risk_score), start=1
    ):
        lines += _finding(f"Finding {index}", assessment)
    if report.citations:
        lines += _rules(report)
    if report.actions:
        lines += _actions(report)
    if report.timeline:
        lines += _timeline(report)
    if report.trace and report.trace.steps:
        lines += _investigation(report.trace)
    lines += ["", "---", "", report.disclaimer.strip(), ""]
    return "\n".join(lines)


def _at_a_glance(report: IncidentReport) -> list[str]:
    provenance = report.provenance
    worst = report.worst
    rows = [
        ("Video", report.video_id),
        ("Report", report.report_id),
        ("Generated", report.generated_at.strftime("%Y-%m-%d %H:%M:%SZ UTC")),
        ("Findings", str(len(report.assessments))),
        (
            "Highest risk",
            f"{worst.risk_score}/20 ({worst.severity} x {worst.likelihood.value})"
            if worst
            else "none",
        ),
        (
            "Earliest warning",
            f"{report.lead_time_s:.1f} s before the predicted event"
            if report.lead_time_s
            else "no finding carried time ahead of the event",
        ),
        ("Vision backend", provenance.perception_backend),
        ("Vision model", provenance.vision_model),
        ("Reasoning model", provenance.reasoning_model),
        (
            "Cost",
            f"{provenance.prompt_tokens + provenance.completion_tokens} tokens in "
            f"{provenance.wall_clock_ms / 1000:.1f} s",
        ),
        (
            "Footage analysed",
            f"{provenance.video_hours_analysed:.3f} h "
            f"({provenance.video_hours_analysed * 3600:.0f} s)",
        ),
    ]
    return ["| | |", "| --- | --- |"] + [f"| **{label}** | {value} |" for label, value in rows]


def _finding(heading: str, assessment: RiskAssessment) -> list[str]:
    window = (
        f"{assessment.clip_window[0]:.1f}-{assessment.clip_window[1]:.1f} s"
        if assessment.clip_window
        else "not established"
    )
    lead = f"{assessment.lead_time_s:.1f} s" if assessment.lead_time_s is not None else "none"
    mechanism = (
        [f"- Mechanism of harm: {assessment.hazard_class.value}"] if assessment.hazard_class else []
    )
    lines = [
        f"## {heading}: {assessment.title}",
        "",
        f"**{_DISPOSITION_LINE[assessment.disposition]}.**",
        "",
        f"- Risk score **{assessment.risk_score}/20** -- severity {assessment.severity}/5, "
        f"likelihood {assessment.likelihood.value}",
        *mechanism,
        f"- Predicted event: {assessment.predicted_event}",
        f"- Who is exposed: {', '.join(assessment.entities_involved) or 'not identified'}",
        f"- Seen in window {window}; warning {lead} ahead of the predicted contact",
        f"- Confidence {assessment.confidence:.2f}",
    ]
    if assessment.reasoning:
        lines.append(f"- Why it was believed: {assessment.reasoning}")
    if assessment.violated_rules:
        lines.append(
            "- Rules breached: "
            + ", ".join(
                f"`{c.rule_id}` ({c.relevance:.2f} relevance)" for c in assessment.violated_rules
            )
        )
    for mitigation in assessment.mitigations:
        lines.append(
            f"- Action ({mitigation.horizon}, {mitigation.owner_role}): {mitigation.action}"
            + (f" -- {mitigation.rationale}" if mitigation.rationale else "")
        )
    return [*lines, ""]


def _rules(report: IncidentReport) -> list[str]:
    lines = ["## The rules cited", ""]
    for citation in report.citations:
        link = f" -- [source]({citation.url})" if citation.url else ""
        lines += [
            f"### `{citation.rule_id}` -- {citation.source}{link}",
            "",
            f"> {citation.text}",
            "",
            f"Retrieved at relevance {citation.relevance:.2f}. Cited by: "
            + (
                ", ".join(
                    a.hazard_id
                    for a in report.assessments
                    if any(c.rule_id == citation.rule_id for c in a.violated_rules)
                )
                or "no surviving finding"
            ),
            "",
        ]
    return lines


def _actions(report: IncidentReport) -> list[str]:
    immediate = [a for a in report.actions if a.horizon == "immediate"]
    later = [a for a in report.actions if a.horizon != "immediate"]
    lines = ["## Recommended actions", ""]
    if immediate:
        lines.append("**Before anyone goes near this area again**")
        lines += [f"- {m.action} _(owner: {m.owner_role})_" for m in immediate]
        lines.append("")
    if later:
        lines.append("**Follow-up**")
        lines += [f"- ({m.horizon}) {m.action} _(owner: {m.owner_role})_" for m in later]
        lines.append("")
    return lines


def _timeline(report: IncidentReport) -> list[str]:
    lines = ["## Timeline", "", "| t (s) | what | seen by |", "| --- | --- | --- |"]
    lines += [
        f"| {entry.t_s:.1f} | {entry.event} | {entry.observed_by} |" for entry in report.timeline
    ]
    return [*lines, ""]


def _investigation(trace: InvestigationTrace) -> list[str]:
    steps = [
        s for s in trace.steps if s.kind in {TraceKind.THOUGHT, TraceKind.DECISION, TraceKind.ERROR}
    ]
    lines = [
        "## How it was investigated",
        "",
        f"{trace.iterations} planning turn(s), {len(trace.steps)} logged steps, "
        f"tools used: {', '.join(trace.tools_used()) or 'none'}.",
        "",
    ]
    for step in steps:
        detail = (step.detail or step.result_summary or "").replace("\n", " ").strip()
        lines.append(
            f"{step.index:>2}. **{step.title}**" + (f" -- {detail[:400]}" if detail else "")
        )
    return [*lines, ""]
