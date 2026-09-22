"""The table ``vigil eval`` prints, and the same table as Markdown for a README.

One renderer for both, because the fastest way to misreport a benchmark is to keep the numbers
that go on screen in one place and the numbers that go in the write-up in another.

One row per authored hazard, not per finding: a benchmark row that reads "missed" has to name
the thing that was missed, and a row count driven by findings would make the table grow every
time the agent files something extra.
"""

from __future__ import annotations

from vigil.eval.groundtruth import TrueHazard
from vigil.eval.harness import EvalRun
from vigil.eval.metrics import ClipScore, Match

HEADER = (
    "clip",
    "authored hazard",
    "found",
    "claimed",
    "available",
    "captured",
    "timing err",
    "extra",
)
SEPARATOR = "| --- | --- | :-: | --: | --: | --: | --: | --: |"


def _seconds(value: float | None) -> str:
    return "--" if value is None else f"{value:.1f}s"


def _pct(value: float | None) -> str:
    return "--" if value is None else f"{value * 100:.0f}%"


def _signed(value: float | None) -> str:
    return "--" if value is None else f"{value:+.1f}s"


def _hazard_rows(clip: ClipScore) -> list[tuple[TrueHazard, Match | None]]:
    """One row per authored hazard, in the order they happen, paired with what stood for it."""
    return [
        (hazard, next((m for m in clip.matches if m.truth is hazard), None))
        for hazard in sorted(clip.truth, key=lambda h: h.t_start)
    ]


def _control_row(clip: ClipScore) -> str:
    """A control clip passes by staying quiet. Anything filed on it is a false positive."""
    verdict = "no" if not clip.unpaired else "FAIL"
    return (
        f"| {clip.clip_id} | nothing (control) | {verdict} | -- | -- | -- | -- "
        f"| {len(clip.unpaired)} |"
    )


def _row(clip: ClipScore, hazard: TrueHazard, match: Match | None, first: bool) -> str:
    found = "yes" if match is not None else "no"
    extra = str(len(clip.unpaired)) if first else ""
    return (
        f"| {clip.clip_id} | {hazard.type.value} sev{hazard.severity} @ {hazard.t_impact:.1f}s "
        f"| {found} | {_seconds(match.claimed_lead_s if match else None)} "
        f"| {_seconds(match.available_lead_s if match else None)} "
        f"| {_pct(match.captured_ratio if match else None)} "
        f"| {_signed(match.timing_error_s if match else None)} "
        f"| {extra} |"
    )


def table(run: EvalRun) -> list[str]:
    """Markdown lines: what ran and what it cost, per-hazard rows, then the aggregate claims."""
    card = run.card
    lines = [
        f"### {'' if run.measured else 'NOT a measurement -- scripted backends. '}Vigil benchmark",
        "",
        f"- Clips scored: {len(card.clips)} ({len(card.hazard_clips)} with authored hazards, "
        f"{len(card.controls)} controls)",
        f"- Backends: {run.backends['perception']} perception / "
        f"{run.backends['reasoning']} reasoning",
        f"- Ran on: {'; '.join(run.models)}",
        f"- Cost: {run.spend['prompt_tokens'] + run.spend['completion_tokens']} tokens, "
        f"{run.spend['live_calls']} live call(s), {run.spend['cache_hits']} cache hit(s), "
        f"~${run.spend['est_usd']:.4f}, {run.seconds:.1f} s wall",
        "",
        "| " + " | ".join(HEADER) + " |",
        SEPARATOR,
    ]
    for clip in card.clips:
        if clip.is_control:
            lines.append(_control_row(clip))
            continue
        for index, (hazard, match) in enumerate(_hazard_rows(clip)):
            lines.append(_row(clip, hazard, match, index == 0))
    integrity = (
        "clean -- no finding claimed more warning than the footage held"
        if not card.over_claims
        else "; ".join(card.over_claims)
    )
    resolved, filed = card.citations
    anticipated = (
        f"- **Anticipated {card.detected} of {card.truth_hazards} authored hazards** "
        f"({_pct(card.recall)})"
    )
    warning = (
        f"- Warning still available where the finding was made: "
        f"{_seconds(card.mean_available_lead_s)} (claimed {_seconds(card.mean_claimed_lead_s)}), "
        f"which is {_pct(card.mean_captured_ratio)} of the warning each clip contains"
    )
    lines += [
        "",
        anticipated,
        warning,
        f"- Contact placed within {_seconds(card.mean_abs_timing_error_s)} of where it landed",
        f"- Mechanism named correctly {_pct(card.mechanism_accuracy)}; severity within one band "
        f"{_pct(card.severity_within_1)}",
        f"- Citations that resolve in the rulebook: {resolved}/{filed}",
        f"- Findings on control clips (false positives): {card.false_positives}",
        f"- Extra findings on hazard clips, not in the sidecar: {card.uncorroborated}",
        f"- Stop-work escalations: {card.escalations}",
        f"- Lead-time integrity: {integrity}",
    ]
    if run.skipped:
        lines.append("- Not scored: " + ", ".join(f"{s.clip_id} ({s.reason})" for s in run.skipped))
    return lines


def render(run: EvalRun) -> str:
    return "\n".join(table(run))


__all__ = ["render", "table"]
