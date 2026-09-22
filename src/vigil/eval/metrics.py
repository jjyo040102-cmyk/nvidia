"""Scoring what Vigil filed against what the clip actually does.

Pure arithmetic over models -- no files, no network, no async -- because these are the numbers
that go on a README. Anything that decides a published figure has to be checkable without
running anything.

Definitions, since a benchmark number without one is a slogan:

* **detected** -- a finding the matcher paired with a true hazard: one finding to one hazard,
  preferring the pair that names the right mechanism and overlaps ``[t_start, t_impact]``.
* **available lead** -- ``t_impact`` minus the end of the window the finding is keyed to: how
  much warning was genuinely left when the last frame behind the claim was captured.
* **claimed lead** -- the report's own ``lead_time_s``.
* **over-claim** -- claimed minus available. A positive number here says a human was told they
  had more time than the footage contained, which is treated as a build failure, not a score.
* **captured ratio** -- available lead over ``anticipatable_s``: how much of the warning the
  clip actually held was taken.
* **timing error** -- ``(window end + claimed lead) - t_impact``: where the report puts the
  contact, against where it happens.

False positives are counted where they are unambiguous. On a control clip every finding is one,
because the clip was authored to contain nothing. On a hazard clip a second, unpaired finding
is reported but not penalised: a missing hi-vis vest beside a live struck-by is a true
observation the sidecar has no column for, and calling it a false positive would teach the
agent to stay quiet about things it correctly saw.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean

from vigil.eval.groundtruth import TrueHazard, VideoTruth
from vigil.models.risk import RiskAssessment

LEAD_TOLERANCE_S = 1.0
"""Slack allowed between a claimed warning and the one the footage supports.

One second at the 12 fps the clips are authored at is a frame and a half of sampling error, and
nothing a person can act on. A gap wider than this is the report lying about its own value.
"""


@dataclass(frozen=True)
class Match:
    """One true hazard and the finding that stands for it."""

    truth: TrueHazard
    finding: RiskAssessment
    overlap_s: float
    mechanism_named: bool
    available_lead_s: float | None
    predicted_contact_s: float | None

    @property
    def hazard_id(self) -> str:
        return self.finding.hazard_id

    @property
    def claimed_lead_s(self) -> float | None:
        return self.finding.lead_time_s

    @property
    def over_claim_s(self) -> float | None:
        """How much warning the report claims beyond what the footage had left. None = unknown."""
        if self.available_lead_s is None or self.claimed_lead_s is None:
            return None
        return self.claimed_lead_s - self.available_lead_s

    @property
    def captured_ratio(self) -> float | None:
        if self.available_lead_s is None:
            return None
        return min(1.0, self.available_lead_s / self.truth.anticipatable_s)

    @property
    def timing_error_s(self) -> float | None:
        """Positive when the report places the contact later than it actually lands."""
        if self.predicted_contact_s is None:
            return None
        return self.predicted_contact_s - self.truth.t_impact

    @property
    def severity_delta(self) -> int:
        return self.finding.severity - self.truth.severity

    @property
    def fails(self) -> str | None:
        """Why this pairing would stop a build, or None when it holds up."""
        over = self.over_claim_s
        claimed = self.claimed_lead_s
        available = self.available_lead_s
        if over is None or claimed is None or available is None:
            return None
        if over <= LEAD_TOLERANCE_S:
            return None
        return (
            f"{self.finding.hazard_id} claims {claimed:.1f}s of warning; the footage had "
            f"{available:.1f}s left where the finding was made"
        )


@dataclass(frozen=True)
class ClipScore:
    clip_id: str
    title: str
    is_control: bool
    duration_s: float
    truth: tuple[TrueHazard, ...]
    matches: tuple[Match, ...]
    missed: tuple[TrueHazard, ...]
    unpaired: tuple[RiskAssessment, ...]

    @property
    def false_positives(self) -> int:
        """Unambiguous ones only: a control clip has nothing to pair a finding with."""
        return len(self.unpaired) if self.is_control else 0

    @property
    def detected(self) -> int:
        return len(self.matches)

    def failures(self) -> tuple[str, ...]:
        return tuple(
            f"{self.clip_id}: {reason}"
            for reason in (m.fails for m in self.matches)
            if reason is not None
        )


def _overlap(window: tuple[float, float] | None, hazard: TrueHazard, duration_s: float) -> float:
    if window is None:
        return 0.0
    start = max(window[0], hazard.t_start)
    end = min(window[1], hazard.t_impact)
    if end < start or start > duration_s:
        return 0.0
    return end - start


def _pair_rank(match: Match) -> tuple[bool, float, int]:
    """Right mechanism first, then the most time in common, then the most dangerous."""
    return (match.mechanism_named, match.overlap_s, match.finding.risk_score)


def _candidate(finding: RiskAssessment, hazard: TrueHazard, duration_s: float) -> Match | None:
    overlap = _overlap(finding.clip_window, hazard, duration_s)
    named = finding.hazard_class is not None and finding.hazard_class == hazard.type
    if overlap <= 0.0 and not named:
        return None
    window_end = None if finding.clip_window is None else finding.clip_window[1]
    available = None if window_end is None else max(0.0, hazard.t_impact - window_end)
    predicted = (
        None
        if window_end is None or finding.lead_time_s is None
        else window_end + finding.lead_time_s
    )
    return Match(
        truth=hazard,
        finding=finding,
        overlap_s=overlap,
        mechanism_named=named,
        available_lead_s=available,
        predicted_contact_s=predicted,
    )


def score_clip(truth: VideoTruth, findings: Sequence[RiskAssessment]) -> ClipScore:
    """Greedy one-to-one pairing, highest-confidence pair taken first.

    Greedy rather than optimal because the ranking is lexicographic: no re-assignment can rescue
    a hazard from a finding that named its mechanism, so the order that matters is already fixed
    by the first key.
    """
    pairs = [
        (finding_index, hazard_index, match)
        for finding_index, finding in enumerate(findings)
        for hazard_index, hazard in enumerate(truth.hazards)
        if (match := _candidate(finding, hazard, truth.duration_s)) is not None
    ]
    pairs.sort(key=lambda entry: _pair_rank(entry[2]), reverse=True)
    used_findings: set[int] = set()
    used_hazards: set[int] = set()
    matches: list[Match] = []
    for finding_index, hazard_index, match in pairs:
        if finding_index in used_findings or hazard_index in used_hazards:
            continue
        used_findings.add(finding_index)
        used_hazards.add(hazard_index)
        matches.append(match)
    return ClipScore(
        clip_id=truth.clip_id,
        title=truth.title,
        is_control=truth.is_control,
        duration_s=truth.duration_s,
        truth=tuple(truth.hazards),
        matches=tuple(sorted(matches, key=lambda m: m.truth.t_start)),
        missed=tuple(h for i, h in enumerate(truth.hazards) if i not in used_hazards),
        unpaired=tuple(f for i, f in enumerate(findings) if i not in used_findings),
    )


@dataclass(frozen=True)
class Scorecard:
    """The whole run, reduced to the numbers a reviewer asks for."""

    clips: tuple[ClipScore, ...]
    known_rule_ids: frozenset[str]

    @property
    def hazard_clips(self) -> tuple[ClipScore, ...]:
        return tuple(c for c in self.clips if not c.is_control)

    @property
    def controls(self) -> tuple[ClipScore, ...]:
        return tuple(c for c in self.clips if c.is_control)

    @property
    def truth_hazards(self) -> int:
        return sum(len(c.truth) for c in self.clips)

    @property
    def detected(self) -> int:
        return sum(c.detected for c in self.clips)

    @property
    def recall(self) -> float:
        return self.detected / self.truth_hazards if self.truth_hazards else 1.0

    @property
    def false_positives(self) -> int:
        return sum(c.false_positives for c in self.clips)

    @property
    def control_findings(self) -> int:
        return sum(len(c.unpaired) for c in self.controls)

    @property
    def uncorroborated(self) -> int:
        """Findings on a hazard clip that no sidecar row covers. Reported, never penalised."""
        return sum(len(c.unpaired) for c in self.hazard_clips)

    @property
    def matches(self) -> tuple[Match, ...]:
        return tuple(m for c in self.clips for m in c.matches)

    @property
    def over_claims(self) -> tuple[str, ...]:
        return tuple(f for c in self.clips for f in c.failures())

    @property
    def max_over_claim_s(self) -> float | None:
        values = [m.over_claim_s for m in self.matches if m.over_claim_s is not None]
        return max(values, default=None)

    @property
    def mean_available_lead_s(self) -> float | None:
        values = [m.available_lead_s for m in self.matches if m.available_lead_s is not None]
        return fmean(values) if values else None

    @property
    def mean_claimed_lead_s(self) -> float | None:
        values = [m.claimed_lead_s for m in self.matches if m.claimed_lead_s is not None]
        return fmean(values) if values else None

    @property
    def mean_captured_ratio(self) -> float | None:
        values = [m.captured_ratio for m in self.matches if m.captured_ratio is not None]
        return fmean(values) if values else None

    @property
    def mean_abs_timing_error_s(self) -> float | None:
        values = [abs(m.timing_error_s) for m in self.matches if m.timing_error_s is not None]
        return fmean(values) if values else None

    @property
    def mechanism_accuracy(self) -> float | None:
        if not self.matches:
            return None
        return sum(1 for m in self.matches if m.mechanism_named) / len(self.matches)

    @property
    def severity_within_1(self) -> float | None:
        if not self.matches:
            return None
        return sum(1 for m in self.matches if abs(m.severity_delta) <= 1) / len(self.matches)

    @property
    def citations(self) -> tuple[int, int]:
        """(resolved, filed). A citation that does not resolve in the rulebook is a defect."""
        filed = [c for clip in self.clips for m in clip.matches for c in m.finding.violated_rules]
        filed += [c for clip in self.clips for f in clip.unpaired for c in f.violated_rules]
        resolved = sum(1 for c in filed if c.rule_id in self.known_rule_ids)
        return resolved, len(filed)

    @property
    def escalations(self) -> int:
        return sum(
            1
            for clip in self.clips
            for m in clip.matches
            if m.finding.disposition.value == "stop_work"
        )

    def as_dict(self) -> dict[str, object]:
        """JSON-safe, and the shape the UI and the README table both read."""

        def num(value: float | None, digits: int = 2) -> float | None:
            return None if value is None else round(value, digits)

        resolved, filed = self.citations
        return {
            "clips": len(self.clips),
            "hazard_clips": len(self.hazard_clips),
            "control_clips": len(self.controls),
            "truth_hazards": self.truth_hazards,
            "detected": self.detected,
            "recall": round(self.recall, 4),
            "missed": self.truth_hazards - self.detected,
            "false_positives_on_controls": self.false_positives,
            "uncorroborated_findings": self.uncorroborated,
            "mean_available_lead_s": num(self.mean_available_lead_s),
            "mean_claimed_lead_s": num(self.mean_claimed_lead_s),
            "max_over_claim_s": num(self.max_over_claim_s),
            "mean_captured_ratio": num(self.mean_captured_ratio),
            "mean_abs_timing_error_s": num(self.mean_abs_timing_error_s),
            "mechanism_accuracy": num(self.mechanism_accuracy),
            "severity_within_1": num(self.severity_within_1),
            "citations_resolved": resolved,
            "citations_filed": filed,
            "stop_work_escalations": self.escalations,
            "integrity_failures": list(self.over_claims),
            "per_clip": [
                {
                    "clip_id": clip.clip_id,
                    "is_control": clip.is_control,
                    "truth": [
                        {
                            "type": h.type.value,
                            "t_start": h.t_start,
                            "t_impact": h.t_impact,
                            "anticipatable_s": h.anticipatable_s,
                            "severity": h.severity,
                        }
                        for h in clip.truth
                    ],
                    "missed": [h.type.value for h in clip.missed],
                    "unpaired": [f.hazard_id for f in clip.unpaired],
                    "matches": [
                        {
                            "hazard_id": m.hazard_id,
                            "type": m.truth.type.value,
                            "window": m.finding.clip_window,
                            "claimed_lead_s": num(m.claimed_lead_s),
                            "available_lead_s": num(m.available_lead_s),
                            "over_claim_s": num(m.over_claim_s),
                            "captured_ratio": num(m.captured_ratio),
                            "timing_error_s": num(m.timing_error_s),
                            "mechanism_named": m.mechanism_named,
                            "severity": m.finding.severity,
                            "true_severity": m.truth.severity,
                            "risk_score": m.finding.risk_score,
                            "disposition": m.finding.disposition.value,
                            "citations": [c.rule_id for c in m.finding.violated_rules],
                        }
                        for m in clip.matches
                    ],
                }
                for clip in self.clips
            ],
        }


def score_run(
    results: Sequence[tuple[VideoTruth, Sequence[RiskAssessment]]],
    known_rule_ids: Sequence[str],
) -> Scorecard:
    """Score every clip in one call. Kept separate so a test can hand in a hand-built run."""
    return Scorecard(
        clips=tuple(score_clip(truth, findings) for truth, findings in results),
        known_rule_ids=frozenset(known_rule_ids),
    )


__all__ = [
    "LEAD_TOLERANCE_S",
    "ClipScore",
    "Match",
    "Scorecard",
    "score_clip",
    "score_run",
]
