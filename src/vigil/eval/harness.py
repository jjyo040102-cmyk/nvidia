"""``vigil eval`` -- run every authored clip and score what comes back.

The product's claim is anticipatory: Vigil warns before the contact. That is a number, and a
number a judge can check is worth more than a demo that looks good. The sidecar ground truth
authored with the clips is what makes the claim falsifiable, so this harness refuses to report a
scripted run as a measurement -- a rule-based controller scoring itself against scenarios it was
written from is a mirror, not a benchmark.

One shared client and one shared rulebook across the whole sweep, wrapped in one spend ledger,
because the trial credit is finite and an honest output includes what it cost.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from vigil.agent import investigate
from vigil.config import Settings
from vigil.eval.groundtruth import VideoTruth, load_truth
from vigil.eval.metrics import ClipScore, Scorecard, score_clip, score_run
from vigil.models.risk import RiskAssessment
from vigil.nebius import NebiusClient, Spend
from vigil.policy.kb import load_rulebook
from vigil.video.source import VideoSource, load_recordings

Progress = Callable[[VideoTruth, ClipScore, str], None]


@dataclass(frozen=True)
class Skipped:
    """A clip that could not be scored, and the sentence explaining why."""

    clip_id: str
    reason: str


@dataclass(frozen=True)
class EvalRun:
    card: Scorecard
    skipped: tuple[Skipped, ...]
    measured: bool
    backends: dict[str, str]
    models: tuple[str, ...]
    spend: dict[str, Any]
    seconds: float

    @property
    def integrity_ok(self) -> bool:
        return not self.card.over_claims

    def verdict(self) -> tuple[str, int]:
        """The last line the command prints, and its exit code.

        A scripted run exits non-zero on purpose. It is the only defence against one of these
        tables being pasted into a README as though it measured a model -- by whoever types the
        command, including a future version of us holding a key and a deadline.
        """
        if not self.measured:
            return (
                "NOT A MEASUREMENT: scripted backends scored themselves against scenarios "
                "written for them. Re-run with --perception nebius --reasoning nebius.",
                1,
            )
        if self.card.over_claims:
            return (
                f"{len(self.card.over_claims)} finding(s) claimed more warning than the footage "
                "held. Those are defects, not misses.",
                1,
            )
        if self.skipped:
            return (f"{len(self.skipped)} clip(s) had no ground truth and were not scored.", 1)
        return (
            f"{self.card.detected}/{self.card.truth_hazards} hazards anticipated, "
            f"{self.card.false_positives} false positive(s) on controls",
            0,
        )

    def as_dict(self, *, generated_at: datetime | None = None) -> dict[str, Any]:
        stamp = generated_at or datetime.now(UTC)
        message, code = self.verdict()
        return {
            "generated_at": stamp.isoformat(timespec="seconds"),
            "measured": self.measured,
            "exit_code": code,
            "verdict": message,
            "backends": self.backends,
            "models": list(self.models),
            "wall_clock_s": round(self.seconds, 1),
            "spend": self.spend,
            "skipped": [{"clip_id": s.clip_id, "reason": s.reason} for s in self.skipped],
            "summary": self.card.as_dict(),
        }


def is_measurement(settings: Settings) -> bool:
    """Only a run that could have been wrong on its own initiative counts as a measurement.

    The scripted planner and the scripted reads are both derived from the same authored
    scenarios as the ground truth, so their score is an echo. Cosmos and Token Factory are not.
    """
    return settings.resolved_perception in {"nebius", "cosmos"} and (
        settings.resolved_reasoning == "nebius"
    )


async def evaluate(
    settings: Settings,
    *,
    clip_ids: Sequence[str] | None = None,
    on_progress: Progress | None = None,
) -> EvalRun:
    """Investigate every clip that has ground truth and score the lot."""
    started = datetime.now(UTC)
    truth = load_truth(settings.clip_dir)
    videos = [
        video
        for video in load_recordings(settings.clip_dir)
        if clip_ids is None or video.video_id in set(clip_ids)
    ]
    skipped: list[Skipped] = []
    chosen: list[tuple[VideoSource, VideoTruth]] = []
    for video in videos:
        sidecar = truth.get(video.video_id)
        if sidecar is None:
            skipped.append(Skipped(video.video_id, "no ground-truth sidecar beside it"))
            continue
        chosen.append((video, sidecar))

    rulebook = load_rulebook(settings.policy_dir)
    hosted = "nebius" in {settings.resolved_perception, settings.resolved_reasoning}
    client = NebiusClient(settings) if hosted else None
    spend = Spend()
    produced: list[tuple[VideoTruth, Sequence[RiskAssessment]]] = []
    try:
        for video, sidecar in chosen:
            outcome = await investigate(
                video, settings, rulebook=rulebook, client=client, spend=spend
            )
            produced.append((sidecar, outcome.findings))
            if on_progress is not None:
                on_progress(sidecar, score_clip(sidecar, outcome.findings), outcome.stopped)
    finally:
        if client is not None:
            await client.aclose()

    seconds = (datetime.now(UTC) - started).total_seconds()
    models = sorted(spend.by_model)
    measured = is_measurement(settings)
    return EvalRun(
        card=score_run(produced, [rule.id for rule in rulebook.rules]),
        skipped=tuple(skipped),
        measured=measured,
        backends={
            "perception": settings.resolved_perception,
            "reasoning": settings.resolved_reasoning,
        },
        models=tuple(models) if models else _labels(settings),
        spend=spend.as_dict(),
        seconds=seconds,
    )


def _labels(settings: Settings) -> tuple[str, ...]:
    """Name what ran, so a table can never be mistaken for a model's numbers."""
    perception = (
        "scripted reads of the authored scenarios"
        if settings.resolved_perception == "mock"
        else settings.resolved_perception
    )
    reasoning = (
        "rule-based controller"
        if settings.resolved_reasoning in {"mock", "scripted"}
        else settings.resolved_reasoning
    )
    return (f"perception: {perception}", f"reasoning: {reasoning}")
