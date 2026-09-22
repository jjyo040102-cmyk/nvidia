"""The command line: build the footage, run an investigation, hand over a report.

Kept thin on purpose. Every command assembles :class:`Settings`, calls into the layer that
owns the work, and writes what comes back -- no policy, no scoring, no printing logic that a
test could not exercise through the same function.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast

import typer

from vigil.agent import investigate
from vigil.config import Backend, Settings
from vigil.eval import ClipScore, VideoTruth, evaluate, render
from vigil.models.report import IncidentReport
from vigil.nebius import NebiusError
from vigil.nebius import probe as probe_account
from vigil.report.markdown import render_report
from vigil.video.scenarios import all_scenarios
from vigil.video.source import VideoSource, load_recordings
from vigil.video.synth import generate_all

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Vigil -- anticipatory safety investigation over CCTV recordings.",
)

BACKENDS = ("auto", "mock", "nebius", "cosmos")


def _settings(
    *,
    perception: str,
    reasoning: str,
    turns: int | None = None,
    usd_budget: float | None = None,
    clip_seconds: float | None = None,
    overlap: float | None = None,
    frames: int | None = None,
) -> Settings:
    overrides: dict[str, Any] = {
        "perception_backend": _backend(perception),
        "reasoning_backend": _backend(reasoning),
    }
    for key, value in (
        ("max_iterations", turns),
        ("usd_budget", usd_budget),
        ("clip_seconds", clip_seconds),
        ("clip_overlap", overlap),
        ("frames_per_clip", frames),
    ):
        if value is not None:
            overrides[key] = value
    settings = Settings(**overrides)
    _require_key_for(settings)
    return settings


def _require_key_for(settings: Settings) -> None:
    """Naming a hosted backend is a promise that there is an account behind it.

    Without this, ``vigil run --perception nebius`` before the key is set dies inside
    ``NebiusClient.__init__`` as a traceback several frames deep -- which reads as a broken tool
    to the one person who can fix it by pasting a key into ``.env``.
    """
    if "nebius" not in {settings.resolved_perception, settings.resolved_reasoning}:
        return
    if settings.has_nebius_key:
        return
    typer.secho(
        "a Nebius backend was asked for but no key is set. Put VIGIL_NEBIUS_API_KEY in .env "
        "(Token Factory console), or run with --perception mock --reasoning mock, which needs "
        "no account and no credit.",
        fg=typer.colors.RED,
        err=True,
    )
    raise typer.Exit(code=2)


def _backend(name: str) -> Backend:
    """Validate at the boundary, then hand the Literal to Settings."""
    if name not in BACKENDS:
        raise typer.BadParameter(f"choose one of {', '.join(BACKENDS)}")
    return cast("Backend", name)


def _write(report: IncidentReport, out_dir: Path, *, trace: bool) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = report.model_dump(mode="json")
    if not trace:
        payload.pop("trace", None)
    stem = out_dir / report.video_id
    json_path = stem.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path = stem.with_suffix(".md")
    md_path.write_text(render_report(report), encoding="utf-8")
    return json_path, md_path


@app.command()
def clips(
    out: Path = typer.Option(Path("data/clips"), help="Where to write the mp4s and sidecars."),
    force: bool = False,
) -> None:
    """Synthesise the sample dataset: authored scenarios, ground truth, and hazard-free controls."""
    records = generate_all(all_scenarios(), out, force=force)
    controls = sum(1 for r in records if not r["hazards"])
    typer.echo(
        f"{len(records)} clip(s) in {out} -- ground truth beside each one, {controls} controls."
    )


@app.command()
def probe() -> None:
    """Ask the Nebius account what it can actually serve, before spending anything on it."""
    settings = Settings()
    if not settings.has_nebius_key:
        raise typer.Exit(code=2)
    try:
        result = asyncio.run(probe_account(settings))
    except NebiusError as exc:
        typer.secho(f"probe failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"key {settings.key_fingerprint}  base {result['base_url']}  {result['model_count']} models"
    )
    for kind in ("reasoning", "vision"):
        typer.echo(f"  picked {kind:9s} -> {result[f'picked_{kind}']}")
    sample = [
        m
        for m in result["models"]
        if any(w in m.lower() for w in ("nemotron", "cosmos", "vision", "omni"))
    ]
    typer.echo(f"  candidates: {', '.join(sample[:12]) or 'none'}")


@app.command()
def run(
    video: str = typer.Option(
        "all", "--video", "-v", help="Clip id, or 'all' for everything in the index."
    ),
    perception: str = typer.Option("auto", help="Who reads the pixels."),
    reasoning: str = typer.Option("auto", help="Who runs the agent loop."),
    turns: int = typer.Option(
        8, min=1, max=24, help="Planning turns before the agent is forced to conclude."
    ),
    usd_budget: float | None = typer.Option(
        None, help="Refuse calls that would take one run past this."
    ),
    clip_seconds: float | None = typer.Option(
        None, help="Length of each window in the opening scan."
    ),
    overlap: float | None = typer.Option(None, help="Overlap between scan windows, 0..0.95."),
    frames: int | None = typer.Option(
        None, min=2, max=32, help="Frames per window sent to vision."
    ),
    out: Path = typer.Option(
        Path("artifacts/reports"), help="Where the JSON and Markdown reports go."
    ),
    trace: bool = typer.Option(True, help="Keep the step-by-step audit trail inside the report."),
) -> None:
    """Investigate recordings and write a report per video."""
    settings = _settings(
        perception=perception,
        reasoning=reasoning,
        turns=turns,
        usd_budget=usd_budget,
        clip_seconds=clip_seconds,
        overlap=overlap,
        frames=frames,
    )
    videos = _select(load_recordings(settings.clip_dir), video)
    typer.echo(
        f"vigil: {len(videos)} video(s)  perception={settings.resolved_perception}  "
        f"reasoning={settings.resolved_reasoning}  "
        f"windows={settings.clip_seconds}s@{settings.clip_overlap:.0%} overlap"
    )
    failed = 0
    for source in videos:
        outcome = asyncio.run(investigate(source, settings))
        report = outcome.report
        json_path, md_path = _write(report, out, trace=trace)
        lead = f"{report.lead_time_s:.1f}s" if report.lead_time_s is not None else "--"
        steps = len(report.trace.steps) if report.trace else 0
        typer.echo(
            f"  {source.video_id:30s} {len(report.assessments)} finding(s)  "
            f"warning {lead:>5s} ahead  {steps} steps  -> {md_path.name}"
        )
        for line in report.narrative.splitlines()[1:]:
            typer.secho(f"      {line}", fg=typer.colors.YELLOW)
        if outcome.stopped:
            failed += 1
            typer.secho(
                f"      incomplete: {outcome.stopped} -- see {json_path.name}", fg=typer.colors.RED
            )
    if settings.resolved_perception == "mock" or settings.resolved_reasoning == "mock":
        typer.secho(
            "  These ran on the scripted backends: they prove the pipeline works end to end, "
            "not that the models are accurate.",
            fg=typer.colors.CYAN,
        )
    if failed:
        raise typer.Exit(code=1)


@app.command()
def demo(
    out: Path = typer.Option(
        Path("artifacts/reports"), help="Where the JSON and Markdown reports go."
    ),
    turns: int = typer.Option(8, min=1, max=24),
) -> None:
    """The whole product, end to end, with no credentials and no API cost."""
    run(
        video="all",
        perception="mock",
        reasoning="mock",
        turns=turns,
        usd_budget=None,
        clip_seconds=None,
        overlap=None,
        frames=None,
        out=out,
        trace=True,
    )


@app.command("eval")
def benchmark(
    clips: str = typer.Option("all", "--clips", help="Clip id(s), comma separated, or 'all'."),
    perception: str = typer.Option("auto", help="Who reads the pixels."),
    reasoning: str = typer.Option("auto", help="Who runs the agent loop."),
    turns: int = typer.Option(8, min=1, max=24, help="Planning turns per clip."),
    usd_budget: float | None = typer.Option(None, help="Refuse calls past this."),
    out: Path = typer.Option(Path("artifacts/eval"), help="Where the table and the JSON go."),
) -> None:
    """Score every authored clip against its ground truth.

    This is the command that decides which numbers are allowed in a README, so it refuses to
    present a scripted run as a measurement and exits non-zero when it gets one.
    """
    settings = _settings(
        perception=perception, reasoning=reasoning, turns=turns, usd_budget=usd_budget
    )
    known = [v.video_id for v in load_recordings(settings.clip_dir)]
    wanted = None if clips in ("all", "*") else [c.strip() for c in clips.split(",") if c.strip()]
    if wanted is not None:
        missing = [c for c in wanted if c not in known]
        if missing:
            raise typer.BadParameter(
                f"no clip(s) {', '.join(missing)} in the index; available: {', '.join(known)}"
            )
    typer.echo(
        f"vigil eval: scoring against ground truth in {settings.clip_dir}  "
        f"perception={settings.resolved_perception}  reasoning={settings.resolved_reasoning}"
    )

    def progress(truth: VideoTruth, score: ClipScore, stopped: str) -> None:
        tail = f"  [{stopped}]" if stopped else ""
        typer.echo(
            f"  {truth.clip_id:32s} {score.detected}/{len(score.truth)} anticipated, "
            f"{len(score.unpaired)} unpaired finding(s), "
            f"{score.false_positives} false positive(s){tail}"
        )

    run = asyncio.run(evaluate(settings, clip_ids=wanted, on_progress=progress))
    out.mkdir(parents=True, exist_ok=True)
    (out / "latest.json").write_text(
        json.dumps(run.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "latest.md").write_text(render(run) + "\n", encoding="utf-8")
    typer.echo("")
    typer.echo(render(run))
    message, code = run.verdict()
    typer.secho(message, fg=typer.colors.GREEN if code == 0 else typer.colors.RED)
    typer.echo(f"  written to {out / 'latest.md'}")
    raise typer.Exit(code=code)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Interface for the judge dashboard."),
    port: int = typer.Option(8000, min=1, max=65535, help="TCP port for the dashboard."),
    reload: bool = typer.Option(False, help="Reload on source changes; development only."),
) -> None:
    """Launch the browser dashboard and API."""
    import uvicorn

    typer.echo(f"Vigil dashboard: http://{host}:{port}")
    typer.echo(
        "DEMO needs no credentials; LIVE requires VIGIL_NEBIUS_API_KEY and never falls back."
    )
    uvicorn.run("vigil.api.app:app", host=host, port=port, reload=reload)


def _select(videos: list[VideoSource], wanted: str) -> list[VideoSource]:
    if wanted in ("all", "*"):
        return videos
    chosen = [v for v in videos if v.video_id == wanted]
    if not chosen:
        raise typer.BadParameter(
            f"no clip {wanted!r} in the index; available: {', '.join(v.video_id for v in videos)}"
        )
    return chosen


if __name__ == "__main__":
    app()
