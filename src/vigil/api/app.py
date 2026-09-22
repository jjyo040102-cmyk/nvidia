"""FastAPI surface for judges and operators.

The web layer is intentionally thin: it lists the same recordings the CLI sees and calls the
same :func:`vigil.agent.investigate` entry point.  There is no second "demo implementation" to
drift away from the product.  The only web-specific policy is mode selection: ``demo`` is
explicitly scripted, while ``live`` is explicitly Nebius and is never allowed to fall back.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from vigil import __version__
from vigil.agent import investigate
from vigil.config import Settings
from vigil.nebius import BudgetExceeded, NebiusError
from vigil.video.source import VideoSource, load_recordings

Mode = Literal["demo", "live"]
_STATIC = Path(__file__).with_name("static")


def _videos(settings: Settings) -> list[VideoSource]:
    try:
        return load_recordings(settings.clip_dir)
    except (FileNotFoundError, ValueError) as exc:
        detail = f"sample footage is unavailable: {exc}"
        raise HTTPException(status_code=503, detail=detail) from exc


def _pick(videos: list[VideoSource], video_id: str) -> VideoSource:
    for video in videos:
        if video.video_id == video_id:
            return video
    available = ", ".join(video.video_id for video in videos)
    raise HTTPException(
        status_code=404,
        detail=f"unknown sample {video_id!r}; available: {available}",
    )


def _run_settings(base: Settings, mode: Mode) -> Settings:
    if mode == "demo":
        return base.model_copy(
            update={"perception_backend": "mock", "reasoning_backend": "mock"}
        )
    if not base.has_nebius_key:
        raise HTTPException(
            status_code=400,
            detail=(
                "LIVE mode requires VIGIL_NEBIUS_API_KEY. Vigil never falls back to scripted "
                "mock models after LIVE has been selected; add the key or switch to DEMO."
            ),
        )
    return base.model_copy(
        update={"perception_backend": "nebius", "reasoning_backend": "nebius"}
    )


def _summary(report: object) -> str:
    # Kept separate from the API model so the dashboard can show a compact status sentence
    # without needing to replicate business rules in JavaScript.
    assessments = getattr(report, "assessments", [])
    lead = getattr(report, "lead_time_s", None)
    if not assessments:
        return "No finding met the reporting bar; the complete evidence trace is still available."
    worst = getattr(report, "worst", None)
    score = getattr(worst, "risk_score", "?") if worst is not None else "?"
    lead_text = (
        f"{lead:.1f}s warning"
        if isinstance(lead, (int, float))
        else "warning time unresolved"
    )
    return f"{len(assessments)} finding(s), highest risk {score}/20, {lead_text}."


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build an app around explicit settings; tests use this to guarantee no network key leaks."""
    base = settings or Settings()
    web = FastAPI(
        title="Vigil",
        version=__version__,
        description="Anticipatory physical-safety investigation over CCTV recordings.",
    )
    web.state.settings = base
    web.mount("/static", StaticFiles(directory=_STATIC), name="static")

    @web.get("/", include_in_schema=False)
    async def dashboard() -> FileResponse:
        return FileResponse(_STATIC / "index.html")

    @web.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "service": "vigil", "version": __version__}

    @web.get("/api/status")
    async def status() -> dict[str, object]:
        videos = _videos(base)
        return {
            "has_nebius_key": base.has_nebius_key,
            "default_mode": "live" if base.has_nebius_key else "demo",
            "demo_is_measurement": False,
            "sample_count": len(videos),
            "resolved_perception": base.resolved_perception,
            "resolved_reasoning": base.resolved_reasoning,
            "provider": "Nebius Token Factory",
        }

    @web.get("/api/samples")
    async def samples() -> dict[str, object]:
        videos = _videos(base)
        return {
            "samples": [
                {
                    "video_id": video.video_id,
                    "camera_label": video.camera_label,
                    "duration_s": video.duration_s,
                    "site": video.site,
                    "area_type": video.area_type,
                    "lighting": video.lighting,
                    "surface": video.surface,
                    "is_control": video.video_id.startswith("control_"),
                    "media_url": f"/media/{video.video_id}",
                }
                for video in videos
            ]
        }

    @web.get("/media/{video_id}", include_in_schema=False)
    async def media(video_id: str) -> FileResponse:
        video = _pick(_videos(base), video_id)
        return FileResponse(video.path, media_type="video/mp4")

    @web.post("/api/investigate/{video_id}")
    async def run_investigation(video_id: str, mode: Mode = "demo") -> dict[str, object]:
        video = _pick(_videos(base), video_id)
        run_settings = _run_settings(base, mode)
        try:
            outcome = await investigate(video, run_settings)
        except BudgetExceeded as exc:
            raise HTTPException(status_code=429, detail=f"run budget reached: {exc}") from exc
        except NebiusError as exc:
            raise HTTPException(status_code=502, detail=f"Nebius inference failed: {exc}") from exc

        report = outcome.report
        return {
            "mode": mode,
            "measured": mode == "live",
            "incomplete": bool(outcome.stopped),
            "stopped": outcome.stopped,
            "summary": _summary(report),
            "spend": outcome.spend or {},
            "report": report.model_dump(mode="json"),
        }

    return web


app = create_app()
