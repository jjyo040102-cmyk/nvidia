"""Test-only helpers.

The bar for living here is "imported by more than one test module". What clears it are the
seams that let the code which *spends the trial credit* be tested without a credit: a
transport that never leaves the process, a canned completion, and a model catalog to pick
from. Those paths are exactly the ones that must not be discovered broken on stage.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from vigil.config import Settings
from vigil.models.risk import Likelihood, RiskAssessment
from vigil.nebius import NebiusClient, Spend

ROOT = Path(__file__).resolve().parents[1]
CLIP_DIR = ROOT / "data" / "clips"
POLICY_DIR = ROOT / "data" / "policy"

Handler = Callable[[httpx.Request], httpx.Response]

CATALOG: dict[str, Any] = {
    "data": [
        {"id": "nvidia/nemotron-3-nano-omni-30b-a3b"},
        {"id": "nvidia/nemotron-3-super-120b-a12b"},
        {"id": "Qwen/Qwen2.5-VL-7B-Instruct"},
        {"id": "nvidia/nemotron-3-nano-omni-30b-a3b"},
    ]
}

# What ``auto`` resolves the vision slot to on a well-stocked account, so a test can pin the
# model name instead of depending on the order of VISION_PREFERENCE.
VISION_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b"


def completion(text: str, *, model: str = "m", **usage: int) -> httpx.Response:
    body: dict[str, Any] = {
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": text}}],
    }
    if usage:
        body["usage"] = usage
    return httpx.Response(200, json=body)


def client_with(
    settings: Settings,
    handler: Handler,
    calls: list[str],
    *,
    spend: Spend | None = None,
) -> NebiusClient:
    """A client whose every request is recorded, against a transport that never leaves the box."""

    async def endpoint(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.method} {request.url.path}")
        return handler(request)

    return NebiusClient(settings, spend=spend, transport=httpx.MockTransport(endpoint))


def skip_without_ffmpeg() -> None:
    """The renderer and the sampler both shell out; a machine without ffmpeg is a skip, not a
    failure, so the rest of the suite still runs there."""
    from vigil.video.sampler import MediaError, require_binaries

    try:
        require_binaries()
    except MediaError as exc:
        pytest.skip(str(exc))


def make_assessment(**overrides: object) -> RiskAssessment:
    """A complete finding with the authored blind-corner collision as its default subject."""
    base: dict[str, object] = {
        "hazard_id": "H1",
        "title": "Struck-by between converging routes: P1 and FL1",
        "severity": 5,
        "likelihood": Likelihood.IMMINENT,
        "predicted_event": "The truck reaches the crossing point as the worker steps into it.",
        "lead_time_s": 2.9,
    }
    base.update(overrides)
    return RiskAssessment(**base)


def touch_video(directory: Path, name: str) -> Path:
    """A file that exists. ``VideoSource`` checks presence, not decodability.

    Deliberately empty: these tests are about windows over a recording, and a real mp4 per
    case would make the suite slow for no extra truth. The paths that do decode pixels are
    exercised against the authored clips, in ``test_synth.py`` and ``test_perception_nebius.py``.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"")
    return path
