"""Shared fixtures.

Two rules hold everywhere in this directory: nothing here may touch the network, and nothing
here may need an API key. The code that *does* talk to Nebius is tested against an in-process
transport, because the paths that spend a trial credit are exactly the ones that must not be
discovered broken on stage.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from vigil.config import Settings
from vigil.policy.kb import Rulebook, load_rulebook
from vigil.video.source import VideoSource, load_recordings

from .helpers import CLIP_DIR, POLICY_DIR, ROOT

CONTROL_PREFIX = "control_"


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    """Settings that cannot read a developer's ``.env`` and cannot spend real credit.

    ``_env_file=None`` matters more than it looks: a machine with a live key in ``.env`` would
    otherwise turn a unit test into a bill, and the same suite would pass here and fail there.
    """
    base: dict[str, Any] = {
        "_env_file": None,
        "cache_dir": tmp_path / "cache",
        "artifact_dir": tmp_path / "artifacts",
        "data_dir": ROOT / "data",
        "policy_dir": POLICY_DIR,
        "clip_dir": CLIP_DIR,
        "mock_latency_ms": 0,
        "nebius_api_key": None,
    }
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def tmp_settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def settings_for(tmp_path: Path) -> Callable[..., Settings]:
    """A factory for the odd test that needs one more knob turned."""

    def factory(**overrides: object) -> Settings:
        return make_settings(tmp_path, **overrides)

    return factory


@pytest.fixture(scope="session")
def rulebook() -> Rulebook:
    return load_rulebook(POLICY_DIR)


@pytest.fixture(scope="session")
def authored_videos() -> list[VideoSource]:
    if not (CLIP_DIR / "index.json").is_file():
        pytest.skip("authored clips are missing; run `vigil clips` first")
    return load_recordings(CLIP_DIR)


@pytest.fixture(scope="session")
def hazard_videos(authored_videos: list[VideoSource]) -> list[VideoSource]:
    return [v for v in authored_videos if not v.video_id.startswith(CONTROL_PREFIX)]


@pytest.fixture(scope="session")
def control_videos(authored_videos: list[VideoSource]) -> list[VideoSource]:
    return [v for v in authored_videos if v.video_id.startswith(CONTROL_PREFIX)]


@pytest.fixture(scope="session")
def blind_corner(authored_videos: list[VideoSource]) -> VideoSource:
    """The hardest clip in the set: occlusion, a converging pair, and a missing hi-vis vest."""
    return next(v for v in authored_videos if v.video_id == "blind_corner_struck_by")
