"""What the agent is pointed at: one camera's recording, addressable by time."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from vigil.perception.base import Clip


class VideoSource(BaseModel):
    """A single continuous recording from a fixed camera.

    ``duration_s`` is required rather than probed on demand: every window the agent asks for
    is clamped against it, and a probe per relook would add a subprocess call to each turn.
    """

    model_config = ConfigDict(extra="forbid")

    video_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    camera_label: str = "cam-1"
    duration_s: float = Field(gt=0.0, le=86_400.0)
    source_fps: float = Field(default=0.0, ge=0.0, description="0 when unknown; reporting only.")
    site: str = ""
    area_type: str = ""
    lighting: str = ""
    surface: str = ""

    @model_validator(mode="after")
    def _file_exists(self) -> Self:
        if not Path(self.path).is_file():
            raise ValueError(f"video file not found: {self.path}")
        return self

    def window(self, t0_s: float, t1_s: float) -> tuple[float, float]:
        """Clamp a requested window into this recording, or say why it cannot be used."""
        t0 = max(0.0, min(t0_s, self.duration_s))
        t1 = max(0.0, min(t1_s, self.duration_s))
        if t1 <= t0:
            raise ValueError(
                f"window {t0_s:.1f}..{t1_s:.1f}s is empty once clamped to a "
                f"{self.duration_s:.1f}s recording"
            )
        return t0, t1

    def clip(self, t0_s: float, t1_s: float) -> Clip:
        t0, t1 = self.window(t0_s, t1_s)
        return Clip(
            video_id=self.video_id,
            path=self.path,
            t0_s=t0,
            t1_s=t1,
            camera_label=self.camera_label,
        )

    def scan(self, *, clip_seconds: float, max_clips: int, overlap: float = 0.0) -> list[Clip]:
        """Windows covering the recording, for the opening survey.

        Overlap is what buys anticipation. A warning can only be counted from a window that
        *ends* before the predicted contact, so a contiguous scan gets zero credit for a hazard
        that starts and finishes inside one window however early the read was. With 0.5 overlap
        there is no gap in coverage, so a hazard with any warning available gets it.
        """
        span = min(clip_seconds, self.duration_s)
        if self.duration_s <= span:
            return [self.clip(0.0, self.duration_s)]
        stride = span * (1.0 - min(0.95, max(0.0, overlap)))
        n = math.ceil((self.duration_s - span) / stride) + 1
        if n > max_clips:
            # Fewer windows, same length: the survey loses overlap before it loses resolution,
            # because a short window that misses the hazard entirely is the expensive failure.
            n = max_clips
            stride = (self.duration_s - span) / (n - 1)
        return [self.clip(i * stride, i * stride + span) for i in range(n)]

    @classmethod
    def from_record(cls, record: dict[str, object], clip_dir: Path) -> Self:
        """Build from a ground-truth sidecar written by ``vigil.video.synth``."""
        scene = record.get("scene")
        meta = scene if isinstance(scene, dict) else {}
        duration = record["duration_s"]
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            raise ValueError(f"sidecar duration_s must be a number, got {duration!r}")
        return cls(
            video_id=str(record["clip_id"]),
            path=str((clip_dir / str(record["path"])).resolve()),
            camera_label=str(record.get("camera_label", "cam-1")),
            duration_s=float(duration),
            site=str(record.get("site", "")),
            area_type=str(meta.get("area_type", "")),
            lighting=str(meta.get("lighting", "")),
            surface=str(meta.get("surface", "")),
        )


def load_recordings(clip_dir: Path) -> list[VideoSource]:
    """Every recording listed by the ground-truth index that ``vigil.video.synth`` wrote.

    Index order, not filename order: the authored clips are listed hardest-first, so a
    truncated demo run still shows the interesting ones.
    """
    path = Path(clip_dir) / "index.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing -- synthesise the sample footage first.")
    records: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"{path} should hold a list of clip records, got {type(records).__name__}")
    videos: list[VideoSource] = []
    broken: list[str] = []
    for item in records:
        if not isinstance(item, dict):
            broken.append(f"<{type(item).__name__}>")
            continue
        try:
            videos.append(VideoSource.from_record(item, path.parent))
        except ValidationError as exc:
            # One line per bad clip. A raw pydantic dump buries the clip id inside a repr of
            # every field, and the person reading it is trying to fix a stale index.
            reason = str(exc.errors()[0].get("msg", exc)).removeprefix("Value error, ")
            broken.append(f"{item.get('clip_id', '?')}: {reason}")
        except (ValueError, KeyError, TypeError) as exc:
            broken.append(f"{item.get('clip_id', '?')}: {exc}")
    if broken:
        raise ValueError(f"{path} has unusable entries -- " + "; ".join(broken))
    return videos
