"""Frame extraction via ffmpeg.

Deliberately not OpenCV's ``VideoCapture``: on Windows it cannot open paths containing
non-ASCII characters, and this project sits inside such a directory. Piping decoded
frames straight out of ffmpeg also keeps the whole pipeline in-memory, so nothing is
written next to the user's footage.

Requires ``ffmpeg`` and ``ffprobe`` on PATH (see ``scripts/check_env``).
"""

from __future__ import annotations

import base64
import io
import json
import shutil
import subprocess
from dataclasses import dataclass

import numpy as np
from PIL import Image

MAX_FRAME_BYTES = 25 * 1024 * 1024


class MediaError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoInfo:
    path: str
    duration_s: float
    fps: float
    width: int
    height: int

    @property
    def est_frames(self) -> int:
        return int(self.duration_s * self.fps)


@dataclass(frozen=True)
class Frame:
    t_s: float
    array: np.ndarray  # HxWx3 uint8, RGB

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.array.shape[:2]
        return w, h

    def to_jpeg_base64(self, quality: int = 82) -> str:
        return base64.b64encode(self.to_jpeg_bytes(quality)).decode("ascii")

    def to_jpeg_bytes(self, quality: int = 82) -> bytes:
        buf = io.BytesIO()
        Image.fromarray(self.array, mode="RGB").save(
            buf, format="JPEG", quality=quality, optimize=True
        )
        return buf.getvalue()


def require_binaries() -> None:
    missing = [b for b in ("ffmpeg", "ffprobe") if shutil.which(b) is None]
    if missing:
        raise MediaError(
            f"{', '.join(missing)} not found on PATH. Install ffmpeg "
            "(winget install Gyan.FFmpeg / brew install ffmpeg) and restart the terminal."
        )


def _run(cmd: list[str], *, timeout: float = 120.0) -> bytes:
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False, timeout=timeout)
    except FileNotFoundError as exc:  # pragma: no cover - guarded by require_binaries
        raise MediaError(f"{exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaError(f"timed out after {timeout}s: {' '.join(cmd[:4])}…") from exc
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        tail = " | ".join(detail[-3:]) if detail else f"exit {proc.returncode}"
        raise MediaError(f"ffmpeg failed: {tail}")
    return proc.stdout


def probe(path: str) -> VideoInfo:
    require_binaries()
    out = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,duration",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            path,
        ]
    )
    try:
        blob = json.loads(out.decode("utf-8", "replace") or "{}")
    except json.JSONDecodeError as exc:  # pragma: no cover
        raise MediaError(f"unparseable ffprobe output for {path}") from exc
    streams = blob.get("streams") or []
    if not streams:
        raise MediaError(f"no video stream in {path}")
    stream = streams[0]
    width, height = int(stream["width"]), int(stream["height"])
    fps = _parse_rate(stream.get("avg_frame_rate"))
    duration = _parse_float(stream.get("duration")) or _parse_float(
        (blob.get("format") or {}).get("duration")
    )
    if not duration:
        duration = _duration_by_demux(path)
    if not fps:
        fps = 25.0
    return VideoInfo(path=path, duration_s=duration, fps=fps, width=width, height=height)


def _duration_by_demux(path: str) -> float:
    """Fallback for containers that declare no duration: count packets, divide by rate."""
    out = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets,r_frame_rate",
            "-of",
            "json",
            path,
        ],
        timeout=300.0,
    )
    try:
        stream = (json.loads(out.decode("utf-8", "replace") or "{}").get("streams") or [{}])[0]
    except json.JSONDecodeError:
        return 0.0
    packets = _parse_float(stream.get("nb_read_packets"))
    rate = _parse_rate(stream.get("r_frame_rate")) or 25.0
    return packets / rate if packets else 0.0


def _parse_rate(raw: object) -> float:
    if not isinstance(raw, str) or not raw or raw == "0/0":
        return 0.0
    if "/" in raw:
        num, _, den = raw.partition("/")
        value = _parse_float(num) / (_parse_float(den) or 1.0)
    else:
        value = _parse_float(raw)
    return value if 0.0 < value < 1000.0 else 0.0


def _parse_float(raw: object) -> float:
    try:
        return float(str(raw))
    except (TypeError, ValueError):
        return 0.0


def _even(value: float) -> int:
    rounded = round(value)
    return max(2, rounded - (rounded % 2))


def plan_size(width: int, height: int, max_px: int) -> tuple[int, int]:
    """Scale so the long edge is <= ``max_px``, keeping even dimensions for rawvideo."""
    if max_px <= 0 or max(width, height) <= max_px:
        return _even(width), _even(height)
    scale = max_px / max(width, height)
    return _even(width * scale), _even(height * scale)


def sample_frames(
    path: str,
    *,
    t0: float = 0.0,
    t1: float | None = None,
    count: int = 8,
    max_px: int = 768,
) -> list[Frame]:
    """Return up to ``count`` RGB frames evenly covering [t0, t1)."""
    require_binaries()
    if count < 1:
        return []
    info = probe(path)
    t0 = max(0.0, min(t0, info.duration_s))
    t1 = info.duration_s if t1 is None else min(max(t1, t0 + 1e-3), info.duration_s)
    window = t1 - t0
    if window <= 0:
        return []

    want = min(count, max(1, int(window * info.fps)))
    out_w, out_h = plan_size(info.width, info.height, max_px)
    fps_filter = max(0.05, want / window)

    raw = _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{t0:.3f}",
            "-to",
            f"{t1:.3f}",
            "-i",
            path,
            "-vf",
            f"fps={fps_filter:.6f},scale={out_w}:{out_h}:flags=bicubic",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        timeout=max(60.0, window * 20),
    )
    if len(raw) > MAX_FRAME_BYTES:
        raise MediaError(
            f"decoded frame payload exceeded {MAX_FRAME_BYTES} bytes; lower frames_per_clip"
        )
    frame_bytes = out_w * out_h * 3
    n = len(raw) // frame_bytes
    if n == 0:
        return []
    cube = np.frombuffer(raw[: n * frame_bytes], dtype=np.uint8).reshape(n, out_h, out_w, 3)
    step = window / n
    return [Frame(t_s=min(t0 + i * step, t1), array=cube[i].copy()) for i in range(n)]


def thumbnail(path: str, *, t_s: float = 0.0, max_px: int = 480, quality: int = 78) -> bytes:
    frames = sample_frames(path, t0=t_s, t1=t_s + 0.5, count=1, max_px=max_px)
    if not frames:
        raise MediaError(f"could not extract a frame at {t_s:.1f}s from {path}")
    return frames[0].to_jpeg_bytes(quality)


def frames_to_base64(frames: list[Frame], quality: int = 82) -> list[str]:
    return [f.to_jpeg_base64(quality) for f in frames]
