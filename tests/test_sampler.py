"""ffmpeg, ffprobe, and every way a stranger's recording can be awkward.

The rest of Vigil asks this module one question -- give me N pixels between two times -- and
that question gets asked about a file nobody here chose. A phone export with no declared
duration, a camera stream reporting ``0/0`` for its frame rate, a truncated download, a machine
with ffmpeg not on PATH: each of those has to come back as a :class:`MediaError` that says what
to do, or the whole product reads as broken on the reviewer's own footage.

``subprocess.run`` is faked for the parsing and the failures, so these tests run on a machine
with no ffmpeg at all. The two that decode real pixels are guarded by ``skip_without_ffmpeg``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pytest

from vigil.video import sampler
from vigil.video.sampler import (
    MediaError,
    VideoInfo,
    _parse_float,
    _parse_rate,
    plan_size,
    probe,
    sample_frames,
    thumbnail,
)

from .helpers import CLIP_DIR, skip_without_ffmpeg


@dataclass
class Finished:
    returncode: int = 0
    stdout: bytes = b""
    stderr: bytes = b""


@dataclass
class Recorder:
    """The replies ffmpeg/ffprobe would have given, in the order they were asked for."""

    replies: list[Any] = field(default_factory=list)
    commands: list[list[str]] = field(default_factory=list)

    def __call__(self, cmd: list[str], **_kwargs: Any) -> Finished:
        self.commands.append(list(cmd))
        reply = self.replies[min(len(self.commands) - 1, len(self.replies) - 1)]
        return cast("Finished", reply(cmd) if callable(reply) else reply)


@pytest.fixture(autouse=True)
def ffmpeg_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for the tools on a machine that has neither, so no test here is env-dependent.

    Only patched when they are genuinely missing: the real-decoder test below has to see the
    actual PATH to decide whether to skip.
    """
    if shutil.which("ffmpeg") is None:
        monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")


def with_replies(monkeypatch: pytest.MonkeyPatch, recorder: Recorder) -> None:
    monkeypatch.setattr(subprocess, "run", recorder)


def ffprobe(
    streams: list[dict[str, Any]] | None = None, fmt: dict[str, Any] | None = None
) -> bytes:
    return json.dumps(
        {"streams": streams if streams is not None else [{}], "format": fmt or {}}
    ).encode()


STREAM = {"width": 64, "height": 48, "avg_frame_rate": "30/1", "duration": "4.0"}


# ---------------------------------------------------------------------------- the tool itself


def test_a_machine_without_ffmpeg_is_told_the_install_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(MediaError, match="ffmpeg, ffprobe not found on PATH") as excinfo:
        sampler.require_binaries()
    assert "winget install Gyan.FFmpeg" in str(excinfo.value)
    assert "restart the terminal" in str(excinfo.value), "PATH changes do not reach an open shell"


def test_a_timed_out_decode_says_so_instead_of_hanging_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(_cmd: list[str]) -> Finished:
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=120.0)

    with_replies(monkeypatch, Recorder(replies=[explode]))
    with pytest.raises(MediaError, match=r"timed out after 120\.0s"):
        probe("clip.mp4")


def test_the_last_three_lines_of_stderr_are_the_useful_part(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = Recorder(
        replies=[Finished(returncode=1, stderr=b"line one\nline two\nline three\nline four")]
    )
    with_replies(monkeypatch, recorder)
    with pytest.raises(MediaError, match="ffmpeg failed") as excinfo:
        probe("clip.mp4")
    assert "line two | line three | line four" in str(excinfo.value)
    assert "line one" not in str(excinfo.value)


def test_a_failure_that_produces_no_explanation_still_produces_an_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with_replies(monkeypatch, Recorder(replies=[Finished(returncode=134)]))
    with pytest.raises(MediaError, match="exit 134"):
        probe("clip.mp4")


# ----------------------------------------------------------------------------------- probing


def test_a_good_file_reports_what_the_agent_needs_to_plan_with(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with_replies(monkeypatch, Recorder(replies=[Finished(stdout=ffprobe([STREAM]))]))
    info = probe("clip.mp4")
    assert (info.width, info.height) == (64, 48)
    assert info.fps == 30.0
    assert info.duration_s == 4.0
    assert info.est_frames == 120
    assert info.path == "clip.mp4"


def test_a_file_with_no_video_stream_is_named_and_not_guessed_at(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with_replies(monkeypatch, Recorder(replies=[Finished(stdout=ffprobe([]))]))
    with pytest.raises(MediaError, match=r"no video stream in clip\.mp4"):
        probe("clip.mp4")


def test_the_duration_on_the_format_is_used_when_the_stream_omits_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = {k: v for k, v in STREAM.items() if k != "duration"}
    with_replies(
        monkeypatch, Recorder(replies=[Finished(stdout=ffprobe([stream], {"duration": "7.5"}))])
    )
    assert probe("clip.mp4").duration_s == 7.5


def test_a_container_that_declares_no_duration_is_counted_packet_by_packet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real case: an ffmpeg-built MP4 with no duration in its header.

    Without the fallback every window would be zero seconds long, which reads as a recording
    with nothing in it rather than a file whose header is unhelpful.
    """
    stream = {k: v for k, v in STREAM.items() if k != "duration"}
    counted = ffprobe([{"nb_read_packets": "150", "r_frame_rate": "30/1"}])
    recorder = Recorder(replies=[Finished(stdout=ffprobe([stream])), Finished(stdout=counted)])
    with_replies(monkeypatch, recorder)
    info = probe("clip.mp4")
    assert info.duration_s == 5.0
    assert [c for c in recorder.commands if "-count_packets" in c], "the fallback did run"


def test_a_garbage_fallback_answer_is_zero_not_a_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = {k: v for k, v in STREAM.items() if k != "duration"}
    with_replies(
        monkeypatch,
        Recorder(replies=[Finished(stdout=ffprobe([stream])), Finished(stdout=b"not json")]),
    )
    assert probe("clip.mp4").duration_s == 0.0


def test_a_missing_packet_count_does_not_invent_a_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = {k: v for k, v in STREAM.items() if k != "duration"}
    with_replies(
        monkeypatch,
        Recorder(
            replies=[
                Finished(stdout=ffprobe([stream])),
                Finished(stdout=ffprobe([{"r_frame_rate": "25/1"}])),
            ]
        ),
    )
    assert probe("clip.mp4").duration_s == 0.0


def test_a_variable_frame_rate_stream_that_reports_nothing_falls_back_to_a_plain_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = {**STREAM, "avg_frame_rate": "0/0"}
    with_replies(monkeypatch, Recorder(replies=[Finished(stdout=ffprobe([stream]))]))
    assert probe("clip.mp4").fps == 25.0


def test_an_estimate_is_a_videoinfo_that_can_be_printed() -> None:
    """``VideoInfo`` is also built by hand, from a sidecar, without touching a file."""
    info = VideoInfo(path="x", duration_s=6.0, fps=24.0, width=1920, height=1080)
    assert info.est_frames == 144


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("30/1", 30.0),
        ("30000/1001", 29.97002997002997),
        ("25", 25.0),
        ("0/0", 0.0),
        ("", 0.0),
        (None, 0.0),
        (12, 0.0),
        ("1500", 0.0),
        ("abc", 0.0),
        ("10/0", 10.0),
    ],
)
def test_a_frame_rate_arrives_in_more_shapes_than_one(raw: object, expected: float) -> None:
    assert _parse_rate(raw) == expected


@pytest.mark.parametrize(("raw", "expected"), [("2.5", 2.5), ("", 0.0), (None, 0.0), ("x", 0.0)])
def test_a_number_that_is_not_a_number_is_zero(raw: object, expected: float) -> None:
    assert _parse_float(raw) == expected


# ---------------------------------------------------------------------------------- planning


@pytest.mark.parametrize(
    ("width", "height", "max_px", "expected"),
    [
        (64, 48, 512, (64, 48)),
        (65, 49, 512, (64, 48)),
        (1920, 1080, 512, (512, 288)),
        (1080, 1920, 512, (288, 512)),
        (100, 100, 0, (100, 100)),
        (1, 1, 512, (2, 2)),
        (1280, 720, 512, (512, 288)),
        (999, 700, 512, (512, 358)),
    ],
)
def test_the_output_size_keeps_the_aspect_ratio_and_even_edges(
    width: int, height: int, max_px: int, expected: tuple[int, int]
) -> None:
    """rawvideo with an odd width shifts every row by half a pixel, which looks like motion."""
    assert plan_size(width, height, max_px) == expected
    assert all(side % 2 == 0 and side >= 2 for side in expected)


# ------------------------------------------------------------------------------- sampling


def test_asking_for_no_frames_does_not_start_a_process(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder(replies=[Finished(stdout=ffprobe([STREAM]))])
    with_replies(monkeypatch, recorder)
    assert sample_frames("clip.mp4", count=0) == []
    assert recorder.commands == []


def test_a_window_past_the_end_of_the_recording_is_empty_not_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = Recorder(replies=[Finished(stdout=ffprobe([STREAM]))])
    with_replies(monkeypatch, recorder)
    assert sample_frames("clip.mp4", t0=4.0, t1=9.0, count=4) == []
    assert len(recorder.commands) == 1, "the decode is still attempted once, at the last frame"


def test_a_decode_that_yields_nothing_complete_returns_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = Recorder(replies=[Finished(stdout=ffprobe([STREAM])), Finished(stdout=b"")])
    with_replies(monkeypatch, recorder)
    assert sample_frames("clip.mp4", t0=0.0, t1=4.0, count=2) == []


def test_a_decode_larger_than_the_ceiling_is_refused_before_it_is_parsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = Recorder(replies=[Finished(stdout=ffprobe([STREAM])), Finished(stdout=b"\x00" * 64)])
    with_replies(monkeypatch, recorder)
    monkeypatch.setattr(sampler, "MAX_FRAME_BYTES", 32)
    with pytest.raises(MediaError, match="lower frames_per_clip"):
        sample_frames("clip.mp4", t0=0.0, t1=4.0, count=2)


def test_the_frames_come_back_in_order_inside_the_window_at_the_planned_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """64x48 at max_px 64 is 9216 bytes a frame, so two frames are exactly 18432."""
    payload = np.zeros((2, 48, 64, 3), dtype=np.uint8)
    payload[1, :, :, 0] = 255
    recorder = Recorder(
        replies=[
            Finished(stdout=ffprobe([{**STREAM, "duration": "4.0"}])),
            Finished(stdout=payload.tobytes()),
        ]
    )
    with_replies(monkeypatch, recorder)
    frames = sample_frames("clip.mp4", t0=1.0, t1=3.0, count=2, max_px=64)
    assert [f.size for f in frames] == [(64, 48), (64, 48)]
    assert [round(f.t_s, 2) for f in frames] == [1.0, 2.0]
    assert frames[1].array[0, 0, 0] == 255
    command = recorder.commands[-1]
    assert command[0] == "ffmpeg"
    assert "1.000" in command
    assert "3.000" in command
    assert any(part.startswith("fps=") for part in command)


def test_a_window_that_is_not_given_ends_at_the_recording(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = np.zeros((1, 48, 64, 3), dtype=np.uint8)
    with_replies(
        monkeypatch,
        Recorder(replies=[Finished(stdout=ffprobe([STREAM])), Finished(stdout=payload.tobytes())]),
    )
    frames = sample_frames("clip.mp4", count=1, max_px=64)
    assert len(frames) == 1


def test_a_thumbnail_is_a_jpeg_and_an_impossible_moment_is_said_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = np.zeros((1, 48, 64, 3), dtype=np.uint8)
    with_replies(
        monkeypatch,
        Recorder(replies=[Finished(stdout=ffprobe([STREAM])), Finished(stdout=payload.tobytes())]),
    )
    image = thumbnail("clip.mp4", t_s=1.0, max_px=64)
    assert image[:2] == b"\xff\xd8", "JPEG magic bytes -- the browser has to open this"

    with_replies(
        monkeypatch,
        Recorder(replies=[Finished(stdout=ffprobe([STREAM])), Finished(stdout=b"")]),
    )
    with pytest.raises(MediaError, match=r"could not extract a frame at 1\.0s from clip\.mp4"):
        thumbnail("clip.mp4", t_s=1.0)


def test_frames_are_encoded_for_the_wire_one_each(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = np.zeros((2, 48, 64, 3), dtype=np.uint8)
    with_replies(
        monkeypatch,
        Recorder(
            replies=[
                Finished(stdout=ffprobe([STREAM])),
                Finished(stdout=payload.tobytes()),
            ]
        ),
    )
    frames = sample_frames("clip.mp4", t0=0.0, t1=4.0, count=2, max_px=64)
    encoded = sampler.frames_to_base64(frames)
    assert len(encoded) == 2
    assert all(e and e.isascii() for e in encoded)


# --------------------------------------------------------------------------- the real decoder


def test_a_real_recording_decodes_to_real_pixels(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fakes above prove the arithmetic; only ffmpeg proves the pipeline."""
    monkeypatch.undo()  # the autouse PATH stand-in must not decide whether this one skips
    skip_without_ffmpeg()
    clip = next(iter(sorted(CLIP_DIR.glob("*.mp4"))), None)
    assert clip is not None, "the shipped dataset should contain clips"

    info = probe(str(clip))
    assert info.duration_s > 5.0, "the authored clips are a scan-length recording, not a snippet"
    assert info.width > 0
    assert info.fps > 0

    end = min(6.0, info.duration_s)
    frames = sample_frames(str(clip), t0=1.0, t1=end, count=4, max_px=256)
    assert 1 <= len(frames) <= 4
    assert all(f.size[0] <= 256 and f.size[0] % 2 == 0 for f in frames)
    assert frames[0].t_s >= 1.0
    assert frames[-1].t_s <= end
    assert max(f.array.shape[2] for f in frames) == 3
    if len(frames) > 1:
        assert not np.array_equal(frames[0].array, frames[-1].array), (
            "the authored scenes move; two identical frames mean the fps filter is ignored"
        )

    still = thumbnail(str(clip), t_s=1.0, max_px=256)
    assert still[:2] == b"\xff\xd8"
    assert 512 < len(still) < 512 * 1024
