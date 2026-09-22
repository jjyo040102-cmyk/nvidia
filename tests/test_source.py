"""The scan, and the arithmetic that decides whether a warning can be counted at all.

A warning only exists from a window that *ends* before the predicted contact, so these tests
are not about tidiness: with a contiguous scan, a hazard that starts and finishes inside one
window earns zero lead time however early the read was.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vigil.perception.base import Clip
from vigil.video.source import VideoSource, load_recordings

from .helpers import CLIP_DIR, touch_video


def source(tmp_path: Path, *, duration: float = 10.0) -> VideoSource:
    path = touch_video(tmp_path, "cam.mp4")
    return VideoSource(video_id="cam", path=str(path), duration_s=duration, camera_label="CAM-01")


def test_the_file_has_to_exist_before_anything_else_does() -> None:
    with pytest.raises(ValueError, match="video file not found"):
        VideoSource(video_id="x", path="nope/missing.mp4", duration_s=5.0)


def test_a_window_is_clamped_to_the_recording_or_it_says_why(tmp_path: Path) -> None:
    video = source(tmp_path, duration=8.0)
    assert video.window(2.0, 99.0) == (2.0, 8.0)
    assert video.window(-3.0, 2.0) == (0.0, 2.0)
    with pytest.raises(ValueError, match="empty once clamped"):
        video.window(9.0, 12.0)
    clip = video.clip(1.0, 20.0)
    assert (clip.t0_s, clip.t1_s) == (1.0, 8.0)
    assert clip.camera_label == "CAM-01"
    assert clip.video_id == "cam"


def test_a_short_recording_is_one_window_whole(tmp_path: Path) -> None:
    video = source(tmp_path, duration=3.0)
    windows = video.scan(clip_seconds=6.0, max_clips=8, overlap=0.5)
    assert [(c.t0_s, c.t1_s) for c in windows] == [(0.0, 3.0)]


def _edges(windows: list[Clip]) -> list[tuple[float, float]]:
    return [(round(c.t0_s, 6), round(c.t1_s, 6)) for c in windows]


def test_a_contiguous_scan_tiles_the_recording_with_no_gaps_or_overlap(tmp_path: Path) -> None:
    video = source(tmp_path, duration=12.0)
    edges = _edges(video.scan(clip_seconds=6.0, max_clips=8, overlap=0.0))
    assert edges == [(0.0, 6.0), (6.0, 12.0)]


def test_overlap_doubles_the_windows_and_covers_the_same_ground(tmp_path: Path) -> None:
    video = source(tmp_path, duration=12.0)
    edges = _edges(video.scan(clip_seconds=6.0, max_clips=8, overlap=0.5))
    assert edges == [(0.0, 6.0), (3.0, 9.0), (6.0, 12.0)]
    # Every second is inside at least one window, and the last window still ends on the end.
    reach = edges[0][1]
    for start, end in edges[1:]:
        assert start <= reach
        reach = max(reach, end)
    assert reach == pytest.approx(12.0)


def best_warning(video: VideoSource, *, impact: float, **scan: object) -> float:
    """How far ahead of a known contact the scan could have raised the alarm."""
    windows = video.scan(**scan)  # type: ignore[arg-type]
    ends = [c.t1_s for c in windows if c.t1_s <= impact + 1e-9]
    return max((impact - end) for end in ends) if ends else 0.0


def test_an_overlapping_scan_buys_more_warning_for_the_same_hazard(tmp_path: Path) -> None:
    """Contact at 6.9 s of a 10 s recording.

    Contiguous 6 s windows end at 6.0 and 10.0, so only the first can warn anybody: 0.9 s. The
    second window already contains the impact, and a finding raised from it has no lead time to
    claim. Overlapping 4 s windows end at 4.0, 6.0, 8.0 and 10.0, and the earliest of those
    gives 2.9 s. Same footage, same hazard, same model: the scan is what changed.
    """
    video = source(tmp_path, duration=10.0)
    contiguous = best_warning(video, impact=6.9, clip_seconds=6.0, max_clips=8, overlap=0.0)
    overlapping = best_warning(video, impact=6.9, clip_seconds=4.0, max_clips=8, overlap=0.5)
    assert overlapping > contiguous


def test_more_clips_is_never_less_coverage(tmp_path: Path) -> None:
    video = source(tmp_path, duration=20.0)
    clamped = video.scan(clip_seconds=4.0, max_clips=3, overlap=0.5)
    assert len(clamped) == 3
    assert clamped[0].t0_s == 0.0
    assert clamped[-1].t1_s == 20.0
    # The clamp spends its budget on overlap before it spends it on resolution.
    span = clamped[0].t1_s - clamped[0].t0_s
    assert span == pytest.approx(4.0)


def test_a_pathological_overlap_is_capped_not_broken(tmp_path: Path) -> None:
    video = source(tmp_path, duration=10.0)
    windows = video.scan(clip_seconds=2.0, max_clips=6, overlap=5.0)
    assert windows
    assert windows[0].t1_s - windows[0].t0_s == pytest.approx(2.0)
    assert len(windows) <= 6


def test_from_record_reads_the_sidecar_and_fills_the_scene_words(tmp_path: Path) -> None:
    path = touch_video(tmp_path, "hazard.mp4")
    record = {
        "clip_id": "hazard",
        "path": "hazard.mp4",
        "duration_s": 11.0,
        "camera_label": "CAM-06 / AISLE-B",
        "site": "Northgate",
        "scene": {"area_type": "warehouse", "lighting": "typical_indoor", "surface": "dry_clear"},
    }
    video = VideoSource.from_record(record, tmp_path)
    assert video.video_id == "hazard"
    assert Path(video.path).name == "hazard.mp4"
    assert (video.area_type, video.lighting, video.surface) == (
        "warehouse",
        "typical_indoor",
        "dry_clear",
    )
    assert video.source_fps == 0.0
    assert path.is_file()


def test_a_sidecar_that_cannot_be_used_is_refused(tmp_path: Path) -> None:
    touch_video(tmp_path, "a.mp4")
    with pytest.raises(ValueError, match="duration_s must be a number"):
        VideoSource.from_record({"clip_id": "a", "path": "a.mp4", "duration_s": "long"}, tmp_path)
    with pytest.raises(ValueError, match="duration_s must be a number"):
        # True passes isinstance(x, int), which is exactly why bool is excluded first.
        VideoSource.from_record({"clip_id": "a", "path": "a.mp4", "duration_s": True}, tmp_path)
    with pytest.raises(ValueError, match="video file not found"):
        VideoSource.from_record({"clip_id": "a", "path": "gone.mp4", "duration_s": 4}, tmp_path)


def _index(tmp_path: Path, records: object) -> Path:
    (tmp_path / "index.json").write_text(json.dumps(records), encoding="utf-8")
    return tmp_path


def test_loading_a_directory_with_no_index_tells_you_the_command(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="synthesise the sample footage"):
        load_recordings(tmp_path)


def test_index_order_is_kept_because_the_dataset_is_ranked_hardest_first(tmp_path: Path) -> None:
    touch_video(tmp_path, "b.mp4")
    touch_video(tmp_path, "a.mp4")
    records = [
        {"clip_id": "b", "path": "b.mp4", "duration_s": 5.0},
        {"clip_id": "a", "path": "a.mp4", "duration_s": 5.0},
    ]
    assert [v.video_id for v in load_recordings(_index(tmp_path, records))] == ["b", "a"]


def test_every_unusable_entry_is_listed_in_one_error(tmp_path: Path) -> None:
    """One error naming everything wrong, not a crash on the first thing: the fix is usually
    a stale index after regenerating the clips, and guessing which entry is slow."""
    touch_video(tmp_path, "good.mp4")
    records = [
        {"clip_id": "good", "path": "good.mp4", "duration_s": 5.0},
        {"clip_id": "missing", "path": "missing.mp4", "duration_s": 5.0},
        "not even an object",
    ]
    with pytest.raises(ValueError, match="unusable entries") as excinfo:
        load_recordings(_index(tmp_path, records))
    message = str(excinfo.value)
    assert "missing: video file not found" in message
    assert "<str>" in message


def test_an_index_that_is_not_a_list_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="should hold a list"):
        load_recordings(_index(tmp_path, {"clip_id": "x"}))


def test_the_shipped_dataset_loads_in_the_authored_order() -> None:
    """Hardest first, so a truncated demo run still shows the interesting clips."""
    videos = load_recordings(CLIP_DIR)
    assert len(videos) == 8
    assert [v.video_id for v in videos][:2] == [
        "blind_corner_struck_by",
        "reversing_dock_no_spotter",
    ]
    assert [v.video_id for v in videos][-2:] == [
        "control_walkway_discipline",
        "control_housekeeping",
    ]
    assert all(v.duration_s > 0 for v in videos)
