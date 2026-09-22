"""The footage generator: the benchmark is only as honest as the clips it scores against.

These clips are the evidence base for every number the submission quotes, so two properties are
tested here rather than assumed. First, every authored scenario renders at every point in its
own timeline without raising -- a scenario that draws a black frame would still "pass" a demo.
Second, the ground truth beside each clip is derived from the same object that drew it, so a
label cannot drift away from the pixels it describes.

The encode test is the slow one (it runs ffmpeg) and is the only proof that a reviewer can
rebuild ``data/clips`` from a fresh clone.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from vigil.video.scenarios import all_scenarios, by_id
from vigil.video.synth import (
    FPS,
    HEIGHT,
    WIDTH,
    Camera,
    HazardLabel,
    Scenario,
    Track,
    encode,
    generate_all,
    ground_truth,
    render_frame,
)

from .helpers import CLIP_DIR, skip_without_ffmpeg


def _t(ref: str, kind: str, path: list[tuple[float, float, float]], **kw: object) -> Track:
    """A keyframed object. ``kind`` decides which drawing routine gets it."""
    return Track(ref=ref, kind=kind, path=path, **kw)  # type: ignore[arg-type]


def _everything_scenario() -> Scenario:
    """One frame containing every primitive the renderer knows how to draw."""
    return Scenario(
        id="kitchen_sink",
        title="Kitchen sink",
        duration_s=4.0,
        area_type="construction_site",
        lighting="backlit_or_glare",
        surface="wet",
        camera_label="CAM-99 / TEST",
        walkway_x=(-2.0, -1.0),
        machine_lane=False,
        tracks=[
            _t("P1", "person", [(0.0, -1.0, 3.0), (4.0, -1.0, 6.0)], role="vulnerable_party"),
            _t(
                "P2",
                "person",
                [(0.0, 1.0, 4.0)],
                role="worker_reach",
                category="operator",
                carries="load",
            ),
            _t(
                "P3",
                "person",
                [(0.0, 2.0, 5.0), (4.0, 2.6, 5.0)],
                lift=1.8,
                lift_window=(1.0, 3.0),
            ),
            _t(
                "FL1",
                "forklift",
                [(0.0, 0.0, 5.0), (4.0, 2.0, 5.0)],
                role="powered_machine",
                carries="rider",
            ),
            _t("V1", "truck", [(0.0, 3.0, 6.5), (4.0, 3.0, 4.5)], role="reverse"),
            _t("K1", "crate", [(0.0, -3.0, 3.0)]),
            _t("R1", "rack", [(0.0, 4.5, 7.0)]),
            _t("S1", "pillar", [(0.0, 0.9, 4.6)]),
            _t("C1", "cone", [(0.0, -2.0, 5.5), (4.0, -2.0, 4.0)]),
            _t("SP1", "spill", [(0.0, 1.5, 3.4)]),
            _t("L1", "ladder", [(0.0, 2.4, 4.2)]),
            _t("M1", "machine", [(0.0, -4.0, 6.4)]),
        ],
        hazards=[
            HazardLabel(
                type="struck_by",
                t_start=1.0,
                t_impact=3.2,
                severity=5,
                entities=["P1", "FL1"],
                description="Converging routes at the pillar.",
            )
        ],
    )


def _variance(img: object) -> float:
    arr = np.asarray(img, dtype=np.float32)
    return float(arr.std())


# ------------------------------------------------------------------------------------ the frames


@pytest.mark.parametrize("clip_id", [s.id for s in all_scenarios()])
def test_every_authored_clip_renders_across_its_whole_timeline(clip_id: str) -> None:
    scenario = by_id(clip_id)
    stamps = [0.0, scenario.duration_s * 0.37, scenario.duration_s * 0.75, scenario.duration_s]
    frames = [render_frame(scenario, t) for t in stamps]

    for frame, t in zip(frames, stamps, strict=True):
        assert frame.size == (WIDTH, HEIGHT), f"{clip_id} at {t}s"
        assert _variance(frame) > 6.0, f"{clip_id} at {t}s is a flat fill, not a scene"
    if scenario.hazards:
        assert _variance(frames[1]) != _variance(frames[2]), "nothing moved between the two reads"


def test_the_kitchen_sink_frame_draws_every_primitive_the_renderer_knows() -> None:
    scenario = _everything_scenario()
    frame = render_frame(scenario, 2.0)
    arr = np.asarray(frame.convert("L"), dtype=np.float32)
    assert frame.size == (WIDTH, HEIGHT)
    assert arr.std() > 6.0
    assert arr.min() < 60, "no dark geometry at all"
    assert arr.max() > 200, "the glare pass should leave highlights"


def test_rendering_is_deterministic_so_a_rebuild_reproduces_the_clip() -> None:
    scenario = _everything_scenario()
    a = np.asarray(render_frame(scenario, 1.5), dtype=np.uint8)
    b = np.asarray(render_frame(scenario, 1.5), dtype=np.uint8)
    assert np.array_equal(a, b), "grain must key off the scenario id, not process state"


def test_the_burned_in_stamp_moves_with_the_clock() -> None:
    """The camera label and time are drawn into the pixels, so an early and a late frame differ
    in the top band even where the scene itself has barely changed."""
    scenario = by_id("control_housekeeping")

    def band(t: float) -> np.ndarray:
        return np.asarray(render_frame(scenario, t).crop((0, 0, WIDTH, 30)), dtype=np.uint8)

    assert not np.array_equal(band(1.0), band(3.0))


# --------------------------------------------------------------------------------------- geometry


def test_the_camera_projects_a_known_point_onto_the_screen() -> None:
    cam = Camera()
    sx, sy, scale = cam.project(0.0, cam.h, 0.0)
    assert sx == pytest.approx(cam.cx)
    assert sy == pytest.approx(cam.cy + cam.fx * cam.h / cam.h)
    assert scale == pytest.approx(cam.fx / cam.h)
    near = cam.ground(0.0, 2.0)
    far = cam.ground(0.0, 8.0)
    assert near[2] > far[2], "closer must be bigger"
    assert near[1] > far[1], "closer ground sits lower in the frame"
    assert cam.ground(0.0, 0.0)[2] == cam.ground(0.0, 0.35)[2], "depth is clamped, not divided"


def test_a_track_interpolates_holds_and_extrapolates_only_a_little() -> None:
    moving = _t("P", "person", [(0.0, 0.0, 3.0), (2.0, 2.0, 3.0)])
    assert moving.state(1.0)[:2] == pytest.approx((1.0, 3.0))
    assert moving.speed_mps(1.0) == pytest.approx(1.0, abs=0.05)
    # Past the last keyframe it keeps the last direction for one segment, then stops.
    assert moving.state(3.0)[:2] == pytest.approx((3.0, 3.0))
    assert moving.state(9.0)[:2] == pytest.approx((4.0, 3.0))
    assert moving.state(-1.0)[:2] == pytest.approx((0.0, 3.0))

    parked = _t("Q", "person", [(0.0, 0.0, 3.0), (2.0, 2.0, 3.0)], hold_from=2.0)
    assert parked.state(7.0)[:2] == pytest.approx((2.0, 3.0))

    alone = _t("R", "crate", [(1.0, 5.0, 5.0)])
    assert alone.state(0.0)[:2] == pytest.approx((5.0, 5.0))
    assert alone.speed_mps(4.0) == pytest.approx(0.0)


def test_a_reversing_vehicle_faces_its_nose_not_its_motion() -> None:
    forward = _t("V", "truck", [(0.0, 0.0, 6.0), (2.0, 0.0, 4.0)])
    assert forward.state(1.0)[2] == pytest.approx(math.pi)
    authored = _t("V", "truck", [(0.0, 0.0, 6.0), (2.0, 0.0, 4.0)], facing_override=0.0)
    assert authored.state(1.0)[2] == pytest.approx(0.0)


def test_a_lift_ramps_over_its_window_and_is_flat_without_one() -> None:
    ramp = _t("W", "person", [(0.0, 0.0, 3.0)], lift=2.0, lift_window=(1.0, 3.0))
    assert ramp.lift_at(0.5) == 0.0
    assert ramp.lift_at(2.0) == pytest.approx(1.0)
    assert ramp.lift_at(5.0) == pytest.approx(2.0)
    assert _t("X", "person", [(0.0, 0.0, 3.0)], lift=2.0).lift_at(0.0) == pytest.approx(2.0)
    assert _t("Y", "person", [(0.0, 0.0, 3.0)]).lift_at(0.0) == 0.0


def test_keyframes_are_sorted_so_the_author_cannot_write_time_backwards() -> None:
    track = _t("P", "person", [(2.0, 1.0, 3.0), (0.0, 0.0, 3.0)])
    assert [p[0] for p in track.path] == [0.0, 2.0]
    assert track.category == "person", "category defaults to kind"
    with pytest.raises(ValueError, match="at least one keyframe"):
        _t("P", "person", [])


def test_a_hazard_label_states_the_window_anticipation_was_possible() -> None:
    label = HazardLabel(
        type="slip_trip",
        t_start=4.2,
        t_impact=8.6,
        severity=3,
        entities=["P1", "SP1"],
        description="Wet floor, no cone.",
    )
    assert label.anticipatable_s == pytest.approx(4.4)
    assert label.as_dict()["anticipatable_s"] == 4.4
    assert label.as_dict()["entities"] == ["P1", "SP1"]
    with pytest.raises(ValueError, match="impact must follow its start"):
        HazardLabel(
            type="slip_trip",
            t_start=5.0,
            t_impact=5.0,
            severity=3,
            entities=["P1"],
            description="too late to matter",
        )


def test_a_scenario_seeds_from_its_id_not_from_the_python_hash() -> None:
    scenario = _everything_scenario()
    assert scenario.seed == _everything_scenario().seed
    assert 0 <= scenario.seed < 65536
    assert scenario.has_hazard
    assert scenario.hazard_types == ["struck_by"]
    quiet = Scenario(id="quiet", title="Quiet", duration_s=1.0, tracks=[], hazards=[], notes="")
    assert not quiet.has_hazard
    assert quiet.seed != scenario.seed


# --------------------------------------------------------------------------------- ground truth


def test_the_ground_truth_record_describes_the_clip_that_was_rendered() -> None:
    scenario = _everything_scenario()
    record = ground_truth(scenario, Path("out/kitchen_sink.mp4"))
    assert record["clip_id"] == "kitchen_sink"
    assert record["path"] == "kitchen_sink.mp4", "sidecars are read relative to the clips"
    assert record["fps"] == FPS
    assert record["resolution"] == [WIDTH, HEIGHT]
    assert record["synthetic"] is True
    assert record["scene"] == {
        "area_type": "construction_site",
        "lighting": "backlit_or_glare",
        "surface": "wet",
    }
    assert record["hazards"] == [h.as_dict() for h in scenario.hazards]


def test_the_shipped_clips_match_the_scenarios_that_generated_them() -> None:
    """A stale ``data/clips`` directory is the quietest possible way to lie about a benchmark."""
    present = [s for s in all_scenarios() if (CLIP_DIR / f"{s.id}.groundtruth.json").exists()]
    if not present:
        pytest.skip("run `vigil clips` to generate the dataset")
    assert len(present) == len(all_scenarios()), "some shipped clips have no sidecar"
    for scenario in present:
        on_disk = json.loads(
            (CLIP_DIR / f"{scenario.id}.groundtruth.json").read_text(encoding="utf-8")
        )
        assert on_disk == ground_truth(scenario, CLIP_DIR / f"{scenario.id}.mp4"), scenario.id


# --------------------------------------------------------------------------------------- encode


def test_encoding_a_clip_writes_a_playable_file_and_a_reusable_sidecar(
    tmp_path: Path,
) -> None:
    skip_without_ffmpeg()
    scenario = _everything_scenario()
    mp4 = tmp_path / "clips" / "kitchen_sink.mp4"

    record = encode(scenario, mp4)
    assert mp4.exists(), record
    assert mp4.stat().st_size > 1000, "an mp4 of a moving bay is not a few hundred bytes"
    assert record["duration_s"] == scenario.duration_s

    from vigil.video.sampler import probe

    info = probe(str(mp4))
    assert (info.width, info.height) == (1024, 576), "the encode default is a committable size"
    assert info.duration_s == pytest.approx(scenario.duration_s, abs=0.35)
    assert info.fps == pytest.approx(FPS, abs=0.6)


def test_generate_all_skips_finished_clips_and_rebuilds_on_demand(tmp_path: Path) -> None:
    skip_without_ffmpeg()
    scenario = _everything_scenario()
    first = generate_all([scenario], tmp_path)
    assert (tmp_path / "index.json").exists()
    assert (tmp_path / "kitchen_sink.groundtruth.json").exists()
    assert (tmp_path / "kitchen_sink.mp4").exists()
    stamp = (tmp_path / "kitchen_sink.mp4").stat().st_mtime_ns

    assert generate_all([scenario], tmp_path) == first, "a re-run must not re-render"
    assert (tmp_path / "kitchen_sink.mp4").stat().st_mtime_ns == stamp

    generate_all([scenario], tmp_path, force=True)
    assert (tmp_path / "kitchen_sink.mp4").stat().st_mtime_ns != stamp
