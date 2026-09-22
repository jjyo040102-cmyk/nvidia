"""The scripted vision backend, and the guarantee it is honest about.

``mock`` derives a scene from the authored scenario, so for these eight clips it is an oracle
rather than a model. That makes it excellent for wiring and terrible as evidence of accuracy,
which is why it refuses unknown clips loudly and why every report says which backend ran.
"""

from __future__ import annotations

import pytest

from vigil.config import Settings
from vigil.models.risk import HazardClass
from vigil.models.scene import Kinematics
from vigil.perception.base import Clip, PerceptionError
from vigil.perception.mock import (
    MockPerception,
    answer_from_scenario,
    find_scenario,
    scene_from_scenario,
)
from vigil.video.scenarios import all_scenarios
from vigil.video.source import VideoSource


def clip(video: VideoSource, t0: float, t1: float) -> Clip:
    return video.clip(t0, t1)


async def test_an_authored_clip_reads_without_a_key(
    authored_videos: list[VideoSource], tmp_settings: Settings
) -> None:
    backend = MockPerception(tmp_settings)
    assert backend.name == "mock"
    assert backend.supports_video is False
    scene, usage = await backend.survey(clip(authored_videos[0], 0.0, 4.0))
    assert usage.model == "mock"
    assert usage.total_tokens == 0
    assert scene.entities
    assert scene.confidence > 0.5
    await backend.aclose()


async def test_a_clip_that_was_not_authored_is_refused_with_the_fix(tmp_settings: Settings) -> None:
    backend = MockPerception(tmp_settings)
    stranger = Clip(video_id="holiday_snippet", path="holiday.mp4", t0_s=0.0, t1_s=4.0)
    with pytest.raises(PerceptionError, match="no fixture"):
        await backend.survey(stranger)
    with pytest.raises(PerceptionError, match="no fixture"):
        await backend.ask(stranger, "Does anyone cross the lane?")


def test_a_scenario_is_found_by_id_or_by_the_filename() -> None:
    """A recording that was re-encoded under its own filename still matches its fixture."""
    by_id = Clip(video_id="ride_on_forks", path="whatever.mp4", t0_s=0.0, t1_s=4.0)
    assert find_scenario(by_id) is not None
    by_stem = Clip(
        video_id="generic-import", path="clips/night_no_hivis_pedestrian.mp4", t0_s=0.0, t1_s=4.0
    )
    assert find_scenario(by_stem) is not None
    assert find_scenario(Clip(video_id="zzz", path="zzz.mp4", t0_s=0.0, t1_s=1.0)) is None


def test_every_authored_scenario_produces_a_valid_scene_at_every_window() -> None:
    """A fixture that fails to validate is a broken dataset, not a failed read.

    Windows straddle hazard starts and impacts in every combination, which is where the
    entity/conflict consistency check bites.
    """
    from vigil.video.synth import Scenario

    scenarios: list[Scenario] = list(all_scenarios())
    assert len(scenarios) == 8
    for scenario in scenarios:
        for t0 in (0.0, 2.5, 5.0, 9.5):
            t1 = min(scenario.duration_s, t0 + 4.0)
            if t1 <= t0:
                continue
            subject = Clip(video_id=scenario.id, path=f"{scenario.id}.mp4", t0_s=t0, t1_s=t1)
            scene = scene_from_scenario(scenario, subject)
            assert scene.t0_s == t0
            assert scene.t1_s == t1
            assert {c.kind for c in scene.conflicts} <= set(HazardClass._value2member_map_)
            assert scene.entities, f"{scenario.id} at {t0}-{t1}s produced nothing to look at"


def test_time_to_contact_is_measured_from_the_end_of_the_window(
    authored_videos: list[VideoSource],
) -> None:
    """The definition every lead-time number in the product rests on."""
    video = next(v for v in authored_videos if v.video_id == "blind_corner_struck_by")
    scenario = find_scenario(clip(video, 0.0, 4.0))
    assert scenario is not None
    impact = scenario.hazards[0].t_impact

    early = scene_from_scenario(scenario, clip(video, 0.0, 4.0))
    assert early.conflicts[0].time_to_event_s == pytest.approx(impact - 4.0, abs=0.01)

    during = scene_from_scenario(scenario, clip(video, 4.0, 8.0))
    assert during.conflicts[0].time_to_event_s == 0.0  # contact is inside this window

    after = scene_from_scenario(scenario, clip(video, 8.0, 10.0))
    assert after.conflicts == []  # the window starts at the impact: nothing left to anticipate


def test_props_that_are_not_movers_still_appear_as_entities(
    authored_videos: list[VideoSource],
) -> None:
    """A spill is not a mover, but "P1 trips onto SP1" needs SP1 declared to validate at all."""
    video = next(v for v in authored_videos if v.video_id == "unmarked_spill_slip")
    subject = video.clip(2.0, 6.0)
    scenario = find_scenario(subject)
    assert scenario is not None
    scene = scene_from_scenario(scenario, subject)
    assert "SP1" in {e.ref for e in scene.entities}
    assert scene.conflicts
    assert "SP1" in scene.conflicts[0].participants
    assert scene.entities_by_ref["SP1"].kinematics is Kinematics.STATIONARY


def test_hazard_entities_are_what_a_follow_up_question_matches_on(
    authored_videos: list[VideoSource],
) -> None:
    video = next(v for v in authored_videos if v.video_id == "blind_corner_struck_by")
    scenario = find_scenario(video.clip(0.0, 4.0))
    assert scenario is not None
    answer = answer_from_scenario(
        scenario,
        video.clip(4.0, 8.0),
        "Do the routes of P1 and FL1 reach the same point while both are there?",
    )
    assert answer.hazard_present is True
    assert answer.confidence > 0.8
    assert answer.usage.model == "mock"
    assert set(answer.entities_mentioned) >= {"P1", "FL1"}


async def test_an_unrelated_question_is_answered_as_clear_not_as_an_error(
    authored_videos: list[VideoSource], tmp_settings: Settings
) -> None:
    backend = MockPerception(tmp_settings)
    video = next(v for v in authored_videos if v.video_id.startswith("control_"))
    answer = await backend.ask(
        video.clip(0.0, 4.0), "Is the conveyer belt guard in place and interlocked?"
    )
    assert answer.hazard_present is False
    assert answer.verdict == "CLEAR"
    assert answer.observed_at_s is None


def test_control_clips_offer_nothing_to_flag(authored_videos: list[VideoSource]) -> None:
    """The two silent clips are the false-positive half of the benchmark.

    They move people and machines through the same frame with no shared point, so a detector
    that keys on "forklift plus pedestrian" rather than on a converging trajectory reports them.
    """
    for video in authored_videos:
        if not video.video_id.startswith("control_"):
            continue
        scenario = find_scenario(video.clip(0.0, 1.0))
        assert scenario is not None
        assert not scenario.has_hazard
        for start in (0.0, 3.0, 6.0):
            subject = video.clip(start, start + 4.0)
            scene = scene_from_scenario(scenario, subject)
            assert scene.conflicts == [], f"{video.video_id} at {start}s is not a control"
            assert scene.entities, f"{video.video_id} at {start}s has nothing in frame"
