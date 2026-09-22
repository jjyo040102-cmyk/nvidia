"""The gate, not the prompt.

Two properties are enforced here rather than requested of the model: a finding can only cite a
rule that was actually retrieved in this run, and a window that has already been read is not
read again. Both are things a language model will do given the chance, and both would produce a
report that looks authoritative and is wrong.
"""

from __future__ import annotations

import pytest

from vigil.agent.actions import (
    ClearAction,
    FinishAction,
    FlagAction,
    PolicyAction,
    RelookAction,
    SurveyAction,
)
from vigil.agent.tools import ToolKit, describe_scene, scene_payload
from vigil.config import Settings
from vigil.models.risk import Likelihood
from vigil.models.scene import SceneUnderstanding
from vigil.perception.base import Clip, PerceptionAnswer, PerceptionError, Usage
from vigil.perception.mock import MockPerception, find_scenario, scene_from_scenario
from vigil.policy.kb import Rulebook
from vigil.video.source import VideoSource


class FlakyPerception(MockPerception):
    """Raises the way a real backend does when the network is unhappy."""

    async def survey(self, clip: Clip) -> tuple[SceneUnderstanding, Usage]:
        raise PerceptionError("vision model unavailable (simulated)")

    async def ask(self, clip: Clip, question: str) -> PerceptionAnswer:
        raise PerceptionError("vision model unavailable (simulated)")


@pytest.fixture
def kit(blind_corner: VideoSource, rulebook: Rulebook, tmp_settings: Settings) -> ToolKit:
    return ToolKit(
        perception=MockPerception(tmp_settings),
        rulebook=rulebook,
        video=blind_corner,
        settings=tmp_settings,
    )


def flag(
    hazard_id: str = "H1", rule_ids: list[str] | None = None, **overrides: object
) -> FlagAction:
    base: dict[str, object] = {
        "hazard_id": hazard_id,
        "title": "Struck-by between converging routes: FL1 and P1",
        "severity": 4,
        "likelihood": Likelihood.LIKELY,
        "predicted_event": "The truck reaches the crossing point as the worker steps into it.",
        "lead_time_s": 2.9,
        "entities_involved": ["P1", "FL1"],
        "rule_ids": rule_ids or [],
    }
    base.update(overrides)
    return FlagAction(**base)


async def test_a_citation_the_agent_was_never_shown_is_dropped_and_said_so(kit: ToolKit) -> None:
    """The single most damaging failure this product can have."""
    await kit._survey(0.0, 4.0)
    obs = await kit.execute(flag(rule_ids=["osha-1910-178-pedestrians", "not-a-rule-at-all"]))
    assert obs.assessment is not None
    assert obs.assessment.violated_rules == []
    assert "never retrieved this run" in obs.body
    assert "not-a-rule-at-all" in obs.body


async def test_a_retrieved_rule_is_attached_and_its_floor_is_applied(kit: ToolKit) -> None:
    await kit._survey(0.0, 4.0)
    found = await kit.execute(
        PolicyAction(query="pedestrian in the path of a powered truck", hazard_type="struck_by")
    )
    best = found.rules[0].rule.id
    assert best in kit.surfaced

    obs = await kit.execute(flag(rule_ids=[best]))
    assessment = obs.assessment
    assert assessment is not None
    assert [c.rule_id for c in assessment.violated_rules] == [best]
    assert assessment.violated_rules[0].relevance == kit.surfaced[best]
    floor = kit.rulebook.get(best)
    assert floor is not None
    assert assessment.severity == max(4, floor.severity_floor)
    if floor.severity_floor > 4:
        assert f"Severity raised 4 -> {floor.severity_floor}" in obs.body


async def test_a_flag_cannot_be_backed_by_a_rule_from_another_run(
    kit: ToolKit, tmp_settings: Settings, blind_corner: VideoSource, rulebook: Rulebook
) -> None:
    """Surfaced is per-investigation. A rule the previous video surfaced is not evidence here."""
    elsewhere = ToolKit(
        perception=MockPerception(tmp_settings),
        rulebook=rulebook,
        video=blind_corner,
        settings=tmp_settings,
    )
    await elsewhere.execute(PolicyAction(query="forklift travel speed", hazard_type="struck_by"))
    assert elsewhere.surfaced is not kit.surfaced
    ids = list(elsewhere.surfaced)
    assert ids
    obs = await kit.execute(flag(rule_ids=ids[:1]))
    assert obs.assessment is not None
    assert obs.assessment.violated_rules == []


async def test_reading_the_same_window_twice_costs_nothing_and_says_so(kit: ToolKit) -> None:
    first = await kit._survey(0.0, 4.0)
    second = await kit._survey(0.0, 4.0)
    assert first.usage is not None
    assert second.usage is None
    assert second.scene == first.scene
    assert "no second call was made" in second.body
    assert second.summary.startswith("Re-read")


async def test_a_relook_is_memoised_on_the_question_not_just_the_window(kit: ToolKit) -> None:
    question = "Do the routes of P1 and FL1 reach the same point while both are there?"
    first = await kit.execute(RelookAction(t0_s=4.0, t1_s=8.0, question=question))
    again = await kit.execute(RelookAction(t0_s=4.0, t1_s=8.0, question=question))
    other = await kit.execute(
        RelookAction(
            t0_s=4.0, t1_s=8.0, question="Is the worker wearing high visibility clothing here?"
        )
    )
    assert first.usage is not None
    assert first.usage.total_tokens >= 0
    assert again.usage is None
    assert "already asked" in again.body
    assert other.usage is not None
    assert "hazard_present=" in first.body


async def test_a_window_outside_the_recording_is_reported_not_raised(kit: ToolKit) -> None:
    obs = await kit._survey(90.0, 94.0)
    assert obs.ok is False
    assert "empty once clamped" in obs.body
    relook = await kit.execute(
        RelookAction(t0_s=90.0, t1_s=94.0, question="Does contact happen in here?")
    )
    assert relook.ok is False


async def test_a_perception_failure_becomes_an_observation_and_is_recorded(
    blind_corner: VideoSource, rulebook: Rulebook, tmp_settings: Settings
) -> None:
    broken = ToolKit(
        perception=FlakyPerception(tmp_settings),
        rulebook=rulebook,
        video=blind_corner,
        settings=tmp_settings,
    )
    survey = await broken._survey(0.0, 4.0)
    assert survey.ok is False
    assert "Perception failed" in survey.summary
    relook = await broken.execute(RelookAction(t0_s=0.0, t1_s=4.0, question="Who is in the lane?"))
    assert relook.ok is False
    assert len(broken.errors) == 2
    assert broken.errors[0].startswith("survey:")
    assert broken.scenes == []


async def test_a_policy_query_that_matches_nothing_tells_the_agent_what_to_try(
    kit: ToolKit,
) -> None:
    obs = await kit.execute(PolicyAction(query="qqqq zzzz xxxx"))
    assert obs.rules == ()
    assert "Nothing in the rulebook" in obs.body


async def test_a_clear_is_recorded_as_an_act_of_not_silence(kit: ToolKit) -> None:
    obs = await kit.execute(
        ClearAction(
            subject="Gate interlock on P1", reason="The guard is closed and the bolt is thrown."
        )
    )
    assert obs.tool == "clear"
    assert obs.summary.startswith("Ruled out")


async def test_the_loop_executes_finish_not_the_toolkit(kit: ToolKit) -> None:
    with pytest.raises(TypeError, match="executed by the loop"):
        await kit.execute(FinishAction(summary="Nothing further to establish."))


def test_describe_scene_is_the_transcript_form_and_carrys_the_numbers() -> None:
    scene = _scene()
    text = describe_scene(scene)
    assert "Window 0.0-4.0s" in text
    assert "CONFLICT FL1+P1: struck_by" in text
    assert "s to contact" in text
    assert "missing hi_vis" in text
    assert "NOT VISIBLE" in text
    assert "NARRATIVE" in text
    assert "confidence 0.85" in text


def test_the_digest_keeps_the_parser_notes_visible() -> None:
    from vigil.perception.base import repair_scene

    payload = scene_payload(_scene())
    payload["conflicts"][0]["participants"] = ["P1", "GHOST"]
    repaired, _notes = repair_scene(payload)
    assert "PARSER NOTE" in describe_scene(repaired)


def test_scene_payload_is_json_safe_and_complete() -> None:
    payload = scene_payload(_scene())
    assert payload["clip_id"].endswith("@0.0-4.0")
    assert {"P1", "FL1"} <= {e["ref"] for e in payload["entities"]}
    assert payload["conflicts"][0]["kind"] == "struck_by"


async def test_toolkit_exposes_the_scenes_it_read_in_insertion_order(kit: ToolKit) -> None:
    """The transcript and the report both iterate this, so the order is the story."""
    await kit.execute(SurveyAction(t0_s=4.0, t1_s=8.0))
    await kit.execute(SurveyAction(t0_s=0.0, t1_s=4.0))
    assert [scene.t0_s for scene in kit.scenes] == [4.0, 0.0]


# --------------------------------------------------------------------- helpers


def _scene() -> SceneUnderstanding:
    """The authored blind-corner read, without going through a video file."""
    subject = Clip(
        video_id="blind_corner_struck_by", path="blind_corner_struck_by.mp4", t0_s=0.0, t1_s=4.0
    )
    scenario = find_scenario(subject)
    assert scenario is not None
    return scene_from_scenario(scenario, subject)
