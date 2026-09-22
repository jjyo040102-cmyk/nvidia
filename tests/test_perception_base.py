"""Parsing what a vision model actually sends back.

The model wraps JSON in prose, invents entity handles in its conflict list and occasionally
names a lighting condition that is not in the vocabulary. Dropping the read is the wrong call
when it saw the danger, and raising is the wrong call when the fix is obvious -- so the rule
here is: repair what is unambiguous, record every guess, refuse the rest.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from vigil.models.scene import AreaType, Entity, Kinematics, Lighting, Role
from vigil.perception.base import (
    Clip,
    PerceptionAnswer,
    PerceptionError,
    Usage,
    answer_from_payload,
    as_flag,
    clamp_unit,
    extract_json,
    make_entity,
    repair_scene,
    scene_json_schema,
)

SCENE = {
    "clip_id": "c1@0.0-4.0",
    "t0_s": 0.0,
    "t1_s": 4.0,
    "area_type": "warehouse",
    "lighting": "typical_indoor",
    "surface": "dry_clear",
    "entities": [
        {"ref": "P1", "category": "worker", "role": "vulnerable_party", "location": "left"},
        {"ref": "FL1", "category": "forklift", "role": "powered_machine", "location": "right"},
    ],
    "conflicts": [
        {
            "participants": ["P1", "FL1"],
            "kind": "struck_by",
            "severity_hint": 4,
            "rationale": "Both reach the aisle mouth within 1.5 s.",
        }
    ],
}


def test_a_clip_is_a_window_you_can_trust() -> None:
    clip = Clip(video_id="v", path="v.mp4", t0_s=1.5, t1_s=4.0, camera_label="CAM-2")
    assert clip.duration_s == 2.5
    assert clip.id == "v@1.5-4.0"
    with pytest.raises(ValidationError, match="must ascend"):
        Clip(video_id="v", path="v.mp4", t0_s=4.0, t1_s=4.0)


def test_usage_and_verdict_are_the_cheap_accessors() -> None:
    assert Usage(prompt_tokens=10, completion_tokens=4).total_tokens == 14
    answer = PerceptionAnswer(question="q", answer="a", hazard_present=True, confidence=0.9)
    assert answer.verdict == "HAZARD"
    assert answer.usage.model == "mock"


# --------------------------------------------------------------------- json recovery


def test_a_plain_object_parses() -> None:
    assert extract_json('{"a": 1}') == {"a": 1}


def test_a_fenced_object_parses_with_or_without_the_language_tag() -> None:
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('```\n{"a": 1}\n```') == {"a": 1}


def test_prose_around_the_object_is_ignored_because_models_always_add_prose() -> None:
    raw = 'Here is the scene you asked for:\n{"a": {"b": 2}}\nLet me know if you want more.'
    assert extract_json(raw) == {"a": {"b": 2}}


@pytest.mark.parametrize("raw", ["", "   ", "no json at all", "[1, 2, 3]", '"a string"'])
def test_replies_that_carry_no_object_are_a_failure_not_a_silence(raw: str) -> None:
    with pytest.raises(PerceptionError):
        extract_json(raw)


def test_a_truncated_reply_is_named_as_having_no_object() -> None:
    with pytest.raises(PerceptionError, match="no JSON object"):
        extract_json('{"a": 1, "b": ')


def test_an_object_that_is_there_but_broken_is_named_as_malformed() -> None:
    with pytest.raises(PerceptionError, match="malformed JSON"):
        extract_json('{"a": 1,, "b": 2}')


# --------------------------------------------------------------------- bounded repair


def test_a_good_payload_needs_no_repair() -> None:
    scene, notes = repair_scene(dict(SCENE))
    assert notes == []
    assert scene.area_type is AreaType.WAREHOUSE
    assert scene.conflicts[0].participants == ["FL1", "P1"]


def test_a_handle_with_a_space_in_it_matches_across_both_sides_of_the_schema() -> None:
    """Entities and conflicts must normalise identically or the scene refuses itself."""
    payload = json.loads(json.dumps(SCENE))
    payload["entities"].append(
        {"ref": "Pedestrian A", "category": "worker", "role": "bystander", "location": "far left"}
    )
    payload["conflicts"][0]["participants"] = ["P1", "pedestrian a"]
    scene, _notes = repair_scene(payload)
    assert scene.entity("Pedestrian A") is not None
    assert scene.conflicts[0].participants == ["P1", "PEDESTRIANA"]


def test_a_ghost_entity_in_a_conflict_is_declared_and_said_so() -> None:
    """The model saw two things converge and only named one of them. Losing that conflict is
    the expensive failure, so the missing entity is created and the guess is recorded."""
    payload = json.loads(json.dumps(SCENE))
    payload["conflicts"][0]["participants"] = ["P1", "PEDESTRIAN A"]
    scene, notes = repair_scene(payload)
    assert scene.entity("PEDESTRIANA") is not None
    assert any("auto-declared" in note for note in notes)
    assert scene.repair_notes == notes


def test_fields_that_are_not_the_wrong_type_are_defaulted_and_recorded() -> None:
    payload = {"entities": "none", "conflicts": None}
    scene, notes = repair_scene(payload)
    assert scene.entities == []
    assert scene.conflicts == []
    assert scene.clip_id == "unknown"
    assert scene.t0_s == 0.0
    assert scene.t1_s == 0.0
    assert any("entities was not a list" in note for note in notes)
    assert any("t0_s missing" in note for note in notes)
    assert any("t1_s missing" in note for note in notes)


def test_a_conflict_of_prose_is_dropped_not_validated_characterwise() -> None:
    """The read survives: one unusable conflict is not a licence to lose the whole frame."""
    payload = json.loads(json.dumps(SCENE))
    payload["conflicts"][0]["participants"] = "P1 and FL1"
    payload["conflicts"].append("not an object either")
    scene, notes = repair_scene(payload)
    assert scene.conflicts == []
    assert any("participants was str" in note for note in notes)
    assert any("not an object" in note for note in notes)


def test_a_window_running_backwards_cannot_be_repaired_into_existence() -> None:
    payload = json.loads(json.dumps(SCENE))
    payload["t0_s"], payload["t1_s"] = 9.0, 4.0
    with pytest.raises(PerceptionError, match="precedes"):
        repair_scene(payload)


def test_the_validation_message_points_at_the_field_that_broke() -> None:
    payload = json.loads(json.dumps(SCENE))
    payload["entities"] = "not a list"
    payload["conflicts"][0]["severity_hint"] = 99
    with pytest.raises(PerceptionError) as excinfo:
        repair_scene(payload)
    assert "severity_hint" in str(excinfo.value)


def test_the_schema_handed_to_the_model_defines_the_things_it_asks_for() -> None:
    schema = json.loads(scene_json_schema())
    assert {"entities", "conflicts", "visibility_limits", "confidence"} <= set(schema["properties"])
    # Every $ref has to resolve somewhere in the same text, or the schema specifies nothing.
    text = scene_json_schema()
    assert "#/$defs/Entity" in text
    assert '"Entity"' in text
    assert {"ref", "category", "role", "location"} <= set(schema["$defs"]["Entity"]["properties"])
    assert schema["$defs"]["Conflict"]["properties"]["participants"]["minItems"] == 2
    # The injury mechanisms are the vocabulary the model must answer in.
    assert "struck_by_reversing_vehicle" in text


def test_make_entity_fills_the_field_the_schema_requires() -> None:
    entity = make_entity("p1", "worker", role=Role.OPERATOR)
    assert entity.ref == "P1"
    assert entity.role is Role.OPERATOR
    assert entity.location == "not specified"
    assert Entity(ref="X", category="crate", location="").location == ""


def test_the_scene_enums_are_the_ones_the_prompts_promise() -> None:
    scene = repair_scene(dict(SCENE))[0]
    assert scene.lighting is Lighting.TYPICAL_INDOOR
    assert Kinematics.REVERSING.value == "reversing"


# --------------------------------------------------------------------- the answer shape

WINDOW = Clip(video_id="v", path="v.mp4", t0_s=4.0, t1_s=7.0)


def test_an_answer_names_the_moment_it_saw_inside_the_window_it_was_shown() -> None:
    answer = answer_from_payload(
        WINDOW,
        "Does the worker step into the aisle?",
        {"answer": " Yes, at the mouth. ", "hazard_present": True, "observed_at_s": 5.5},
        usage=Usage(prompt_tokens=100, completion_tokens=8, model="cosmos/x"),
    )
    assert answer.answer == "Yes, at the mouth."
    assert answer.verdict == "HAZARD"
    assert answer.observed_at_s == 5.5
    assert answer.usage.model == "cosmos/x"


@pytest.mark.parametrize(("observed", "expected"), [(1.0, 4.0), (40.0, 7.0), ("abc", None)])
def test_a_time_outside_the_window_is_pulled_onto_its_edge(
    observed: object, expected: float | None
) -> None:
    """A model that answers with a moment it was not shown is wrong about *when*.

    Clamping keeps the sighting inside the footage the report cites instead of letting a
    hallucinated timestamp become evidence about a time nobody looked at.
    """
    answer = answer_from_payload(
        WINDOW, "q", {"answer": "a", "observed_at_s": observed}, usage=Usage()
    )
    assert answer.observed_at_s == expected


def test_a_blank_or_absent_answer_still_says_what_happened() -> None:
    for payload in ({"answer": "   "}, {}, {"answer": ""}):
        answer = answer_from_payload(WINDOW, "q", payload, usage=Usage())
        assert answer.answer == "no answer returned"
        assert not answer.hazard_present


@pytest.mark.parametrize(
    ("value", "flag"),
    [
        (True, True),
        ("true", True),
        ("yes", True),
        ("False", False),
        ("no", False),
        ("none", False),
        ("", False),
        (0, False),
        (None, False),
        ("1", True),
    ],
)
def test_a_boolean_written_any_way_a_language_model_writes_one_comes_out_the_right_way(
    value: object, flag: bool
) -> None:
    """``bool("false")`` is True in Python, and that particular line turns a clear scene into a
    stop-work alarm on a system whose recommendation has to be trustworthy.
    """
    assert as_flag(value) is flag
    answer = answer_from_payload(WINDOW, "q", {"hazard_present": value}, usage=Usage())
    assert answer.hazard_present is flag


@pytest.mark.parametrize(("value", "expected"), [(0.5, 0.5), ("0.8", 0.8), (2, 1.0), (-1, 0.0)])
def test_confidence_is_clamped_onto_a_scale_that_means_something(
    value: object, expected: float
) -> None:
    assert clamp_unit(value, 0.4) == expected


@pytest.mark.parametrize("value", [None, "abc", {}, []])
def test_a_confidence_that_is_not_a_number_becomes_the_stated_default(value: object) -> None:
    assert clamp_unit(value, 0.4) == 0.4


def test_entities_are_kept_short_clean_and_in_the_order_the_model_gave_them() -> None:
    answer = answer_from_payload(
        WINDOW,
        "q",
        {"answer": "a", "entities_mentioned": ["P1", "  ", "FL1", *[f"E{i}" for i in range(20)]]},
        usage=Usage(),
    )
    assert answer.entities_mentioned[:2] == ["P1", "FL1"]
    assert len(answer.entities_mentioned) == 8, "a runaway list is not going into the transcript"
