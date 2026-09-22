"""The planner that thinks in actions, tested against an endpoint that is not real.

``ScriptedReasoner`` gets exercised by every end-to-end run, so the parts of this file that
matter are the ones a scripted run never touches: the transcript actually handed to Nemotron,
the one bounded repair turn, and the error paths that decide whether a bad reply costs the run
or costs one retry. The rest is the two-clocks logic in :func:`_likelihood` -- the most
argued-about twenty lines in the repository, and the first thing a safety reviewer will probe.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest

from vigil.agent.actions import FinishAction, FlagAction, PolicyAction, RelookAction, Turn
from vigil.agent.prompts import AGENT_SYSTEM, ASKING
from vigil.agent.reasoner import (
    Briefing,
    NemotronReasoner,
    Reasoner,
    ReasonerError,
    ScriptedReasoner,
    _Lead,
    _likelihood,
    _parse_turn,
    build_reasoner,
)
from vigil.agent.tools import Observation
from vigil.config import Settings
from vigil.models.risk import HazardClass, Likelihood
from vigil.models.scene import Conflict, Entity, KeyMoment, SceneUnderstanding
from vigil.perception.base import PerceptionAnswer, Usage
from vigil.policy.kb import Rule, Scored
from vigil.video.source import VideoSource

from .helpers import CATALOG, client_with, completion, make_assessment, touch_video

REASONING_MODEL = "nvidia/nemotron-3-super-120b-a12b"

FINISH_REPLY = json.dumps(
    {
        "thought": "Nothing further changes the recommendation.",
        "action": {"tool": "finish", "summary": "Two windows read, one finding filed."},
    }
)
RELOOK_REPLY = json.dumps(
    {
        "thought": "The timing is what decides the severity, so settle it.",
        "action": {
            "tool": "relook",
            "t0_s": 5.2,
            "t1_s": 7.4,
            "question": "Do their routes reach the same point at the same time?",
        },
    }
)
NO_JSON_REPLY = "I will look at the footage now and tell you what I see."
UNKNOWN_TOOL_REPLY = json.dumps({"thought": "Sending it.", "action": {"tool": "teleport"}})


def catalog_only(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=CATALOG)


def refusing(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(404, text="no such model")


def wire(
    settings_for: Callable[..., Settings],
    replies: list[str],
    calls: list[str],
    **overrides: object,
) -> tuple[NemotronReasoner, list[dict[str, Any]]]:
    """A planner wired to a fake account: every request body recorded, every reply canned."""
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return catalog_only(request)
        bodies.append(json.loads(request.content.decode()))
        return completion(
            replies[min(len(bodies) - 1, len(replies) - 1)],
            model=REASONING_MODEL,
            prompt_tokens=500,
            completion_tokens=60,
        )

    settings = settings_for(nebius_api_key="k", reasoning_model=REASONING_MODEL, **overrides)
    return NemotronReasoner(settings, client_with(settings, handler, calls)), bodies


@pytest.fixture
def calls() -> list[str]:
    return []


@pytest.fixture
def briefing(blind_corner: VideoSource) -> Briefing:
    return Briefing(video=blind_corner, turn=3, max_turns=8, tokens_used=4200, token_budget=60_000)


# ----------------------------------------------------------------------- the transcript sent


def test_the_briefing_shows_the_planner_the_evidence_in_the_order_it_arrived(
    tmp_path: Path,
) -> None:
    """The planner sees a rendering, not a database. Whatever is not in here does not exist."""
    video = VideoSource(
        video_id="bay-7",
        path=str(touch_video(tmp_path, "bay7.mp4")),
        camera_label="CAM-07",
        duration_s=18.5,
        site="Northgate",
        area_type="vehicle_yard",
        lighting="backlit_or_glare",
        surface="wet",
    )
    text = Briefing(
        video=video,
        turn=2,
        max_turns=6,
        tokens_used=900,
        token_budget=1000,
        log=(
            Observation(tool="survey", summary="s", body="first read"),
            Observation(tool="relook", summary="q", body="second read"),
        ),
        findings=(make_assessment(),),
        cleared=("H0",),
    ).render()

    assert "Video bay-7 -- camera CAM-07, 18.5 s of footage" in text
    assert "Site: Northgate. Area: vehicle_yard." in text
    assert "Lighting: backlit_or_glare. Surface: wet." in text
    assert "Observation 1 from survey:\nfirst read" in text
    assert "Observation 2 from relook:\nsecond read" in text
    assert "Turn 2 of 6. Tokens used: 900 of 1000." in text
    assert "Findings recorded: 1. Cleared: 1." in text
    assert text.rstrip().endswith(ASKING)


def test_a_video_with_no_site_metadata_says_not_stated_rather_than_blanking_the_line(
    tmp_path: Path,
) -> None:
    video = VideoSource(
        video_id="bare", path=str(touch_video(tmp_path, "bare.mp4")), duration_s=4.0
    )
    text = Briefing(video=video, turn=1, max_turns=8, tokens_used=0, token_budget=60_000).render()
    assert "Site: not stated. Area: not stated. Lighting: not stated. Surface: not stated." in text
    assert "Observation" not in text
    assert "0 window(s) read" in text


async def test_the_planner_is_handed_the_system_grammar_and_the_rendered_transcript(
    settings_for: Callable[..., Settings], calls: list[str], briefing: Briefing
) -> None:
    reasoner, bodies = wire(settings_for, [FINISH_REPLY], calls)
    turn = await reasoner.plan(briefing)

    assert isinstance(turn.action, FinishAction)
    assert turn.thought == "Nothing further changes the recommendation."
    messages = bodies[0]["messages"]
    assert messages[0] == {"role": "system", "content": AGENT_SYSTEM}
    assert messages[1] == {"role": "user", "content": briefing.render()}
    assert bodies[0]["model"] == REASONING_MODEL
    assert bodies[0]["response_format"] == {"type": "json_object"}


async def test_every_turn_costs_exactly_one_call_and_is_billed_for(
    settings_for: Callable[..., Settings], calls: list[str], briefing: Briefing
) -> None:
    reasoner, bodies = wire(settings_for, [RELOOK_REPLY, FINISH_REPLY], calls)
    first = await reasoner.plan(briefing)
    # A turn that advanced is a transcript that changed. Re-sending the same bytes would be
    # served from the cache, and then this test would prove nothing about the per-turn cost.
    second = await reasoner.plan(replace(briefing, turn=briefing.turn + 1))

    assert isinstance(first.action, RelookAction)
    assert first.action.question.startswith("Do their routes")
    assert isinstance(second.action, FinishAction)
    assert len(bodies) == 2
    assert calls.count("POST /v1/chat/completions") == 2
    # Accumulated rather than reset: the ceiling is enforced against this number.
    assert (reasoner.usage.prompt_tokens, reasoner.usage.completion_tokens) == (1000, 120)
    assert reasoner.usage.model == REASONING_MODEL


async def test_a_briefing_that_did_not_change_does_not_bill_the_credit_twice(
    settings_for: Callable[..., Settings], calls: list[str], briefing: Briefing
) -> None:
    """The whole point of the disk cache: a re-run of the same decision is free.

    Worth pinning here rather than only at the client, because this is where a bug would be
    invisible -- a planner that re-asked an identical question would still work, and would
    quietly spend the trial credit on its own repetition.
    """
    reasoner, bodies = wire(settings_for, [RELOOK_REPLY], calls)
    first = await reasoner.plan(briefing)
    again = await reasoner.plan(briefing)

    assert calls.count("POST /v1/chat/completions") == 1
    assert len(bodies) == 1
    assert again == first


# ----------------------------------------------------------------------- bad replies


async def test_a_flattened_reply_is_re_wrapped_without_spending_a_second_call(
    settings_for: Callable[..., Settings], calls: list[str], briefing: Briefing
) -> None:
    reasoner, bodies = wire(
        settings_for,
        ['{"thought": "Ending.", "tool": "finish", "summary": "Nothing else to test here."}'],
        calls,
    )
    turn = await reasoner.plan(briefing)
    assert isinstance(turn.action, FinishAction)
    assert len(bodies) == 1, "re-wrapping is free; re-asking costs a turn and the budget"


async def test_a_reply_with_no_reasoning_is_labelled_as_one(
    settings_for: Callable[..., Settings], calls: list[str], briefing: Briefing
) -> None:
    reasoner, _bodies = wire(
        settings_for,
        ['{"action": {"tool": "finish", "summary": "No thought supplied."}}'],
        calls,
    )
    turn = await reasoner.plan(briefing)
    assert turn.thought == "(the model supplied no reasoning for this move)"


async def test_a_first_reply_that_is_not_an_action_buys_exactly_one_repair_turn(
    settings_for: Callable[..., Settings], calls: list[str], briefing: Briefing
) -> None:
    reasoner, bodies = wire(settings_for, [NO_JSON_REPLY, FINISH_REPLY], calls)
    turn = await reasoner.plan(briefing)

    assert isinstance(turn.action, FinishAction)
    assert len(bodies) == 2
    sent = bodies[1]["messages"]
    assert sent[0]["content"] == AGENT_SYSTEM
    assert sent[2] == {"role": "assistant", "content": NO_JSON_REPLY}
    # The repair has to name the error, or the model repeats the same mistake and the
    # second call is another token bill for nothing.
    assert "could not be used as an action" in sent[3]["content"]
    assert "no JSON object in response" in sent[3]["content"]


async def test_a_second_bad_reply_stops_rather_than_retrying_into_silence(
    settings_for: Callable[..., Settings], calls: list[str], briefing: Briefing
) -> None:
    reasoner, bodies = wire(settings_for, [NO_JSON_REPLY, UNKNOWN_TOOL_REPLY], calls)
    with pytest.raises(ReasonerError, match="after one repair attempt") as exc:
        await reasoner.plan(briefing)
    assert len(bodies) == 2, "bounded at two calls however much budget is left"
    assert "First:" in str(exc.value)
    assert "Second:" in str(exc.value)


def test_an_action_that_is_not_an_object_is_named_as_one() -> None:
    with pytest.raises(ReasonerError, match=r"'action' was list"):
        _parse_turn('{"thought": "t", "action": [1, 2]}')


async def test_an_endpoint_failure_becomes_the_error_the_loop_catches(
    settings_for: Callable[..., Settings], calls: list[str], briefing: Briefing
) -> None:
    settings = settings_for(nebius_api_key="k", reasoning_model=REASONING_MODEL, max_retries=0)
    reasoner = NemotronReasoner(settings, client_with(settings, refusing, calls))
    with pytest.raises(ReasonerError, match="HTTP 404"):
        await reasoner.plan(briefing)
    await reasoner.aclose()


# ----------------------------------------------------------------------- model resolution


async def test_the_reasoning_model_is_resolved_once_and_then_remembered(
    settings_for: Callable[..., Settings], calls: list[str]
) -> None:
    settings = settings_for(nebius_api_key="k")
    reasoner = NemotronReasoner(settings, client_with(settings, catalog_only, calls))
    assert await reasoner.model_id() == REASONING_MODEL
    assert await reasoner.model_id() == REASONING_MODEL
    assert calls == ["GET /v1/models"]
    assert reasoner.usage.model == REASONING_MODEL
    await reasoner.aclose()


async def test_an_account_with_no_reasoning_model_fails_twice_rather_than_once(
    settings_for: Callable[..., Settings], calls: list[str]
) -> None:
    def bare(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "someone/vision-only"}]})

    settings = settings_for(nebius_api_key="k")
    reasoner = NemotronReasoner(settings, client_with(settings, bare, calls))
    assert reasoner.usage.model == "unresolved"
    for _ in range(2):
        with pytest.raises(ReasonerError, match="Set VIGIL_REASONING_MODEL"):
            await reasoner.model_id()
    await reasoner.aclose()


async def test_closing_is_left_to_whoever_built_the_client(
    settings_for: Callable[..., Settings], calls: list[str]
) -> None:
    settings = settings_for(nebius_api_key="k", reasoning_model=REASONING_MODEL)
    shared = client_with(settings, catalog_only, calls)
    await NemotronReasoner(settings, shared).aclose()
    assert not shared._http.is_closed, "one client is shared by vision and reasoning"
    await shared.aclose()

    owned = NemotronReasoner(settings)
    assert owned._own_client
    await owned.aclose()
    assert owned.client._http.is_closed


# ----------------------------------------------------------------------- backend selection


async def test_the_reasoning_backend_switch_mirrors_the_perception_one(
    settings_for: Callable[..., Settings],
) -> None:
    assert isinstance(build_reasoner(settings_for(reasoning_backend="mock")), ScriptedReasoner)
    assert isinstance(build_reasoner(settings_for()), ScriptedReasoner)
    hosted = settings_for(nebius_api_key="k", reasoning_model=REASONING_MODEL)
    built = build_reasoner(hosted)
    assert isinstance(built, NemotronReasoner)
    assert built.name == "nebius"
    await built.aclose()


async def test_a_reasoning_backend_that_cannot_plan_says_so(
    settings_for: Callable[..., Settings], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Settings, "resolved_reasoning", property(lambda self: "clipo"))
    with pytest.raises(ReasonerError, match="cosmos is a perception backend"):
        build_reasoner(settings_for())
    with pytest.raises(ReasonerError, match="use mock, nebius or auto"):
        build_reasoner(settings_for(reasoning_backend="cosmos"))


async def test_a_planner_with_no_backend_of_its_own_names_itself_and_holds_nothing() -> None:
    class Bare(Reasoner):
        name = "bare"

        async def plan(self, briefing: Briefing) -> Turn:
            raise AssertionError("not the thing under test")

        @property
        def usage(self) -> Usage:
            return Usage(prompt_tokens=3)

    bare = Bare()
    assert await bare.model_id() == "bare"
    assert bare.usage.total_tokens == 3
    await bare.aclose()


# ----------------------------------------------------------------------- the two clocks


def _lead(**overrides: object) -> _Lead:
    base = _Lead(
        key="struck_by:P1-V1",
        hazard_type="struck_by",
        mitigation_key="struck_by",
        participants=("P1", "V1"),
        rationale="Both routes cross the pillar before the window ends.",
        severity=5,
        lead_time_s=2.9,
        window=(0.0, 4.0),
        confidence=0.7,
    )
    for name, value in overrides.items():
        setattr(base, name, value)
    return base


@pytest.mark.parametrize(
    ("contact_s", "expected"),
    [
        (0.0, Likelihood.IMMINENT),
        (1.0, Likelihood.IMMINENT),
        (1.1, Likelihood.LIKELY),
        (6.0, Likelihood.LIKELY),
        (6.1, Likelihood.POSSIBLE),
        (12.0, Likelihood.POSSIBLE),
        (12.1, Likelihood.REMOTE),
    ],
)
def test_urgency_is_read_from_the_countdown_never_from_the_warning(
    contact_s: float, expected: Likelihood
) -> None:
    assert _likelihood(_lead(contact_s=contact_s, lead_time_s=30.0)) is expected


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [(0.6, Likelihood.POSSIBLE), (0.59, Likelihood.REMOTE)],
)
def test_a_standing_exposure_with_no_countdown_is_graded_on_confidence(
    confidence: float, expected: Likelihood
) -> None:
    assert _likelihood(_lead(contact_s=None, confidence=confidence)) is expected


def test_a_longer_warning_does_not_make_the_same_collision_less_dangerous() -> None:
    """The bug these lines pin: reading the warning clock as if it were the countdown meant a
    finer sample produced a longer warning and therefore a *lower* risk score."""
    early = _lead(contact_s=0.4, lead_time_s=2.9)
    late = _lead(contact_s=0.4, lead_time_s=0.3)
    assert _likelihood(early) is _likelihood(late)


# ----------------------------------------------------------------------- the scripted planner


def _conflict_scene(
    *, time_to_event_s: float | None, span: tuple[float, float] = (0.0, 4.0)
) -> SceneUnderstanding:
    return SceneUnderstanding(
        clip_id="s",
        t0_s=span[0],
        t1_s=span[1],
        entities=[
            Entity(ref="P1", category="worker", location="crossing"),
            Entity(ref="V1", category="forklift", location="backing out"),
        ],
        conflicts=[
            Conflict(
                participants=["P1", "V1"],
                kind="struck_by",
                time_to_event_s=time_to_event_s,
                severity_hint=5,
                rationale="Routes cross at the pillar.",
            )
        ],
        key_moments=[KeyMoment(t_s=2.0, description="V1 moves.")],
        confidence=0.7,
    )


def _briefing(
    video: VideoSource,
    *scenes: SceneUnderstanding,
    turn: int = 1,
    log: tuple[Observation, ...] = (),
) -> Briefing:
    return Briefing(
        video=video,
        turn=turn,
        max_turns=8,
        tokens_used=0,
        token_budget=60_000,
        log=log,
        scenes=tuple(scenes),
    )


async def test_a_relook_aims_at_the_moment_the_prediction_puts_the_contact_at(
    blind_corner: VideoSource,
) -> None:
    planner = ScriptedReasoner()
    turn = await planner.plan(_briefing(blind_corner, _conflict_scene(time_to_event_s=1.4)))
    assert isinstance(turn.action, RelookAction)
    # The window ends at 4.0 s and contact is 1.4 s after that, so a follow-up that stayed
    # inside the window could not settle the thing it was asked to settle.
    assert turn.action.t1_s > 4.0
    assert turn.action.t1_s <= blind_corner.duration_s


async def test_a_prediction_the_model_cannot_time_falls_back_to_the_whole_window(
    blind_corner: VideoSource,
) -> None:
    planner = ScriptedReasoner()
    turn = await planner.plan(_briefing(blind_corner, _conflict_scene(time_to_event_s=None)))
    assert isinstance(turn.action, RelookAction)
    assert (turn.action.t0_s, turn.action.t1_s) == (0.0, 4.0)


async def test_a_prediction_beyond_the_footage_does_not_sample_a_window_that_is_not_there(
    blind_corner: VideoSource,
) -> None:
    """A contact 14 s into a 10 s clip is unobservable. Re-looking at the original window is
    the honest move; an empty slice past the end of the file is a wasted vision call.
    """
    assert blind_corner.duration_s < 12.0
    planner = ScriptedReasoner()
    turn = await planner.plan(_briefing(blind_corner, _conflict_scene(time_to_event_s=10.0)))
    assert isinstance(turn.action, RelookAction)
    assert (turn.action.t0_s, turn.action.t1_s) == (0.0, 4.0)


async def test_an_observation_is_folded_into_the_lead_that_asked_for_it(
    blind_corner: VideoSource,
) -> None:
    """The transcript comes back as a list; the lead keeps the part it asked for.

    Without this the re-sample is a vision call whose result goes nowhere, and the finding is
    filed on the first pass's confidence -- which is the exact behaviour a reviewer would call
    "it looked again and learned nothing".
    """
    scene = _conflict_scene(time_to_event_s=1.4)
    planner = ScriptedReasoner()
    await planner.plan(_briefing(blind_corner, scene))
    key = planner._order[0]
    lead = planner._leads[key]
    assert lead.answer is None
    assert lead.rules == ()

    answer = PerceptionAnswer(
        question="Do their routes meet?",
        answer="Yes, at the pillar.",
        hazard_present=True,
        confidence=0.91,
    )
    await planner.plan(
        _briefing(
            blind_corner,
            scene,
            turn=2,
            log=(Observation(tool="relook", summary="s", body="held", answer=answer),),
        )
    )
    assert lead.answer is answer

    rules = (
        Scored(
            rule=Rule(
                id="osha-struck-by",
                source="29 CFR 1910.178",
                title="Pedestrians",
                text="Keep pedestrians clear of the travel route.",
                hazards=[HazardClass.STRUCK_BY],
                severity_floor=4,
            ),
            score=0.8,
        ),
    )
    flagged = await planner.plan(
        _briefing(
            blind_corner,
            scene,
            turn=3,
            log=(Observation(tool="policy_search", summary="s", body="rules", rules=rules),),
        )
    )
    assert lead.rules == rules
    assert isinstance(flagged.action, FlagAction)
    assert flagged.action.rule_ids == [scored.rule.id for scored in rules], (
        "a finding cites what its own search returned, nothing else"
    )


async def test_a_harvest_ignores_an_observation_from_the_wrong_tool(
    blind_corner: VideoSource,
) -> None:
    """An answer to a different question must not be mistaken for the one still owed."""
    scene = _conflict_scene(time_to_event_s=1.4)
    planner = ScriptedReasoner()
    await planner.plan(_briefing(blind_corner, scene))
    key = planner._order[0]
    assert planner._leads[key].answer is None

    off_topic = _briefing(
        blind_corner,
        scene,
        turn=2,
        log=(Observation(tool="policy_search", summary="s", body="a list of rules"),),
    )
    turn = await planner.plan(off_topic)
    assert planner._leads[key].answer is None, "a rule list is not an answer about the footage"
    assert isinstance(turn.action, PolicyAction), "the relook is still owed, the rule is next"


async def test_a_lead_whose_re_sample_returned_nothing_still_reaches_a_flag(
    blind_corner: VideoSource,
) -> None:
    """A re-sample that came back empty is not a reason to drop the finding.

    ``answer`` stays None through the whole pipeline here, which is what happens when the vision
    model replies in prose instead of JSON: the run degrades to uncited and low-confidence, it
    does not go silent on a struck-by.
    """
    scene = _conflict_scene(time_to_event_s=1.4)
    planner = ScriptedReasoner()
    actions = [
        (await planner.plan(_briefing(blind_corner, scene, turn=turn))).action
        for turn in range(1, 4)
    ]
    assert isinstance(actions[0], RelookAction)
    assert isinstance(actions[1], PolicyAction)
    flag = actions[2]
    assert isinstance(flag, FlagAction)
    assert flag.rule_ids == []
    assert flag.confidence < 0.7, "nothing retrieved, so nothing to claim confidently"
    assert "No rule was retrieved" in flag.reasoning
    assert flag.severity == 5, "the countdown stands on its own, citations or not"
    finish = await planner.plan(_briefing(blind_corner, scene, turn=4))
    assert isinstance(finish.action, FinishAction)
