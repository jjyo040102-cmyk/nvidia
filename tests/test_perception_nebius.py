"""The vision backend that spends the credit, exercised against an endpoint that is not real.

``survey`` and ``ask`` are the only two places in Vigil that turn pixels into a claim, so they
are the two places where a mock backend proves the least. Pinned here are the parts a scripted
read cannot pin for us: that the frames actually reach the wire as JPEGs at the requested
density, that a follow-up re-samples denser than the first pass did, that a reply wrapped in
prose still becomes a scene, and that every failure arrives as a :class:`PerceptionError` the
agent loop already knows how to catch -- not as an httpx traceback, and not as a charge.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from typing import Any

import httpx
import numpy as np
import pytest

from vigil.config import Settings
from vigil.nebius import NebiusError
from vigil.perception.base import Clip, PerceptionError, scene_json_schema
from vigil.perception.nebius import NebiusPerception
from vigil.prompts import ANSWER_SCHEMA, QUERY_SYSTEM, SURVEY_SYSTEM
from vigil.video.sampler import Frame, MediaError
from vigil.video.source import VideoSource

from .helpers import CATALOG, VISION_MODEL, client_with, completion, skip_without_ffmpeg

USAGE = {"prompt_tokens": 900, "completion_tokens": 120}

SCENE_REPLY = json.dumps(
    {
        "clip_id": "c",
        "t0_s": 2.0,
        "t1_s": 6.0,
        "area_type": "warehouse",
        "entities": [
            {"ref": "P1", "category": "worker", "location": "centre-left, on the machine lane"},
            {"ref": "V1", "category": "forklift", "location": "right, backing out"},
        ],
        "conflicts": [
            {
                "participants": ["P1", "V1"],
                "kind": "struck_by",
                "time_to_event_s": 1.4,
                "severity_hint": 5,
                "rationale": "Both routes cross at the pillar before the window ends.",
            }
        ],
        "key_moments": [{"t_s": 4.2, "description": "V1's reverse lights come on."}],
        "confidence": 0.72,
    }
)

ANSWER_REPLY = json.dumps(
    {
        "answer": "Yes. Contact is at 5.5 s, with P1 already standing in the lane.",
        "hazard_present": True,
        "confidence": 0.8,
        "observed_at_s": 5.5,
        "entities_mentioned": ["P1", "V1"],
    }
)


# --------------------------------------------------------------------------- the fake endpoint


class FakeEndpoint:
    """Answers with what the test hands it, and remembers exactly what it was sent."""

    def __init__(
        self,
        reply: str | list[str],
        *,
        catalog: dict[str, Any] | None = CATALOG,
        status: int | None = None,
    ) -> None:
        self.replies = [reply] if isinstance(reply, str) else list(reply)
        self.catalog = catalog
        self.status = status
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            if self.catalog is None:
                return httpx.Response(404, text="no catalog was requested")
            return httpx.Response(200, json=self.catalog)
        if self.status is not None:
            return httpx.Response(self.status, text="upstream said no")
        self.bodies.append(json.loads(request.content.decode()))
        reply = self.replies[min(len(self.bodies) - 1, len(self.replies) - 1)]
        return completion(reply, model=VISION_MODEL, **USAGE)

    @property
    def parts(self) -> list[dict[str, Any]]:
        content = self.bodies[-1]["messages"][-1]["content"]
        return list(content) if isinstance(content, list) else [{"type": "text", "text": content}]

    @property
    def system(self) -> str:
        return str(self.bodies[-1]["messages"][0]["content"])

    @property
    def text(self) -> str:
        return "".join(str(p.get("text", "")) for p in self.parts if p["type"] == "text")

    @property
    def images(self) -> list[str]:
        return [str(p["image_url"]["url"]) for p in self.parts if p["type"] == "image_url"]


class FakeSampler:
    """Stands in for ffmpeg: records the sampling plan, hands back frames of the asked-for size."""

    def __init__(self, *, empty: bool = False, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.empty = empty
        self.error = error

    def __call__(self, path: str, **kwargs: Any) -> list[Frame]:
        self.calls.append({"path": path, **kwargs})
        if self.error is not None:
            raise self.error
        if self.empty:
            return []
        t0, t1 = float(kwargs["t0"]), float(kwargs["t1"])
        count = int(kwargs["count"])
        step = (t1 - t0) / count
        return [
            Frame(t_s=min(t0 + i * step, t1), array=np.full((8, 8, 3), i * 9 % 256, np.uint8))
            for i in range(count)
        ]


def install_sampler(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> FakeSampler:
    sampler = FakeSampler(**kwargs)
    monkeypatch.setattr("vigil.perception.nebius.sample_frames", sampler)
    return sampler


def build(
    settings_for: Callable[..., Settings],
    endpoint: FakeEndpoint,
    calls: list[str],
    **overrides: object,
) -> NebiusPerception:
    """A backend wired to the fake account, with the model name pinned so no test has to
    depend on the order of ``VISION_PREFERENCE`` unless that is the thing under test."""
    settings = settings_for(nebius_api_key="k", vision_model=VISION_MODEL, **overrides)
    return NebiusPerception(settings, client_with(settings, endpoint, calls))


@pytest.fixture
def clip() -> Clip:
    return Clip(
        video_id="blind_corner_struck_by",
        path="data/clips/blind_corner_struck_by.mp4",
        t0_s=2.0,
        t1_s=6.0,
        camera_label="CAM-03",
    )


@pytest.fixture
def calls() -> list[str]:
    return []


# ----------------------------------------------------------------------- model resolution


async def test_the_vision_slot_is_chosen_from_the_account_once_and_then_remembered(
    settings_for: Callable[..., Settings], calls: list[str]
) -> None:
    endpoint = FakeEndpoint(SCENE_REPLY)
    settings = settings_for(nebius_api_key="k")
    backend = NebiusPerception(settings, client_with(settings, endpoint, calls))
    assert await backend.model() == VISION_MODEL
    assert await backend.model() == VISION_MODEL
    assert calls == ["GET /v1/models"], "a catalog fetch per survey would be a per-clip tax"
    await backend.aclose()


async def test_an_account_with_no_vision_model_is_a_named_failure_not_an_http_traceback(
    settings_for: Callable[..., Settings], calls: list[str]
) -> None:
    endpoint = FakeEndpoint(SCENE_REPLY, catalog={"data": [{"id": "someone/text-only"}]})
    settings = settings_for(nebius_api_key="k")
    backend = NebiusPerception(settings, client_with(settings, endpoint, calls))
    for _ in range(2):
        # Not remembered as resolved, and not cached as a blank either: the next clip gets the
        # same clear error instead of a half-initialised backend.
        with pytest.raises(PerceptionError, match="Set VIGIL_VISION_MODEL"):
            await backend.model()
    await backend.aclose()


async def test_a_pinned_model_never_spends_a_request_on_the_catalog(
    settings_for: Callable[..., Settings],
    calls: list[str],
    clip: Clip,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_sampler(monkeypatch)
    # ``catalog=None`` makes the endpoint answer 404, so a stray discovery request fails the
    # test loudly instead of quietly succeeding.
    endpoint = FakeEndpoint(SCENE_REPLY, catalog=None)
    backend = build(settings_for, endpoint, calls)
    await backend.survey(clip)
    assert calls == ["POST /v1/chat/completions"]


# ----------------------------------------------------------------------- what survey sends


async def test_a_survey_sends_every_frame_as_a_jpeg_and_asks_for_json(
    settings_for: Callable[..., Settings],
    calls: list[str],
    clip: Clip,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampler = install_sampler(monkeypatch)
    endpoint = FakeEndpoint(SCENE_REPLY)
    backend = build(settings_for, endpoint, calls)

    await backend.survey(clip)

    body = endpoint.bodies[0]
    assert body["model"] == VISION_MODEL
    assert body["response_format"] == {"type": "json_object"}
    assert body["max_tokens"] == 1200
    assert endpoint.system == SURVEY_SYSTEM.format(schema=scene_json_schema())
    assert sampler.calls == [{"path": clip.path, "t0": 2.0, "t1": 6.0, "count": 6, "max_px": 512}]
    assert len(endpoint.images) == 6
    for url in endpoint.images:
        assert url.startswith("data:image/jpeg;base64,")
        assert base64.b64decode(url.split(",", 1)[1])[:2] == b"\xff\xd8"


async def test_the_task_text_tells_the_model_which_clip_and_which_instants(
    settings_for: Callable[..., Settings],
    calls: list[str],
    clip: Clip,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_sampler(monkeypatch)
    endpoint = FakeEndpoint(SCENE_REPLY)
    backend = build(settings_for, endpoint, calls)
    await backend.survey(clip)

    # Absolute times, not "six evenly spaced frames": the speed the model derives comes from
    # these numbers, and a wrong assumption here is a wrong time-to-contact downstream.
    assert "blind_corner_struck_by@2.0-6.0" in endpoint.text
    assert "CAM-03" in endpoint.text
    assert "covering 2.0s to 6.0s" in endpoint.text
    assert "at these absolute times: 2.0s, 2.7s, 3.3s, 4.0s, 4.7s, 5.3s" in endpoint.text
    assert "0.0s" not in endpoint.text, "the clock must run from the source, not from the clip"


async def test_a_survey_returns_the_scene_and_the_bill_for_it(
    settings_for: Callable[..., Settings],
    calls: list[str],
    clip: Clip,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_sampler(monkeypatch)
    endpoint = FakeEndpoint(SCENE_REPLY)
    backend = build(settings_for, endpoint, calls)

    scene, usage = await backend.survey(clip)

    assert scene.area_type == "warehouse"
    assert scene.most_urgent is not None
    assert scene.most_urgent.time_to_event_s == pytest.approx(1.4)
    assert (usage.prompt_tokens, usage.completion_tokens) == (900, 120)
    assert usage.total_tokens == 1020
    # Prefixed with the backend, the same way the cosmos path labels itself, so a report can
    # never show two different names for one model within a single investigation.
    assert usage.model == f"nebius/{VISION_MODEL}"
    assert backend.client.spend.calls == 1


async def test_a_run_that_never_looked_says_so_instead_of_risking_a_catalog_call(
    settings_for: Callable[..., Settings],
    calls: list[str],
    clip: Clip,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``model_id`` is read while the report is being written.

    Resolving the model there could spend a request, and if that request failed it would throw
    away an investigation that had already spent its credit. So before any read the answer is
    the truth -- no model looked at anything -- and after one it is the name of what did.
    """
    install_sampler(monkeypatch)
    backend = build(settings_for, FakeEndpoint(SCENE_REPLY), calls)
    assert await backend.model_id() == "not reached"
    assert calls == []

    await backend.survey(clip)
    assert await backend.model_id() == VISION_MODEL, "the footer has to name the model that read"


async def test_an_account_that_cannot_list_its_models_fails_the_way_the_loop_catches(
    settings_for: Callable[..., Settings], calls: list[str]
) -> None:
    """Auto-pick needs the catalog. A key that cannot read it must surface as a PerceptionError,
    which the agent loop already knows how to turn into a stopped run and a printed reason --
    not as an httpx traceback from inside the first survey."""
    settings = settings_for(nebius_api_key="sk-not-a-real-one")
    backend = NebiusPerception(
        settings, client_with(settings, FakeEndpoint(SCENE_REPLY, catalog=None), calls)
    )

    with pytest.raises(PerceptionError) as excinfo:
        await backend.model()
    assert not isinstance(excinfo.value, httpx.HTTPError), "the wire detail stays behind the seam"
    assert calls, "the account was in fact asked"


async def test_a_reply_wrapped_in_prose_is_still_a_scene(
    settings_for: Callable[..., Settings],
    calls: list[str],
    clip: Clip,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_sampler(monkeypatch)
    endpoint = FakeEndpoint(f"Here is what I see:\n```json\n{SCENE_REPLY}\n```\nAnything else?")
    backend = build(settings_for, endpoint, calls)
    scene, _ = await backend.survey(clip)
    assert scene.clip_id == "c"


async def test_a_conflict_over_an_undeclared_entity_is_repaired_rather_than_thrown_away(
    settings_for: Callable[..., Settings],
    calls: list[str],
    clip: Clip,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_sampler(monkeypatch)
    raw = json.loads(SCENE_REPLY)
    raw["conflicts"][0]["participants"] = ["P1", "Pedestrian A"]
    endpoint = FakeEndpoint(json.dumps(raw))
    backend = build(settings_for, endpoint, calls)

    scene, _ = await backend.survey(clip)

    assert any("PEDESTRIANA" in note for note in scene.repair_notes)
    # Looked up in the raw form the model wrote, which is the whole point of ``normalise_ref``.
    ghost = scene.entity("Pedestrian A")
    assert ghost is not None
    assert ghost.category == "unspecified"


async def test_a_scene_that_cannot_be_read_names_the_broken_field(
    settings_for: Callable[..., Settings],
    calls: list[str],
    clip: Clip,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_sampler(monkeypatch)
    endpoint = FakeEndpoint(
        json.dumps(
            {
                "clip_id": "c",
                "t0_s": 2.0,
                "t1_s": 6.0,
                "conflicts": [
                    {
                        "participants": ["P1"],
                        "kind": "struck_by",
                        "severity_hint": 4,
                        "rationale": "one participant is not a conflict",
                    }
                ],
            }
        )
    )
    backend = build(settings_for, endpoint, calls)
    with pytest.raises(PerceptionError, match=r"scene failed validation -> conflicts"):
        await backend.survey(clip)


async def test_a_reply_with_no_json_in_it_says_so(
    settings_for: Callable[..., Settings],
    calls: list[str],
    clip: Clip,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_sampler(monkeypatch)
    backend = build(settings_for, FakeEndpoint("I am not able to describe video."), calls)
    with pytest.raises(PerceptionError, match="no JSON object in response"):
        await backend.survey(clip)


async def test_an_http_failure_on_the_survey_path_is_renamed_for_the_caller(
    settings_for: Callable[..., Settings],
    calls: list[str],
    clip: Clip,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_sampler(monkeypatch)
    backend = build(settings_for, FakeEndpoint("", status=404), calls)
    with pytest.raises(PerceptionError, match=r"survey call failed: .*HTTP 404"):
        await backend.survey(clip)


async def test_a_window_that_decodes_to_nothing_is_not_billed_for(
    settings_for: Callable[..., Settings],
    calls: list[str],
    clip: Clip,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_sampler(monkeypatch, empty=True)
    endpoint = FakeEndpoint(SCENE_REPLY)
    backend = build(settings_for, endpoint, calls)
    with pytest.raises(PerceptionError, match="no decodable frames"):
        await backend.survey(clip)
    assert endpoint.bodies == [], "an empty window must not be sent as an empty request"


async def test_a_recording_that_will_not_decode_is_not_blamed_on_the_api(
    settings_for: Callable[..., Settings],
    calls: list[str],
    clip: Clip,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_sampler(monkeypatch, error=MediaError("ffprobe: invalid data found"))
    backend = build(settings_for, FakeEndpoint(SCENE_REPLY), calls)
    with pytest.raises(PerceptionError, match="ffprobe: invalid data") as exc:
        await backend.survey(clip)
    assert "call failed" not in str(exc.value)


# ----------------------------------------------------------------------- what ask sends


@pytest.mark.parametrize(
    ("frames_per_clip", "dense"), [(6, 12), (2, 4), (9, 16), (16, 16), (32, 16)]
)
async def test_a_follow_up_re_samples_the_same_window_more_densely_capped_at_sixteen(
    settings_for: Callable[..., Settings],
    calls: list[str],
    clip: Clip,
    monkeypatch: pytest.MonkeyPatch,
    frames_per_clip: int,
    dense: int,
) -> None:
    sampler = install_sampler(monkeypatch)
    endpoint = FakeEndpoint(ANSWER_REPLY)
    backend = build(settings_for, endpoint, calls, frames_per_clip=frames_per_clip)

    await backend.ask(clip, "Do their routes meet?")

    assert sampler.calls[0]["count"] == dense
    assert len(endpoint.images) == dense
    # Same window, same cost model: the follow-up is denser, not wider.
    assert sampler.calls[0]["t0"] == 2.0
    assert sampler.calls[0]["t1"] == 6.0


async def test_an_ask_asks_one_question_about_one_window(
    settings_for: Callable[..., Settings],
    calls: list[str],
    clip: Clip,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_sampler(monkeypatch)
    endpoint = FakeEndpoint(ANSWER_REPLY)
    backend = build(settings_for, endpoint, calls)

    answer = await backend.ask(clip, "Do their routes meet?")

    assert "Do their routes meet?" in endpoint.text
    assert "window 2.0s to 6.0s" in endpoint.text
    # The prompt verbatim, not "roughly the right shape": a judge reviewing the benchmark will
    # want to know exactly what the vision model was asked, and this is the only thing that says.
    assert endpoint.system == QUERY_SYSTEM.format(schema=ANSWER_SCHEMA)
    assert endpoint.bodies[0]["max_tokens"] == 420
    assert answer.verdict == "HAZARD"
    assert answer.confidence == pytest.approx(0.8)
    assert answer.observed_at_s == pytest.approx(5.5)
    assert answer.entities_mentioned == ["P1", "V1"]
    assert answer.usage.model == f"nebius/{VISION_MODEL}"


@pytest.mark.parametrize(("observed", "expected"), [(99.0, 6.0), (-4.0, 2.0), (None, None)])
async def test_a_time_the_model_reports_is_pulled_back_inside_the_window_it_was_shown(
    settings_for: Callable[..., Settings],
    calls: list[str],
    clip: Clip,
    monkeypatch: pytest.MonkeyPatch,
    observed: float | None,
    expected: float | None,
) -> None:
    install_sampler(monkeypatch)
    payload = json.loads(ANSWER_REPLY)
    payload["observed_at_s"] = observed
    backend = build(settings_for, FakeEndpoint(json.dumps(payload)), calls)

    answer = await backend.ask(clip, "When?")

    assert answer.observed_at_s == expected


async def test_a_reply_that_offers_nothing_is_reported_as_offering_nothing(
    settings_for: Callable[..., Settings],
    calls: list[str],
    clip: Clip,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_sampler(monkeypatch)
    backend = build(settings_for, FakeEndpoint("{}"), calls)

    answer = await backend.ask(clip, "Anything?")

    assert answer.answer == "no answer returned"
    assert answer.verdict == "CLEAR"
    assert answer.confidence == pytest.approx(0.4)
    assert answer.observed_at_s is None
    assert answer.entities_mentioned == []


async def test_entities_are_trimmed_to_the_ones_that_can_actually_be_read(
    settings_for: Callable[..., Settings],
    calls: list[str],
    clip: Clip,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_sampler(monkeypatch)
    payload = json.dumps(
        {"answer": "yes", "entities_mentioned": ["", "   ", 7, *[f"E{i}" for i in range(20)]]}
    )
    backend = build(settings_for, FakeEndpoint(payload), calls)

    answer = await backend.ask(clip, "Who?")

    assert answer.entities_mentioned == ["7", *[f"E{i}" for i in range(7)]]


async def test_an_http_failure_on_the_follow_up_path_is_renamed_for_the_caller(
    settings_for: Callable[..., Settings],
    calls: list[str],
    clip: Clip,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_sampler(monkeypatch)
    # 500 is retryable, so the ceiling is switched off here: this test is about which error
    # name the caller sees, not about how many times the client knocks.
    backend = build(settings_for, FakeEndpoint("", status=500), calls, max_retries=0)
    with pytest.raises(PerceptionError, match="ask call failed") as exc:
        await backend.ask(clip, "When?")
    assert "survey" not in str(exc.value)


async def test_a_follow_up_over_an_undecodable_recording_fails_the_same_way(
    settings_for: Callable[..., Settings],
    calls: list[str],
    clip: Clip,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dense re-sample is a second ffmpeg run, and it can be the one that breaks."""
    install_sampler(monkeypatch, error=MediaError("no frames after 6.0s"))
    endpoint = FakeEndpoint(ANSWER_REPLY)
    backend = build(settings_for, endpoint, calls)
    with pytest.raises(PerceptionError, match=r"no frames after 6\.0s"):
        await backend.ask(clip, "When do they meet?")
    assert endpoint.bodies == []


# ----------------------------------------------------------------------- the real pixels


def test_a_real_recording_is_sampled_at_the_configured_long_edge(
    settings_for: Callable[..., Settings], calls: list[str], blind_corner: VideoSource
) -> None:
    """The one test here that runs ffmpeg, because everything above it trusts ``_frames``."""
    skip_without_ffmpeg()
    endpoint = FakeEndpoint(SCENE_REPLY)
    backend = build(settings_for, endpoint, calls)
    clip = blind_corner.clip(0.0, min(3.0, blind_corner.duration_s))

    b64, frames = backend._frames(clip, density=4)

    assert 1 <= len(b64) == len(frames) <= 4
    assert frames[0].size == (512, 288), "frame_max_px caps the long edge, aspect kept"
    assert frames[0].t_s >= 0.0
    assert frames[-1].t_s <= clip.t1_s
    assert all(f.array.shape[2] == 3 for f in frames)
    assert endpoint.bodies == []


# ----------------------------------------------------------------------- ownership


async def test_closing_is_left_to_whoever_built_the_client(
    settings_for: Callable[..., Settings], calls: list[str]
) -> None:
    settings = settings_for(nebius_api_key="k", vision_model=VISION_MODEL)
    shared = client_with(settings, FakeEndpoint(SCENE_REPLY), calls)
    await NebiusPerception(settings, shared).aclose()
    assert not shared._http.is_closed, "the loop shares one client across vision and reasoning"
    await shared.aclose()

    owned = NebiusPerception(settings)
    assert owned._own_client
    await owned.aclose()
    assert owned.client._http.is_closed


def test_no_key_means_no_hosted_vision_backend_at_all(
    tmp_settings: Settings,
) -> None:
    with pytest.raises(NebiusError, match="VIGIL_NEBIUS_API_KEY"):
        NebiusPerception(tmp_settings)


def test_the_backend_says_what_it_can_do() -> None:
    assert NebiusPerception.name == "nebius"
    assert NebiusPerception.supports_video is True
