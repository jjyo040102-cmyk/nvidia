"""The local edge path, run against a GPU that is not there.

``cosmos`` is the backend whose existence is the pitch -- physical common sense from NVIDIA's
model, on the customer's machine, with nothing leaving the building -- so having never executed
it was the largest untested claim in the repository. Nothing below pretends to test Cosmos: the
weights are not installed and cannot be. What is pinned is the part Vigil owns -- that a machine
which cannot run it is told why in the same breath, that the clip is handed over as video rather
than as a fabricated frame count, that one resident model means one inference at a time, and
that whatever string comes back becomes the same scene and answer the hosted path produces.

The fakes are deliberately thin. Where real HuggingFace behaviour could disagree with them --
the ``device_map`` spelling, which dtype keyword a given transformers major accepts -- the module
under test carries the comment that says why that spelling was chosen.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
import types
from collections.abc import Callable
from typing import Any, cast

import pytest

from vigil.config import Settings
from vigil.perception.base import Clip, PerceptionError, scene_json_schema
from vigil.perception.cosmos import CosmosPerception
from vigil.prompts import ANSWER_SCHEMA, QUERY_SYSTEM, SURVEY_SYSTEM

PROMPT_LEN = 400
COMPLETION_LEN = 24
MODEL_ID = "nvidia/Cosmos-Reason2-2B"

SCENE_REPLY = json.dumps(
    {
        "clip_id": "v@2.0-6.0",
        "t0_s": 2.0,
        "t1_s": 6.0,
        "area_type": "warehouse",
        "entities": [
            {"ref": "P1", "category": "worker", "role": "vulnerable_party", "location": "aisle"},
            {
                "ref": "FL1",
                "category": "forklift",
                "role": "powered_machine",
                "location": "approaching",
            },
        ],
        "conflicts": [
            {
                "participants": ["P1", "FL1"],
                "kind": "struck_by",
                "severity_hint": 4,
                "time_to_event_s": 2.5,
                "rationale": "Both reach the aisle mouth within 2.5 s.",
            }
        ],
        "dynamics_narrative": "FL1 is reversing toward the aisle mouth as P1 walks into it.",
    }
)

ANSWER_REPLY = json.dumps(
    {
        "answer": "The horn is not sounded and the driver's head is down.",
        "hazard_present": True,
        "confidence": 0.72,
        "observed_at_s": 4.4,
        "entities_mentioned": ["P1", "FL1"],
    }
)

CLIP = Clip(video_id="v", path="data/clips/v.mp4", t0_s=2.0, t1_s=6.0, camera_label="CAM-3")


class FakeTensor:
    """Shape, ``to()`` and a slice -- everything the module asks a tensor to do."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.devices: list[Any] = []

    def to(self, device: Any) -> FakeTensor:
        self.devices.append(device)
        return self

    def __getitem__(self, key: Any) -> FakeTensor:
        return self


class Processor:
    def __init__(self, stack: FakeStack) -> None:
        self.stack = stack

    def apply_chat_template(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> dict[str, FakeTensor]:
        self.stack.chat_calls.append((messages, kwargs))
        return {"input_ids": FakeTensor((1, PROMPT_LEN))}

    def batch_decode(self, _trimmed: FakeTensor, **_kwargs: Any) -> list[str]:
        return [self.stack.reply]


class Model:
    device = "cuda:0"

    def __init__(self, stack: FakeStack) -> None:
        self.stack = stack

    def generate(self, **kwargs: Any) -> FakeTensor:
        return self.stack.generate(**kwargs)


class FakeStack:
    """The three imports ``_ensure_loaded`` makes, recorded rather than performed."""

    def __init__(self, reply: str = SCENE_REPLY) -> None:
        self.reply = reply
        self.cuda_available = True
        self.free_mib = 20_000
        self.model_error: Exception | None = None
        self.generate_error: Exception | None = None
        self.processors: list[str] = []
        self.loads: list[tuple[str, dict[str, Any]]] = []
        self.chat_calls: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []
        self.generations: list[dict[str, Any]] = []
        self._concurrent = 0
        self._peak = 0
        self._gate = threading.Lock()

    @property
    def peak_concurrency(self) -> int:
        return self._peak

    def install(
        self, monkeypatch: pytest.MonkeyPatch, *, torch: bool = True, bits: bool = True
    ) -> None:
        """Patch the modules in; ``torch=False`` stands in for an optional extra not installed.

        ``SimpleNamespace`` rather than ``ModuleType`` because these stand in for the attributes
        the module reaches for, and the cast is where the lie is admitted: ``sys.modules`` holds
        modules, and a namespace with the right members is close enough for an import statement.
        """
        if torch:
            monkeypatch.setitem(
                sys.modules,
                "torch",
                cast(
                    "types.ModuleType",
                    types.SimpleNamespace(
                        bfloat16="bfloat16",
                        cuda=types.SimpleNamespace(
                            is_available=lambda: self.cuda_available,
                            mem_get_info=lambda: (self.free_mib * 1024 * 1024, 24 * 1024**3),
                        ),
                    ),
                ),
            )
        else:
            # ``delitem`` is not enough on a developer workstation that really has torch:
            # the import system would immediately load it again from site-packages.  A None
            # sentinel is how Python records a failed import and makes this test independent
            # of what optional packages happen to be installed on the host running pytest.
            monkeypatch.setitem(sys.modules, "torch", None)
        if bits:
            monkeypatch.setitem(sys.modules, "bitsandbytes", types.ModuleType("bitsandbytes"))
        else:
            monkeypatch.setitem(sys.modules, "bitsandbytes", None)

        stack = self

        class AutoProcessor:
            @staticmethod
            def from_pretrained(model_id: str, **_kwargs: Any) -> Processor:
                stack.processors.append(model_id)
                return Processor(stack)

        class AutoModelForCausalLM:
            @staticmethod
            def from_pretrained(model_id: str, **kwargs: Any) -> Model:
                if stack.model_error is not None:
                    raise stack.model_error
                stack.loads.append((model_id, kwargs))
                return Model(stack)

        monkeypatch.setitem(
            sys.modules,
            "transformers",
            cast(
                "types.ModuleType",
                types.SimpleNamespace(
                    AutoProcessor=AutoProcessor, AutoModelForCausalLM=AutoModelForCausalLM
                ),
            ),
        )

    def generate(self, **kwargs: Any) -> FakeTensor:
        self.generations.append(kwargs)
        with self._gate:
            self._concurrent += 1
            self._peak = max(self._peak, self._concurrent)
        try:
            if self.generate_error is not None:
                raise self.generate_error
            time.sleep(0.02)  # long enough that an unlocked backend would overlap here
            return FakeTensor((1, PROMPT_LEN + COMPLETION_LEN))
        finally:
            with self._gate:
                self._concurrent -= 1


@pytest.fixture
def stack(monkeypatch: pytest.MonkeyPatch) -> FakeStack:
    fake = FakeStack()
    fake.install(monkeypatch)
    return fake


def cosmos(settings_for: Callable[..., Settings], **overrides: Any) -> CosmosPerception:
    return CosmosPerception(settings_for(perception_backend="cosmos", **overrides))


# -------------------------------------------------------------------------- what gets loaded


async def test_naming_the_model_never_pulls_weights_onto_the_gpu(
    settings_for: Callable[..., Settings], stack: FakeStack
) -> None:
    """A report footer asks for this string; loading gigabytes to print it is a bad trade."""
    assert await cosmos(settings_for).model_id() == f"cosmos/{MODEL_ID}"
    assert stack.loads == []
    assert stack.processors == []


async def test_a_machine_without_the_optional_extra_is_told_the_command(
    settings_for: Callable[..., Settings], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeStack()
    fake.install(monkeypatch, torch=False)
    with pytest.raises(PerceptionError, match=r'pip install -e "\.\[cosmos\]"'):
        await cosmos(settings_for).survey(CLIP)


async def test_quantising_without_bitsandbytes_names_that_package_not_the_licence(
    settings_for: Callable[..., Settings], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``load_in_4bit`` fails deep inside ``from_pretrained``; the honest fix is one import."""
    fake = FakeStack()
    fake.install(monkeypatch, bits=False)
    with pytest.raises(PerceptionError, match="bitsandbytes"):
        await cosmos(settings_for).survey(CLIP)


async def test_a_full_precision_load_does_not_demand_the_quantiser(
    settings_for: Callable[..., Settings], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeStack()
    fake.install(monkeypatch, bits=False)
    await cosmos(settings_for, cosmos_quantize="none").survey(CLIP)
    assert "load_in_4bit" not in fake.loads[0][1]


async def test_no_cuda_is_answered_with_the_way_out(
    settings_for: Callable[..., Settings], stack: FakeStack
) -> None:
    stack.cuda_available = False
    with pytest.raises(PerceptionError, match="no CUDA device visible") as excinfo:
        await cosmos(settings_for).survey(CLIP)
    assert "VIGIL_PERCEPTION_BACKEND=nebius" in str(excinfo.value)


async def test_a_card_that_is_too_small_says_how_much_is_free(
    settings_for: Callable[..., Settings], stack: FakeStack
) -> None:
    stack.free_mib = 4000
    with pytest.raises(PerceptionError, match="only 4000 MiB of VRAM is free"):
        await cosmos(settings_for).survey(CLIP)
    assert stack.loads == []


async def test_a_load_failure_is_reported_as_the_gated_model_it_is(
    settings_for: Callable[..., Settings], stack: FakeStack
) -> None:
    stack.model_error = OSError("401 client error")
    with pytest.raises(PerceptionError, match="gated model") as excinfo:
        await cosmos(settings_for).survey(CLIP)
    assert "huggingface-cli login" in str(excinfo.value)
    assert MODEL_ID in str(excinfo.value)


async def test_the_weights_are_read_into_memory_once(
    settings_for: Callable[..., Settings], stack: FakeStack
) -> None:
    backend = cosmos(settings_for)
    await backend.survey(CLIP)
    await backend.ask(CLIP, "Is the driver looking at the aisle?")
    assert len(stack.loads) == 1, "a second load would double both the VRAM and the wait"


@pytest.mark.parametrize(
    ("quantize", "flag"), [("4bit", "load_in_4bit"), ("8bit", "load_in_8bit"), ("none", None)]
)
async def test_the_quantise_setting_becomes_the_load_argument(
    settings_for: Callable[..., Settings], stack: FakeStack, quantize: str, flag: str | None
) -> None:
    await cosmos(settings_for, cosmos_quantize=quantize).survey(CLIP)
    model_id, kwargs = stack.loads[0]
    assert model_id == MODEL_ID
    assert {k for k in ("load_in_4bit", "load_in_8bit") if k in kwargs} == (
        {flag} if flag else set()
    )
    assert kwargs["torch_dtype"] == "bfloat16"


async def test_a_pinned_device_is_a_single_value_map_because_a_bare_string_is_not(
    settings_for: Callable[..., Settings], stack: FakeStack
) -> None:
    await cosmos(settings_for, cosmos_device="cuda:1").survey(CLIP)
    assert stack.loads[0][1]["device_map"] == {"": "cuda:1"}
    assert stack.generations[0]["input_ids"].devices == ["cuda:0"]


async def test_auto_device_leaves_the_splitting_to_accelerate(
    settings_for: Callable[..., Settings], stack: FakeStack
) -> None:
    await cosmos(settings_for, cosmos_device="auto").survey(CLIP)
    assert stack.loads[0][1]["device_map"] == "auto"


# ---------------------------------------------------------------------------- what gets asked


async def test_the_clip_is_handed_over_as_video_never_as_a_fabricated_frame_count(
    settings_for: Callable[..., Settings], stack: FakeStack
) -> None:
    """The template used to say "0 frames are attached" while a whole clip was attached.

    A prompt that describes the wrong input is worse than a shorter one: the model either
    refuses or invents frames to describe.
    """
    await cosmos(settings_for).survey(CLIP)
    messages, template_kwargs = stack.chat_calls[0]
    assert template_kwargs == {
        "add_generation_prompt": True,
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    assert messages[0]["role"] == "system"
    assert messages[0]["content"][0]["text"] == SURVEY_SYSTEM.format(schema=scene_json_schema())
    user = messages[1]["content"]
    assert {"type": "video", "video": CLIP.path} in user
    text = user[-1]["text"]
    assert "0 frames" not in text
    assert "video segment" in text
    assert "CAM-3" in text
    assert "2.0s to 6.0s" in text


async def test_a_survey_becomes_a_scene_and_a_token_bill(
    settings_for: Callable[..., Settings], stack: FakeStack
) -> None:
    scene, usage = await cosmos(settings_for).survey(CLIP)
    assert scene.area_type.value == "warehouse"
    assert scene.conflicts[0].time_to_event_s == 2.5
    assert scene.repair_notes == [], "the reply was well formed, so nothing was guessed"
    assert usage.prompt_tokens == PROMPT_LEN
    assert usage.completion_tokens == COMPLETION_LEN
    assert usage.model == f"cosmos/{MODEL_ID}"
    assert stack.generations[0]["max_new_tokens"] == 1024
    assert stack.generations[0]["do_sample"] is False


async def test_a_follow_up_asks_the_same_question_the_hosted_path_asks(
    settings_for: Callable[..., Settings], stack: FakeStack
) -> None:
    """The local path once sent a schema holding only an ``answer`` field.

    ``hazard_present`` was therefore never requested, so every re-look came back CLEAR by
    construction -- a follow-up that reads like evidence and cannot contradict anything.
    """
    stack.reply = ANSWER_REPLY
    answer = await cosmos(settings_for).ask(CLIP, "Is the horn sounded?")
    messages, _kwargs = stack.chat_calls[0]
    system = messages[0]["content"][0]["text"]
    assert system == QUERY_SYSTEM.format(schema=ANSWER_SCHEMA)
    assert "hazard_present" in system
    assert answer.hazard_present is True
    assert answer.confidence == pytest.approx(0.72)
    assert answer.observed_at_s == pytest.approx(4.4)
    assert answer.entities_mentioned == ["P1", "FL1"]
    assert answer.usage.total_tokens == PROMPT_LEN + COMPLETION_LEN


async def test_a_prose_answer_is_kept_rather_than_thrown_away(
    settings_for: Callable[..., Settings], stack: FakeStack
) -> None:
    """A small model that answers in sentences still answered; the confidence says how little."""
    stack.reply = "The worker steps into the aisle just as the forklift backs toward it."
    answer = await cosmos(settings_for).ask(CLIP, "Does the worker enter the aisle?")
    assert answer.answer.startswith("The worker steps into the aisle")
    assert answer.hazard_present is False
    assert answer.confidence == pytest.approx(0.4)
    assert answer.observed_at_s is None


async def test_inference_failure_is_a_perception_error_not_a_cuda_traceback(
    settings_for: Callable[..., Settings], stack: FakeStack
) -> None:
    stack.generate_error = RuntimeError("CUDA out of memory")
    with pytest.raises(PerceptionError, match=r"Cosmos inference failed on v@2\.0-6\.0") as excinfo:
        await cosmos(settings_for).survey(CLIP)
    assert "out of memory" in str(excinfo.value)


async def test_one_resident_model_means_one_inference_at_a_time(
    settings_for: Callable[..., Settings], stack: FakeStack
) -> None:
    """Two parallel re-looks on one model in one pool of VRAM is an OOM, not a speed-up."""
    backend = cosmos(settings_for)
    await asyncio.gather(backend.survey(CLIP), backend.survey(CLIP))
    assert stack.peak_concurrency == 1
    assert len(stack.generations) == 2
