"""NVIDIA Cosmos Reason, run locally -- the edge path.

EXPERIMENTAL by design and labelled as such. Cosmos Reason is the model that gives Vigil its
thesis -- physical common sense about what is *about to* happen -- but running it here means
CUDA, torch and several gigabytes of weights on a machine with 12 GB of VRAM and 32 GB of free
disk. Official guidance puts Cosmos-Reason2-2B at 24 GB in BF16, so this module quantises and
degrades rather than pretending it will fit.

Everything that can go wrong here raises :class:`PerceptionError` with the actual fix, because
a non-obvious CUDA stack trace is the difference between "the entrant can run it" and "the
entrant gives up".
"""

from __future__ import annotations

import asyncio
from typing import Any

from vigil.config import Settings
from vigil.models.scene import SceneUnderstanding
from vigil.perception.base import (
    Clip,
    PerceptionAnswer,
    PerceptionBackend,
    PerceptionError,
    Usage,
    answer_from_payload,
    extract_json,
    repair_scene,
    scene_json_schema,
)
from vigil.prompts import (
    ANSWER_SCHEMA,
    ATTACHED_VIDEO,
    QUERY_SYSTEM,
    QUERY_TASK,
    SURVEY_SYSTEM,
    SURVEY_TASK,
)


class CosmosPerception(PerceptionBackend):
    name = "cosmos"
    supports_video = True

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: Any = None
        self._processor: Any = None
        self._lock = asyncio.Lock()

    async def model_id(self) -> str:
        # From settings, never from ``_ensure_loaded``: naming the model must not be a reason
        # to pull several gigabytes of weights onto the GPU while a report is being written.
        return f"cosmos/{self.settings.cosmos_model_id}"

    async def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        quant = self.settings.cosmos_quantize
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor

            if quant != "none":
                # Imported here rather than left to ``load_in_4bit``, which fails inside
                # ``from_pretrained`` several frames deep: without this backend the quantised
                # load cannot happen at all, and the licence is not the fix for that.
                import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise PerceptionError(
                f"the cosmos backend needs torch + transformers"
                f"{', and bitsandbytes to quantise' if quant != 'none' else ''}: "
                'pip install -e ".[cosmos]"  (roughly 4 GB of downloads)'
            ) from exc

        if not torch.cuda.is_available():
            raise PerceptionError(
                "no CUDA device visible, so Cosmos cannot run here. "
                "Use VIGIL_PERCEPTION_BACKEND=nebius or =mock."
            )
        free_mib = torch.cuda.mem_get_info()[0] // (1024 * 1024)
        if free_mib < 9000:
            raise PerceptionError(
                f"only {free_mib} MiB of VRAM is free; Cosmos Reason needs about 9 GiB after "
                "4-bit quantisation. Close other GPU work or use the nebius backend."
            )

        device = self.settings.cosmos_device
        kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            # ``{"": "cuda"}`` is accelerate's documented way of saying "all of it, on this
            # device"; a bare device string is not a valid device_map. ``auto`` leaves the
            # splitting to accelerate, which is what a pinned device is an override for.
            "device_map": "auto" if device in {"", "auto"} else {"": device},
        }
        if quant == "4bit":
            kwargs["load_in_4bit"] = True
            kwargs["torch_dtype"] = torch.bfloat16
        elif quant == "8bit":
            kwargs["load_in_8bit"] = True
            kwargs["torch_dtype"] = torch.bfloat16
        else:
            kwargs["torch_dtype"] = torch.bfloat16

        model_id = self.settings.cosmos_model_id
        try:
            self._processor = await asyncio.to_thread(
                AutoProcessor.from_pretrained, model_id, trust_remote_code=True
            )
            self._model = await asyncio.to_thread(
                AutoModelForCausalLM.from_pretrained, model_id, **kwargs
            )
        except Exception as exc:
            raise PerceptionError(
                f"could not load {model_id}: {type(exc).__name__}: {exc}. "
                "Cosmos Reason is a gated model: accept the licence on huggingface.co and run "
                "'huggingface-cli login', then retry. If the failure is about memory, keep "
                "VIGIL_COSMOS_QUANTIZE=4bit or use the nebius backend."
            ) from exc

    async def _generate(self, clip: Clip, system: str, instruction: str) -> tuple[str, Usage]:
        """One turn of inference, with the clip itself as the attached media.

        The lock is the whole point: this is a single model resident in one pool of VRAM, so two
        concurrent follow-ups would either OOM or quietly share a decoder state. The agent loop
        asks one question at a time already, and this makes that ordering safe against any other
        caller.
        """
        async with self._lock:
            await self._ensure_loaded()

            messages = [
                {"role": "system", "content": [{"type": "text", "text": system}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": clip.path},
                        {"type": "text", "text": instruction},
                    ],
                },
            ]
            try:
                inputs = await asyncio.to_thread(
                    self._processor.apply_chat_template,
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
                prompt_len = int(inputs["input_ids"].shape[-1])
                async with asyncio.timeout(600):
                    generated = await asyncio.to_thread(
                        self._model.generate,
                        **inputs,
                        max_new_tokens=1024,
                        do_sample=False,
                    )
                trimmed = generated[:, prompt_len:]
                text = self._processor.batch_decode(trimmed, skip_special_tokens=True)[0]
            except Exception as exc:
                raise PerceptionError(
                    f"Cosmos inference failed on {clip.id}: {type(exc).__name__}: {exc}"
                ) from exc

            return text, Usage(
                prompt_tokens=prompt_len,
                completion_tokens=max(0, int(generated.shape[-1]) - prompt_len),
                model=f"cosmos/{self.settings.cosmos_model_id}",
            )

    async def survey(self, clip: Clip) -> tuple[SceneUnderstanding, Usage]:
        instruction = SURVEY_TASK.format(
            clip_id=clip.id,
            t0=clip.t0_s,
            t1=clip.t1_s,
            camera=clip.camera_label,
            attached=ATTACHED_VIDEO,
        )
        system = SURVEY_SYSTEM.format(schema=scene_json_schema())
        text, usage = await self._generate(clip, system, instruction)
        scene, _notes = repair_scene(extract_json(text))
        return scene, usage

    async def ask(self, clip: Clip, question: str) -> PerceptionAnswer:
        instruction = QUERY_TASK.format(
            question=question,
            camera=clip.camera_label,
            t0=clip.t0_s,
            t1=clip.t1_s,
            attached=ATTACHED_VIDEO,
        )
        system = QUERY_SYSTEM.format(schema=ANSWER_SCHEMA)
        text, usage = await self._generate(clip, system, instruction)
        try:
            payload = extract_json(text)
        except PerceptionError:
            # A small local model that answers a pointed question in plain prose has still done
            # its job, so the sentence is kept. The shared parser then marks the confidence as
            # the unstated default it is, rather than inventing a number.
            payload = {"answer": text.strip()[:600]}
        return answer_from_payload(clip, question, payload, usage=usage)
