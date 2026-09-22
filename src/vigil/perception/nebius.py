"""Vision over the Nebius Token Factory OpenAI-compatible endpoint.

Frames are sent as base64 JPEGs rather than a video part: a video content type is not
documented for this API, and frame-as-image is what the hosted vision models are actually
exercised with. The trade-off is temporal aliasing -- six frames across a six-second clip
will miss a fast slip -- which is exactly what the ``ask`` follow-up exists to recover from:
the agent re-samples a narrower window at higher density instead of accepting the first pass.
"""

from __future__ import annotations

import asyncio
import functools
from typing import Any

from vigil.config import Settings
from vigil.models.scene import SceneUnderstanding
from vigil.nebius import NebiusClient, NebiusError, image_part, text_part
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
    ATTACHED_FRAMES,
    QUERY_SYSTEM,
    QUERY_TASK,
    SURVEY_SYSTEM,
    SURVEY_TASK,
)
from vigil.video.sampler import Frame, MediaError, sample_frames


class NebiusPerception(PerceptionBackend):
    name = "nebius"
    supports_video = True

    def __init__(self, settings: Settings, client: NebiusClient | None = None) -> None:
        self.settings = settings
        self._own_client = client is None
        self.client = client or NebiusClient(settings)
        self._model: str | None = None

    async def model(self) -> str:
        if self._model is None:
            try:
                self._model = await self.client.pick("vision")
            except NebiusError as exc:
                raise PerceptionError(str(exc)) from exc
        return self._model

    async def model_id(self) -> str:
        if self._model is None:
            # A run that never surveyed has no model to name. Reporting that beats risking a
            # catalog call here, because this runs while building the report: a failure now
            # would throw away an investigation that already spent its credit.
            return "not reached"
        return self._model

    def _frames(self, clip: Clip, *, density: int | None = None) -> tuple[list[str], list[Frame]]:
        n = density or self.settings.frames_per_clip
        frames = sample_frames(
            clip.path,
            t0=clip.t0_s,
            t1=clip.t1_s,
            count=n,
            max_px=self.settings.frame_max_px,
        )
        if not frames:
            raise PerceptionError(
                f"no decodable frames in {clip.path} between {clip.t0_s}-{clip.t1_s}s"
            )
        return [f.to_jpeg_base64() for f in frames], frames

    async def survey(self, clip: Clip) -> tuple[SceneUnderstanding, Usage]:
        try:
            b64, frames = await asyncio.to_thread(self._frames, clip)
        except MediaError as exc:
            raise PerceptionError(str(exc)) from exc

        content: list[dict[str, Any]] = [
            text_part(
                SURVEY_TASK.format(
                    clip_id=clip.id,
                    t0=clip.t0_s,
                    t1=clip.t1_s,
                    camera=clip.camera_label,
                    attached=ATTACHED_FRAMES.format(
                        n=len(b64), times=", ".join(f"{f.t_s:.1f}s" for f in frames)
                    ),
                )
            )
        ]
        content += [image_part(b) for b in b64]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SURVEY_SYSTEM.format(schema=scene_json_schema())},
            {"role": "user", "content": content},
        ]

        try:
            result = await self.client.chat(
                messages, model=await self.model(), kind="vision", json_mode=True
            )
        except NebiusError as exc:
            raise PerceptionError(f"survey call failed: {exc}") from exc

        scene, _notes = repair_scene(extract_json(result.text))
        usage = Usage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            latency_ms=result.latency_ms,
            model=f"nebius/{result.model}",
        )
        return scene, usage

    async def ask(self, clip: Clip, question: str) -> PerceptionAnswer:
        try:
            # follow-ups earn the dense sampling: the agent is asking because the first pass
            # was not good enough, so give it more temporal resolution on the narrow window.
            dense = min(16, self.settings.frames_per_clip * 2)
            # partial, not a second positional: ``density`` is keyword-only, and a TypeError
            # here would only ever surface on a machine with a real key attached.
            b64, frames = await asyncio.to_thread(
                functools.partial(self._frames, clip, density=dense)
            )
        except MediaError as exc:
            raise PerceptionError(str(exc)) from exc

        content: list[dict[str, Any]] = [
            text_part(
                QUERY_TASK.format(
                    question=question,
                    camera=clip.camera_label,
                    t0=clip.t0_s,
                    t1=clip.t1_s,
                    attached=ATTACHED_FRAMES.format(
                        n=len(b64), times=", ".join(f"{f.t_s:.1f}s" for f in frames)
                    ),
                )
            )
        ]
        content += [image_part(b) for b in b64]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": QUERY_SYSTEM.format(schema=ANSWER_SCHEMA)},
            {"role": "user", "content": content},
        ]
        try:
            result = await self.client.chat(
                messages, model=await self.model(), kind="vision", json_mode=True, max_tokens=420
            )
        except NebiusError as exc:
            raise PerceptionError(f"ask call failed: {exc}") from exc

        return answer_from_payload(
            clip,
            question,
            extract_json(result.text),
            usage=Usage(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                latency_ms=result.latency_ms,
                model=f"nebius/{result.model}",
            ),
        )

    async def aclose(self) -> None:
        if self._own_client:
            await self.client.aclose()
