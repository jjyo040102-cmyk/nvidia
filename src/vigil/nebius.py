"""Nebius Token Factory client: model discovery, retries, caching, cost accounting.

Why discovery instead of a hardcoded model name: the published NVIDIA model IDs on Token
Factory have changed shape before (``nvidia/nemotron-3-super-120b-a12b`` today), and which
multimodal models an account can actually reach varies by tier. Asking ``GET /v1/models``
at startup turns "the demo 404s on stage" into a logged, visible choice.

Why a disk cache: the trial credit is USD 1. Re-running the demo, re-recording the video and
re-scoring the benchmark all hit identical requests, and paying for them twice is the easiest
way to run out of budget before judging.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from vigil.config import REASONING_PREFERENCE, VISION_PREFERENCE, Settings
from vigil.pricing import estimate_cost

RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504}


class NebiusError(RuntimeError):
    pass


class BudgetExceeded(RuntimeError):
    """Raised before a call that would cross the configured ceiling."""


@dataclass
class ChatResult:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    cached: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def usd(self) -> float:
        return estimate_cost(self.model, self.prompt_tokens, self.completion_tokens)


@dataclass
class Spend:
    """Accumulated across one process so the budget guard has something to compare to."""

    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usd: float = 0.0
    by_model: dict[str, int] = field(default_factory=dict)

    def record(self, result: ChatResult) -> None:
        if result.cached:
            self.cache_hits += 1
        else:
            self.calls += 1
        self.prompt_tokens += result.prompt_tokens
        self.completion_tokens += result.completion_tokens
        self.usd += result.usd
        self.by_model[result.model] = self.by_model.get(result.model, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "live_calls": self.calls,
            "cache_hits": self.cache_hits,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "est_usd": round(self.usd, 6),
            "by_model": self.by_model,
        }

    def __str__(self) -> str:
        return (
            f"{self.calls} live calls, {self.cache_hits} cache hits, "
            f"{self.prompt_tokens}+{self.completion_tokens} tokens, ~${self.usd:.4f}"
        )


def image_part(jpeg_base64: str) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{jpeg_base64}"}}


def text_part(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


class ResponseCache:
    """Content-addressed cache of chat replies. Best-effort: a broken cache never breaks a run."""

    def __init__(self, directory: Path, *, enabled: bool = True) -> None:
        self.dir = Path(directory)
        self.enabled = enabled

    def key(self, model: str, messages: list[dict[str, Any]], extra: dict[str, Any]) -> str:
        blob = json.dumps({"m": model, "msg": messages, "x": extra}, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        path = self.dir / f"{key}.json"
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            return None
        try:
            cached = json.loads(raw)
        except json.JSONDecodeError:
            return None
        # A list or a bare string in a cache file is corruption, not a hit: returning it
        # would surface as an AttributeError deep inside a reply handler.
        return cached if isinstance(cached, dict) else None

    def put(self, key: str, value: dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            tmp = self.dir / f"{key}.json.tmp"
            tmp.write_text(json.dumps(value, default=str), encoding="utf-8")
            tmp.replace(self.dir / f"{key}.json")
        except OSError:
            return


class NebiusClient:
    def __init__(
        self,
        settings: Settings,
        *,
        spend: Spend | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """``transport`` exists so the retry, caching and budget paths can be tested against a
        fake endpoint. Without it every test of the code that spends a $1 credit would need a
        real key, which is precisely the code that must not be discovered broken on stage.
        """
        if not settings.has_nebius_key:
            raise NebiusError(
                "no Nebius API key. Set VIGIL_NEBIUS_API_KEY in .env, or run with "
                "VIGIL_PERCEPTION_BACKEND=mock / VIGIL_REASONING_BACKEND=mock."
            )
        self.settings = settings
        self.spend = spend or Spend()
        self._cache = ResponseCache(settings.cache_dir)
        self._models: list[str] | None = None
        self._headers = {
            "Authorization": (
                f"Bearer {settings.nebius_api_key.get_secret_value()}"  # type: ignore[union-attr]
            ),
            "Content-Type": "application/json",
        }
        self._http = httpx.AsyncClient(
            base_url=settings.nebius_base_url,
            headers=self._headers,
            timeout=settings.request_timeout_s,
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    # ------------------------------------------------------------------ catalog
    async def list_models(self, *, refresh: bool = False) -> list[str]:
        if self._models is not None and not refresh:
            return self._models
        try:
            response = await self._http.get("/models")
            response.raise_for_status()
            data = response.json().get("data", [])
        except httpx.HTTPError as exc:
            raise NebiusError(
                f"could not list models from {self.settings.nebius_base_url}/models: {exc}"
            ) from exc
        self._models = sorted({str(item.get("id")) for item in data if item.get("id")})
        if not self._models:
            raise NebiusError(
                "the account returned an empty model list -- check the key's permissions"
            )
        return self._models

    async def pick(self, kind: str) -> str:
        """Resolve a configured model, or choose the best one the account actually has."""
        configured = (
            self.settings.vision_model if kind == "vision" else self.settings.reasoning_model
        )
        if configured:
            return configured
        preference = VISION_PREFERENCE if kind == "vision" else REASONING_PREFERENCE
        available = await self.list_models()
        lowered = {m: m.lower() for m in available}
        for needle in preference:
            for model, low in lowered.items():
                if needle in low:
                    if (
                        kind == "vision"
                        and self.settings.prefer_nvidia_vision
                        and "nvidia" not in low
                    ):
                        continue
                    return model
        for needle in preference:
            for model, low in lowered.items():
                if needle in low:
                    return model
        env = "VISION" if kind == "vision" else "REASONING"
        raise NebiusError(
            f"no {kind} model matched {preference} among {len(available)} available. "
            f"Set VIGIL_{env}_MODEL explicitly. Available: {available[:12]}"
        )

    # ------------------------------------------------------------------ chat
    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        kind: str = "reasoning",
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> ChatResult:
        chosen = model or await self.pick(kind)
        params: dict[str, Any] = {
            "model": chosen,
            "messages": messages,
            "max_tokens": max_tokens or self.settings.max_output_tokens,
            "temperature": self.settings.temperature if temperature is None else temperature,
        }
        if json_mode:
            params["response_format"] = {"type": "json_object"}

        projected = self.spend.usd + estimate_cost(
            chosen, _rough_prompt_tokens(messages), params["max_tokens"]
        )
        if projected > self.settings.usd_budget:
            raise BudgetExceeded(
                f"this call would take the run to ~${projected:.4f}, over the "
                f"${self.settings.usd_budget:.2f} ceiling. Raise VIGIL_USD_BUDGET "
                "or lower frames_per_clip."
            )

        key = self._cache.key(
            chosen, messages, {k: v for k, v in params.items() if k != "messages"}
        )
        if (hit := self._cache.get(key)) is not None:
            result = ChatResult(
                text=str(hit.get("text", "")),
                model=chosen,
                prompt_tokens=int(hit.get("prompt_tokens", 0)),
                completion_tokens=int(hit.get("completion_tokens", 0)),
                cached=True,
            )
            self.spend.record(result)
            return result

        payload, result = await self._post_with_retry(params, chosen)
        self._cache.put(key, payload)
        self.spend.record(result)
        return result

    async def _post_with_retry(
        self, params: dict[str, Any], model: str
    ) -> tuple[dict[str, Any], ChatResult]:
        attempts = self.settings.max_retries + 1
        last: str = "unknown"
        for attempt in range(attempts):
            started = time.perf_counter()
            try:
                response = await self._http.post("/chat/completions", json=params)
            except httpx.TimeoutException:
                last = "timeout"
            except httpx.HTTPError as exc:
                last = f"transport error: {exc}"
            else:
                if response.status_code == 200:
                    body = _safe_json(response)
                    text = _extract_text(body)
                    usage = body.get("usage") or {}
                    result = ChatResult(
                        text=text,
                        model=str(body.get("model") or model),
                        prompt_tokens=int(
                            usage.get("prompt_tokens") or _rough_prompt_tokens(params["messages"])
                        ),
                        completion_tokens=int(usage.get("completion_tokens") or 0),
                        latency_ms=int((time.perf_counter() - started) * 1000),
                    )
                    payload = {
                        "text": result.text,
                        "prompt_tokens": result.prompt_tokens,
                        "completion_tokens": result.completion_tokens,
                    }
                    return payload, result

                if response.status_code == 400 and "response_format" in params:
                    # Not every deployment honours JSON mode; drop it and try again rather
                    # than failing the run over a nicety.
                    params = {k: v for k, v in params.items() if k != "response_format"}
                    last = "response_format rejected; retried without it"
                    continue
                last = f"HTTP {response.status_code}: {response.text[:280]}"
                if response.status_code not in RETRYABLE:
                    break

            if attempt + 1 < attempts:
                await asyncio.sleep(min(8.0, 0.8 * 2**attempt))
        raise NebiusError(f"chat call to {model} failed after {attempts} attempts -- {last}")


def _safe_json(response: httpx.Response) -> dict[str, Any]:
    try:
        parsed = response.json()
    except (json.JSONDecodeError, ValueError) as exc:
        raise NebiusError(f"non-JSON response: {response.text[:200]!r}") from exc
    return parsed if isinstance(parsed, dict) else {}


def _extract_text(body: dict[str, Any]) -> str:
    choices = body.get("choices") or []
    if not choices:
        raise NebiusError(f"response had no choices: {json.dumps(body)[:240]}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # some gateways return content parts
        return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        return reasoning
    raise NebiusError(f"unrecognised message shape: {json.dumps(message)[:240]}")


def _rough_prompt_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate before the first live call, so the budget guard can act pre-emptively.

    Images dominate a survey request and the API's own tokenizer is not available here, so
    each one is charged at a fixed allowance -- deliberately generous, because under-estimating
    is what turns a $1 credit into a surprise bill.
    """
    total = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            total += len(content) // 4
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    total += len(str(part.get("text", ""))) // 4
                elif part.get("type") == "image_url":
                    total += 1200
    return total


async def probe(settings: Settings) -> dict[str, Any]:
    """What ``vigil probe`` shows: reachability, models, and the two picks we would use."""
    client = NebiusClient(settings)
    try:
        models = await client.list_models()
        out: dict[str, Any] = {
            "base_url": settings.nebius_base_url,
            "key": settings.key_fingerprint,
            "model_count": len(models),
            "models": models,
        }
        for kind in ("reasoning", "vision"):
            try:
                out[f"picked_{kind}"] = await client.pick(kind)
            except NebiusError as exc:
                out[f"picked_{kind}"] = f"UNRESOLVED: {exc}"
        return out
    finally:
        await client.aclose()
