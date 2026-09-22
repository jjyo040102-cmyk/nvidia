"""The client that spends the trial credit, tested against an endpoint that is not real.

Everything here would otherwise need a live key: model discovery, retries, the content-addressed
cache and the USD ceiling. Those four are the difference between a demo that survives a re-run and
one that burns a stranger's credit, so they are exercised against an in-process transport instead
of being read optimistically on stage.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from vigil.config import Settings
from vigil.nebius import (
    BudgetExceeded,
    ChatResult,
    NebiusClient,
    NebiusError,
    ResponseCache,
    Spend,
    _rough_prompt_tokens,
    probe,
)

from .helpers import CATALOG, client_with, completion

EXPECTED_CATALOG = [
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "nvidia/nemotron-3-nano-omni-30b-a3b",
    "nvidia/nemotron-3-super-120b-a12b",
]


@pytest.fixture
def calls() -> list[str]:
    return []


# ----------------------------------------------------------------------- construction


async def test_no_key_means_no_client_and_the_message_names_the_way_out(
    tmp_settings: Settings,
) -> None:
    with pytest.raises(NebiusError, match="VIGIL_PERCEPTION_BACKEND=mock"):
        NebiusClient(tmp_settings)


def test_the_key_reaches_the_header_and_nowhere_else(
    settings_for: Callable[..., Settings], calls: list[str]
) -> None:
    secret = "super-secret-value-1234"
    client = client_with(
        settings_for(nebius_api_key=secret), lambda r: httpx.Response(200, json=CATALOG), calls
    )
    assert client._headers["Authorization"] == f"Bearer {secret}"
    key = client._cache.key("m", [{"role": "user", "content": "x"}], {})
    assert secret not in key


# ----------------------------------------------------------------------- discovery


async def test_models_come_back_sorted_deduplicated_and_only_fetched_once(
    settings_for: Callable[..., Settings], calls: list[str]
) -> None:
    client = client_with(
        settings_for(nebius_api_key="k"), lambda r: httpx.Response(200, json=CATALOG), calls
    )
    first = await client.list_models()
    assert await client.list_models() == first
    assert first == EXPECTED_CATALOG
    assert calls == ["GET /v1/models"]
    await client.aclose()


async def test_refresh_actually_refetches(
    settings_for: Callable[..., Settings], calls: list[str]
) -> None:
    client = client_with(
        settings_for(nebius_api_key="k"), lambda r: httpx.Response(200, json=CATALOG), calls
    )
    await client.list_models()
    await client.list_models(refresh=True)
    assert calls == ["GET /v1/models", "GET /v1/models"]
    await client.aclose()


async def test_an_empty_catalog_is_a_permissions_problem_not_a_blank_pick(
    settings_for: Callable[..., Settings], calls: list[str]
) -> None:
    client = client_with(
        settings_for(nebius_api_key="k"),
        lambda r: httpx.Response(200, json={"data": []}),
        calls,
    )
    with pytest.raises(NebiusError, match="permissions"):
        await client.list_models()
    await client.aclose()


async def test_an_unreachable_catalog_says_which_url_it_tried(
    settings_for: Callable[..., Settings], calls: list[str]
) -> None:
    client = client_with(
        settings_for(nebius_api_key="k", nebius_base_url="https://api.test/v1"),
        lambda r: httpx.Response(503, text="nope"),
        calls,
    )
    with pytest.raises(NebiusError, match=r"https://api\.test/v1/models"):
        await client.list_models()
    await client.aclose()


# ----------------------------------------------------------------------- picking


async def test_a_configured_model_is_used_without_probing_the_account(
    settings_for: Callable[..., Settings], calls: list[str]
) -> None:
    client = client_with(
        settings_for(nebius_api_key="k", reasoning_model="mine/particular"),
        lambda r: httpx.Response(200, json=CATALOG),
        calls,
    )
    assert await client.pick("reasoning") == "mine/particular"
    assert calls == []
    await client.aclose()


async def test_preference_order_decides_which_model_the_account_gets(
    settings_for: Callable[..., Settings], calls: list[str]
) -> None:
    client = client_with(
        settings_for(nebius_api_key="k"), lambda r: httpx.Response(200, json=CATALOG), calls
    )
    assert await client.pick("reasoning") == "nvidia/nemotron-3-super-120b-a12b"
    assert await client.pick("vision") == "nvidia/nemotron-3-nano-omni-30b-a3b"
    await client.aclose()


async def test_the_nvidia_preference_is_a_preference_not_a_dead_end(
    settings_for: Callable[..., Settings], calls: list[str]
) -> None:
    """An account with no NVIDIA vision model still has to produce *a* vision model."""
    catalog = {"data": [{"id": "some/qwen2.5-vl-7b"}, {"id": "nvidia/llava-onevision-nd"}]}
    client = client_with(
        settings_for(nebius_api_key="k", prefer_nvidia_vision=True),
        lambda r: httpx.Response(200, json=catalog),
        calls,
    )
    # Both match "qwen2.5-vl"; the NVIDIA one wins the first pass.
    assert await client.pick("vision") == "nvidia/llava-onevision-nd"
    await client.aclose()


async def test_an_unmatched_preference_lists_what_was_available(
    settings_for: Callable[..., Settings], calls: list[str]
) -> None:
    client = client_with(
        settings_for(nebius_api_key="k"),
        lambda r: httpx.Response(200, json={"data": [{"id": "someone/else"}]}),
        calls,
    )
    with pytest.raises(NebiusError, match=r"Set VIGIL_VISION_MODEL explicitly.*someone/else"):
        await client.pick("vision")
    await client.aclose()


# ----------------------------------------------------------------------- chat


async def test_a_chat_call_sends_the_params_and_accounts_for_the_tokens(
    settings_for: Callable[..., Settings], calls: list[str]
) -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content.decode()))
        return completion("hello", prompt_tokens=11, completion_tokens=7)

    spend = Spend()
    client = client_with(
        settings_for(nebius_api_key="k", reasoning_model="m"),
        handler,
        calls,
        spend=spend,
    )
    result = await client.chat([{"role": "user", "content": "say hi"}])
    assert result.text == "hello"
    assert (result.prompt_tokens, result.completion_tokens) == (11, 7)
    assert result.total_tokens == 18
    assert result.usd > 0.0
    assert seen["model"] == "m"
    assert seen["temperature"] == 0.2
    assert seen["max_tokens"] == 1200
    assert "response_format" not in seen
    assert spend.calls == 1
    assert spend.usd == pytest.approx(result.usd)
    await client.aclose()


async def test_json_mode_asked_for_once_is_dropped_if_the_endpoint_refuses_it(
    settings_for: Callable[..., Settings], calls: list[str]
) -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        bodies.append(body)
        if "response_format" in body:
            return httpx.Response(400, text="unknown field response_format")
        return completion('{"ok": true}')

    client = client_with(
        settings_for(nebius_api_key="k", reasoning_model="m", max_retries=1),
        handler,
        calls,
    )
    result = await client.chat([{"role": "user", "content": "x"}], json_mode=True)
    assert result.text == '{"ok": true}'
    assert "response_format" in bodies[0]
    assert "response_format" not in bodies[1]
    await client.aclose()


async def test_a_transient_failure_is_retried(
    settings_for: Callable[..., Settings], calls: list[str]
) -> None:
    attempts = {"n": 0}

    def flaky(_request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(503, text="busy")
        return completion("recovered")

    client = client_with(
        settings_for(nebius_api_key="k", reasoning_model="m", max_retries=2), flaky, calls
    )
    assert (await client.chat([{"role": "user", "content": "x"}])).text == "recovered"
    assert len(calls) == 2
    await client.aclose()


async def test_a_permanent_failure_is_not_retried_into_silence(
    settings_for: Callable[..., Settings], calls: list[str]
) -> None:
    client = client_with(
        settings_for(nebius_api_key="k", reasoning_model="m", max_retries=4),
        lambda r: httpx.Response(404, text="no such model"),
        calls,
    )
    with pytest.raises(NebiusError, match="HTTP 404"):
        await client.chat([{"role": "user", "content": "x"}])
    assert len(calls) == 1
    await client.aclose()


async def test_a_transport_failure_reaches_the_caller_as_one_named_error(
    settings_for: Callable[..., Settings], calls: list[str]
) -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = client_with(
        settings_for(nebius_api_key="k", reasoning_model="m", max_retries=0), boom, calls
    )
    with pytest.raises(NebiusError, match="failed after 1 attempts"):
        await client.chat([{"role": "user", "content": "x"}])
    await client.aclose()


async def test_a_reply_without_usage_still_costs_something(
    settings_for: Callable[..., Settings], calls: list[str]
) -> None:
    client = client_with(
        settings_for(nebius_api_key="k", reasoning_model="m"),
        lambda r: completion("a" * 400),
        calls,
    )
    messages = [{"role": "user", "content": "x"}]
    result = await client.chat(messages)
    assert result.prompt_tokens == _rough_prompt_tokens(messages)
    assert result.completion_tokens == 0
    await client.aclose()


@pytest.mark.parametrize(
    ("body", "expected", "refused"),
    [
        ({"choices": []}, "", "no choices"),
        (
            {"choices": [{"message": {"content": [{"text": "part one "}, {"text": "part two"}]}}]},
            "part one part two",
            "",
        ),
        ({"choices": [{"message": {"reasoning_content": "only thinking"}}]}, "only thinking", ""),
        ({"choices": [{"message": {"content": 5}}]}, "", "unrecognised message shape"),
        ({"model": 1, "choices": [{"message": {}}]}, "", "unrecognised message shape"),
    ],
)
async def test_message_shapes_the_gateways_actually_return(
    settings_for: Callable[..., Settings],
    calls: list[str],
    body: dict[str, Any],
    expected: str,
    refused: str,
) -> None:
    client = client_with(
        settings_for(nebius_api_key="k", reasoning_model="m"),
        lambda r: httpx.Response(200, json=body),
        calls,
    )
    if refused:
        with pytest.raises(NebiusError, match=refused):
            await client.chat([{"role": "user", "content": "x"}])
    else:
        assert (await client.chat([{"role": "user", "content": "x"}])).text == expected
    await client.aclose()


async def test_a_200_that_is_not_json_is_called_what_it_is(
    settings_for: Callable[..., Settings], calls: list[str]
) -> None:
    client = client_with(
        settings_for(nebius_api_key="k", reasoning_model="m"),
        lambda r: httpx.Response(200, text="<html>proxy error</html>"),
        calls,
    )
    with pytest.raises(NebiusError, match="non-JSON response"):
        await client.chat([{"role": "user", "content": "x"}])
    await client.aclose()


async def test_a_json_array_body_is_not_mistaken_for_a_reply(
    settings_for: Callable[..., Settings], calls: list[str]
) -> None:
    client = client_with(
        settings_for(nebius_api_key="k", reasoning_model="m"),
        lambda r: httpx.Response(200, json=[1, 2]),
        calls,
    )
    with pytest.raises(NebiusError, match="no choices"):
        await client.chat([{"role": "user", "content": "x"}])
    await client.aclose()


# ----------------------------------------------------------------------- the ceiling


async def test_a_call_that_would_cross_the_ceiling_is_refused_before_it_is_sent(
    settings_for: Callable[..., Settings], calls: list[str]
) -> None:
    client = client_with(
        settings_for(nebius_api_key="k", reasoning_model="m", usd_budget=0.0001),
        lambda r: completion("x"),
        calls,
    )
    with pytest.raises(BudgetExceeded, match="VIGIL_USD_BUDGET"):
        await client.chat([{"role": "user", "content": "y" * 4000}])
    assert calls == [], "the guard has to fire before the request leaves the process"
    await client.aclose()


def test_an_image_is_charged_enough_that_the_guard_cannot_be_fooled() -> None:
    text_only = _rough_prompt_tokens([{"role": "user", "content": "look"}])
    with_image = _rough_prompt_tokens(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}},
                ],
            }
        ]
    )
    assert with_image - text_only >= 1200
    assert _rough_prompt_tokens([{"role": "user", "content": None}]) == 0


# ----------------------------------------------------------------------- caching


async def test_an_identical_call_the_second_time_costs_nothing_and_says_so(
    tmp_path: Path, settings_for: Callable[..., Settings], calls: list[str]
) -> None:
    spend = Spend()
    client = client_with(
        settings_for(nebius_api_key="k", reasoning_model="m", cache_dir=tmp_path / "cache"),
        lambda r: completion("cached?", prompt_tokens=3, completion_tokens=4),
        calls,
        spend=spend,
    )
    messages = [{"role": "user", "content": "same question"}]
    first = await client.chat(messages)
    second = await client.chat(messages)
    assert first.cached is False
    assert second.cached is True
    assert second.total_tokens == 7
    assert calls == ["POST /v1/chat/completions"]
    assert (spend.calls, spend.cache_hits) == (1, 1)
    assert (spend.prompt_tokens, spend.completion_tokens) == (6, 8)
    assert spend.by_model == {"m": 2}
    await client.aclose()


async def test_a_different_question_model_or_parameter_is_not_a_hit(
    tmp_path: Path, settings_for: Callable[..., Settings], calls: list[str]
) -> None:
    client = client_with(
        settings_for(nebius_api_key="k", reasoning_model="m", cache_dir=tmp_path / "cache"),
        lambda r: completion("x"),
        calls,
    )
    cache = client._cache
    messages = [{"role": "user", "content": "one"}]
    base = cache.key("m", messages, {"t": 0.2})
    assert cache.key("m", [{"role": "user", "content": "two"}], {"t": 0.2}) != base
    assert cache.key("other", messages, {"t": 0.2}) != base
    assert cache.key("m", messages, {"t": 0.5}) != base
    assert cache.key("m", messages, {"t": 0.2}) == base
    await client.aclose()


def test_a_corrupt_cache_file_is_a_miss_not_a_crash(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache")
    cache.put("good", {"text": "hi"})
    assert cache.get("good") == {"text": "hi"}

    (tmp_path / "cache" / "broken.json").write_text("{not json", encoding="utf-8")
    assert cache.get("broken") is None
    (tmp_path / "cache" / "listy.json").write_text("[1, 2]", encoding="utf-8")
    assert cache.get("listy") is None
    assert cache.get("absent") is None

    off = ResponseCache(tmp_path / "cache", enabled=False)
    off.put("never", {"text": "x"})
    assert off.get("never") is None
    assert not (tmp_path / "cache" / "never.json").exists()


def test_a_cache_directory_that_cannot_be_written_does_not_break_the_run(
    tmp_path: Path,
) -> None:
    blocker = tmp_path / "blocked"
    blocker.write_text("a file where a directory should be", encoding="utf-8")
    cache = ResponseCache(blocker / "cache")
    cache.put("k", {"text": "x"})
    assert cache.get("k") is None


def test_spend_separates_live_calls_from_hits() -> None:
    spend = Spend()
    spend.record(ChatResult(text="a", model="m", prompt_tokens=10, completion_tokens=5))
    spend.record(
        ChatResult(text="a", model="m", prompt_tokens=10, completion_tokens=5, cached=True)
    )
    assert "1 live calls, 1 cache hits" in str(spend)
    assert spend.by_model == {"m": 2}
    assert spend.as_dict()["live_calls"] == 1


# ----------------------------------------------------------------------- probe


async def test_probe_reports_the_key_without_leaving_it(
    settings_for: Callable[..., Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "probe-secret-key-123456"
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=CATALOG))
    monkeypatch.setattr(
        "vigil.nebius.NebiusClient",
        lambda settings: NebiusClient(settings, transport=transport),
    )
    out = await probe(settings_for(nebius_api_key=secret))
    assert out["key"] == "prob…3456"
    assert secret not in json.dumps(out)
    assert out["model_count"] == 3
    assert out["picked_reasoning"] == "nvidia/nemotron-3-super-120b-a12b"
    assert out["picked_vision"] == "nvidia/nemotron-3-nano-omni-30b-a3b"


async def test_probe_survives_a_catalog_it_cannot_serve(
    settings_for: Callable[..., Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"data": [{"id": "x/y"}]}))
    monkeypatch.setattr(
        "vigil.nebius.NebiusClient",
        lambda settings: NebiusClient(settings, transport=transport),
    )
    out = await probe(settings_for(nebius_api_key="k"))
    assert str(out["picked_vision"]).startswith("UNRESOLVED")
    assert "Set VIGIL_VISION_MODEL" in str(out["picked_vision"])

async def test_default_reasoning_never_falls_back_to_a_non_nvidia_model(
    settings_for: Callable[..., Settings], calls: list[str]
) -> None:
    """The hackathon requires an NVIDIA open model; default LIVE must preserve that guarantee."""
    catalog = {"data": [{"id": "openai/gpt-oss-120b"}, {"id": "deepseek/deepseek-v3"}]}
    client = client_with(
        settings_for(nebius_api_key="k"), lambda r: httpx.Response(200, json=catalog), calls
    )
    with pytest.raises(NebiusError, match="VIGIL_REASONING_MODEL"):
        await client.pick("reasoning")
    await client.aclose()
