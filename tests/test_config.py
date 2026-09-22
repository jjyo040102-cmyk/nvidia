"""Configuration: the file that decides whether the app boots with or without credentials."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from vigil import config as config_module
from vigil.config import Settings
from vigil.pricing import estimate_cost, rate_for


def test_boots_with_no_credentials_at_all(tmp_settings: Settings) -> None:
    assert tmp_settings.has_nebius_key is False
    assert tmp_settings.resolved_perception == "mock"
    assert tmp_settings.resolved_reasoning == "mock"


def test_a_key_is_the_only_condition_that_switches_backend(
    settings_for: Callable[..., Settings],
) -> None:
    with_key = settings_for(nebius_api_key="nvidia-key-123456")
    assert with_key.has_nebius_key is True
    assert with_key.resolved_perception == "nebius"
    # An explicit choice is never second-guessed: silently downgrading mid-demo is worse
    # than a loud failure at model-resolution time.
    assert settings_for(perception_backend="cosmos").resolved_perception == "cosmos"


def test_blank_key_does_not_count_as_present(settings_for: Callable[..., Settings]) -> None:
    settings = settings_for(nebius_api_key="   ")
    assert settings.has_nebius_key is False
    assert settings.key_fingerprint == "<none>"


def test_base_url_loses_its_trailing_slash() -> None:
    settings = Settings(_env_file=None, nebius_base_url="https://example.test/v1/")
    assert settings.nebius_base_url == "https://example.test/v1"


def test_fingerprint_proves_which_key_ran_without_leaking_it(
    settings_for: Callable[..., Settings],
) -> None:
    secret = "abcdefghij1234567890"
    settings = settings_for(nebius_api_key=secret)
    printed = settings.key_fingerprint
    assert secret not in printed
    assert printed == "abcd…7890"


def test_a_short_key_only_reports_that_it_is_set(settings_for: Callable[..., Settings]) -> None:
    settings = settings_for(nebius_api_key="shorty")
    assert settings.key_fingerprint == "set"


def test_env_file_on_disk_is_ignored_by_the_test_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIGIL_MAX_ITERATIONS", "3")
    assert Settings(_env_file=None).max_iterations == 3  # real env still applies
    assert Settings(_env_file=None, max_iterations=9).max_iterations == 9  # and loses to init


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("clip_overlap", 1.0),
        ("clip_overlap", -0.01),
        ("clip_seconds", 0.0),
        ("frames_per_clip", 1),
        ("token_budget", 999),
        ("usd_budget", 0.0),
        ("perception_backend", "openai"),
        ("max_retries", 9),
    ],
)
def test_runaway_and_broken_knobs_are_rejected_at_the_boundary(field: str, value: object) -> None:
    kwargs: dict[str, Any] = {field: value}
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **kwargs)


def test_settings_cache_resets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    config_module.reset_settings_cache()
    first = config_module.get_settings()
    assert config_module.get_settings() is first
    config_module.reset_settings_cache()
    assert config_module.get_settings() is not first


def test_unknown_model_overestimates_rather_than_underestimates() -> None:
    # The point of the fallback rate: an unrecognised model must never look free.
    assert rate_for("some/new-model") == (1.00, 3.00)
    assert estimate_cost("x", 1_000_000, 0) == pytest.approx(1.0)
    assert estimate_cost("nvidia/nemotron-3-super", 1_000_000, 1_000_000) == pytest.approx(1.20)


def test_price_overrides_are_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIGIL_MODEL_PRICES", '{"cheap": [0.01, 0.02], "broken": ["x", 1]}')
    assert rate_for("vendor/cheap-model") == (0.01, 0.02)
    # A malformed entry is skipped, not fatal: the whole product must not hinge on a typo
    # in an optional override.
    assert rate_for("vendor/broken-model") == (1.00, 3.00)
    monkeypatch.delenv("VIGIL_MODEL_PRICES")
    assert rate_for("nvidia/nemotron-3-super") == (0.30, 0.90)


def test_negative_tokens_cannot_produce_negative_spend() -> None:
    assert estimate_cost("nemotron", -500, -500) == 0.0
