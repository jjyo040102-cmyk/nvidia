"""Rough USD estimates, so a $1 trial credit cannot be silently exhausted.

These are the publicly listed rates as gathered in September 2026 and are NOT authoritative
-- Nebius gates real pricing behind the console. Override what you actually pay with
``VIGIL_MODEL_PRICES='{"nvidia/x": [in_per_mtok, out_per_mtok]}'``. Every number Vigil
prints as cost is labelled *estimated* for this reason.
"""

from __future__ import annotations

import json
import os

# model substring -> (USD per 1M prompt tokens, USD per 1M completion tokens)
RATES: dict[str, tuple[float, float]] = {
    "nemotron-3-ultra": (1.00, 3.00),
    "nemotron-3-super": (0.30, 0.90),
    "nemotron-3-nano-omni": (0.06, 0.24),
    "nemotron-3-nano": (0.05, 0.20),
    "nemotron": (0.10, 0.30),
    "gpt-oss-120b": (0.15, 0.60),
    "gpt-oss-20b": (0.05, 0.20),
    "qwen2.5-vl": (0.20, 0.60),
    "qwen3-vl": (0.20, 0.60),
    "deepseek": (0.30, 0.90),
}

# Charged when nothing matches: deliberately the priciest entry, so an unknown model
# over-estimates spend instead of under-estimating it.
FALLBACK: tuple[float, float] = (1.00, 3.00)


def _overrides() -> dict[str, tuple[float, float]]:
    raw = os.environ.get("VIGIL_MODEL_PRICES", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    out: dict[str, tuple[float, float]] = {}
    for key, value in parsed.items() if isinstance(parsed, dict) else []:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            try:
                out[str(key).lower()] = (float(value[0]), float(value[1]))
            except (TypeError, ValueError):
                continue
    return out


def rate_for(model: str) -> tuple[float, float]:
    low = (model or "").lower()
    for key, value in _overrides().items():
        if key in low:
            return value
    for key, value in RATES.items():
        if key in low:
            return value
    return FALLBACK


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    in_rate, out_rate = rate_for(model)
    return (max(0, prompt_tokens) * in_rate + max(0, completion_tokens) * out_rate) / 1_000_000
