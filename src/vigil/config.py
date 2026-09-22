"""Runtime configuration.

Two rules shaped this file. First, the app must boot and pass its test suite with no
credentials at all, so every model backend defaults to ``auto`` and falls back to the
deterministic mock rather than raising. Second, the entrant runs on free Nebius credit,
so there is a hard token/cost ceiling per run instead of an unbounded loop.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Backend = Literal["auto", "mock", "nebius", "cosmos"]

NEBIUS_TOKEN_FACTORY_BASE = "https://api.tokenfactory.nebius.com/v1"

# Preference order, most preferred first. Matched case-insensitively as substrings against
# whatever GET /v1/models actually returns, because the published IDs have drifted before
# and an unmatched default is a silent, confusing failure at demo time.
REASONING_PREFERENCE: tuple[str, ...] = (
    # Default LIVE mode is a hackathon path: it must not silently become ineligible by
    # falling back to a non-NVIDIA reasoner when Nemotron is unavailable.  A user may
    # still configure an explicit model, but automatic selection is intentionally strict.
    "nemotron-3-super",
    "nemotron-3-nano",
    "nemotron",
)
VISION_PREFERENCE: tuple[str, ...] = (
    "nemotron-3-nano-omni",
    "cosmos",
    "qwen2.5-vl",
    "qwen3-vl",
    "qwen2-vl",
    "minicpm-v",
    "llava",
    "pixtral",
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VIGIL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------------------------------------------------------------- backends
    perception_backend: Backend = Field(default="auto", description="Who reads the pixels.")
    reasoning_backend: Backend = Field(default="auto", description="Who runs the agent loop.")
    mock_latency_ms: int = Field(
        default=0, ge=0, description="Artificial delay for mock runs; 0 in tests."
    )

    # ---------------------------------------------------------------- nebius
    nebius_api_key: SecretStr | None = None
    nebius_base_url: str = NEBIUS_TOKEN_FACTORY_BASE
    # Leave empty to auto-pick from GET /v1/models. Nemotron-3 Super and Ultra are
    # text-only, so the vision slot needs an omni/video-capable model or a hosted VLM.
    reasoning_model: str = ""
    vision_model: str = ""
    request_timeout_s: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=5)
    prefer_nvidia_vision: bool = Field(
        default=True,
        description="Prefer an NVIDIA open vision model when the account exposes one.",
    )

    # ---------------------------------------------------------------- cosmos
    cosmos_model_id: str = Field(
        default="nvidia/Cosmos-Reason2-2B",
        description="A gated licence on huggingface.co must be accepted before it downloads.",
    )
    cosmos_device: str = Field(
        default="cuda",
        description="Where the weights go. 'auto' lets accelerate split them across devices.",
    )
    cosmos_quantize: Literal["none", "4bit", "8bit"] = Field(
        default="4bit",
        description="Needs bitsandbytes. 'none' only fits on a card with ~24 GB free.",
    )

    # ---------------------------------------------------------------- agent
    max_iterations: int = Field(default=8, ge=1, le=24)
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    max_output_tokens: int = Field(default=1200, ge=128, le=8192)
    # A $1 trial credit is the realistic budget, so ceilings are deliberately small and
    # every run prints what it spent.
    token_budget: int = Field(
        default=60_000, ge=1000, description="Hard ceiling for one investigation."
    )
    usd_budget: float = Field(
        default=0.10, gt=0, description="Refuse to spend beyond this in one run."
    )

    # ---------------------------------------------------------------- video
    clip_seconds: float = Field(default=4.0, gt=0, le=60)
    clip_overlap: float = Field(
        default=0.5,
        ge=0.0,
        lt=1.0,
        description="How much the opening scan overlaps. 0 means lead time depends on where the "
        "window edges fall; 0.5 means any hazard with warning available gets it counted.",
    )
    frames_per_clip: int = Field(
        default=6, ge=2, le=32, description="Fewer frames, cheaper survey."
    )
    max_clips_per_video: int = Field(default=8, ge=1, le=200)
    frame_max_px: int = Field(default=512, ge=128, le=1280)
    cache_dir: Path = Field(
        default=Path("artifacts/cache"), description="Model responses, keyed by request."
    )

    # ---------------------------------------------------------------- paths
    data_dir: Path = Path("data")
    artifact_dir: Path = Path("artifacts")
    policy_dir: Path = Field(
        default=Path("data/policy"), description="JSON rule corpus for retrieval."
    )
    clip_dir: Path = Field(default=Path("data/clips"), description="Sample footage.")

    site_name: str = "Northgate Logistics -- Bay 3"
    locale: Literal["en", "ko"] = "en"

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @field_validator("nebius_base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @field_validator("data_dir", "artifact_dir", "policy_dir", "clip_dir")
    @classmethod
    def _expand(cls, v: Path) -> Path:
        return v.expanduser()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_nebius_key(self) -> bool:
        return bool(self.nebius_api_key and self.nebius_api_key.get_secret_value().strip())

    def resolve(self, requested: Backend) -> str:
        """Map ``auto`` onto a real backend without ever raising.

        A key is the only condition: when the account turns out to expose no vision model,
        ``NebiusClient.pick`` fails with the model list in the message, which is a far more
        useful error than silently downgrading to scripted data mid-demo.
        """
        if requested != "auto":
            return requested
        return "nebius" if self.has_nebius_key else "mock"

    @property
    def resolved_perception(self) -> str:
        return self.resolve(self.perception_backend)

    @property
    def resolved_reasoning(self) -> str:
        return self.resolve(self.reasoning_backend)

    @property
    def key_fingerprint(self) -> str:
        """Safe to log: proves which credential was used without leaking it."""
        if not self.has_nebius_key:
            return "<none>"
        value = self.nebius_api_key.get_secret_value() if self.nebius_api_key else ""
        return f"{value[:4]}…{value[-4:]}" if len(value) > 12 else "set"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
