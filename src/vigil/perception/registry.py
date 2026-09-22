"""Backend selection. One place decides who reads the pixels."""

from __future__ import annotations

from collections.abc import Callable

from vigil.config import Settings
from vigil.nebius import NebiusClient
from vigil.perception.base import PerceptionBackend, PerceptionError
from vigil.perception.cosmos import CosmosPerception
from vigil.perception.mock import MockPerception
from vigil.perception.nebius import NebiusPerception

_BUILDERS: dict[str, Callable[[Settings, NebiusClient | None], PerceptionBackend]] = {
    "mock": lambda s, _client: MockPerception(s),
    "nebius": lambda s, c: NebiusPerception(s, c),
    "cosmos": lambda s, _client: CosmosPerception(s),
}


def build_perception(settings: Settings, client: NebiusClient | None = None) -> PerceptionBackend:
    name = settings.resolved_perception
    builder = _BUILDERS.get(name)
    if builder is None:
        raise PerceptionError(
            f"unknown perception backend {name!r}; choose one of {sorted(_BUILDERS)}"
        )
    return builder(settings, client)


__all__ = ["PerceptionError", "build_perception"]
