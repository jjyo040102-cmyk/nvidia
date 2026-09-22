"""Backend selection: the one switch that decides who reads the pixels.

A dozen lines of dict lookup, and it is the line the demo stands on. ``auto`` has to land on
mock when there is no key and on Token Factory when there is one, and a name with no builder has
to say which names do exist rather than raising a KeyError from inside an investigation.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from vigil.config import Settings
from vigil.perception.base import PerceptionBackend
from vigil.perception.cosmos import CosmosPerception
from vigil.perception.mock import MockPerception
from vigil.perception.nebius import NebiusPerception
from vigil.perception.registry import PerceptionError, build_perception


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(None, MockPerception), ("mock", MockPerception), ("cosmos", CosmosPerception)],
)
def test_selection_follows_the_setting_rather_than_what_happens_to_be_installed(
    settings_for: Callable[..., Settings],
    requested: str | None,
    expected: type[PerceptionBackend],
) -> None:
    overrides: dict[str, object] = {} if requested is None else {"perception_backend": requested}
    built = build_perception(settings_for(**overrides))
    assert type(built) is expected
    assert built.supports_video is (expected is CosmosPerception)


async def test_a_key_is_the_only_thing_auto_needs_to_reach_for_the_hosted_backend(
    settings_for: Callable[..., Settings],
) -> None:
    settings = settings_for(nebius_api_key="k", vision_model="some/vision-model")
    built = build_perception(settings)
    assert type(built) is NebiusPerception
    # No downgrade to mock behind the user's back: a run that quietly ran on scripted data
    # would be the worst thing this product could do on stage.
    assert settings.resolved_perception == "nebius"
    await built.aclose()


def test_a_name_with_no_builder_names_the_ones_that_do(
    settings_for: Callable[..., Settings], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reachable only if a ``Backend`` member gains no builder -- the half-wired state this guards.

    ``Settings`` rejects a name outside the Literal before this code runs, so the property is
    turned off here rather than pretending a typo reaches this far.
    """
    settings = settings_for(perception_backend="cosmos")
    monkeypatch.setattr(Settings, "resolved_perception", property(lambda self: "clipo"))
    with pytest.raises(PerceptionError, match=r"choose one of \['cosmos', 'mock', 'nebius'\]"):
        build_perception(settings)
