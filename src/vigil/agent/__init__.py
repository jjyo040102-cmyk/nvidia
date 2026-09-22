"""The investigation layer: a reasoner that chooses, a toolkit that executes, a loop that accounts.

Re-exports the entry points only. ``vigil.agent.actions`` and ``vigil.agent.prompts`` are
importable on their own and deliberately not surfaced here, so a caller that only needs the
action grammar does not drag in the rulebook loader and the video stack.
"""

from __future__ import annotations

from vigil.agent.actions import (
    ACTION_ADAPTER,
    Action,
    ClearAction,
    FinishAction,
    FlagAction,
    PolicyAction,
    RelookAction,
    SurveyAction,
    Turn,
)
from vigil.agent.loop import Investigator, Outcome, investigate
from vigil.agent.reasoner import Briefing, Reasoner, ReasonerError, build_reasoner
from vigil.agent.tools import Observation, ToolKit

__all__ = [
    "ACTION_ADAPTER",
    "Action",
    "Briefing",
    "ClearAction",
    "FinishAction",
    "FlagAction",
    "Investigator",
    "Observation",
    "Outcome",
    "PolicyAction",
    "Reasoner",
    "ReasonerError",
    "RelookAction",
    "SurveyAction",
    "ToolKit",
    "Turn",
    "build_reasoner",
    "investigate",
]
