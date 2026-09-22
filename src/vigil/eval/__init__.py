"""The benchmark: authored ground truth, the arithmetic that scores a run against it, and
the table that reports both.

Read :mod:`vigil.eval.metrics` first. Every published claim about Vigil's accuracy is decided
there, in pure functions, with the definitions written down.
"""

from vigil.eval.groundtruth import TrueHazard, VideoTruth, load_truth
from vigil.eval.harness import EvalRun, Skipped, evaluate, is_measurement
from vigil.eval.metrics import ClipScore, Match, Scorecard, score_clip, score_run
from vigil.eval.table import render

__all__ = [
    "ClipScore",
    "EvalRun",
    "Match",
    "Scorecard",
    "Skipped",
    "TrueHazard",
    "VideoTruth",
    "evaluate",
    "is_measurement",
    "load_truth",
    "render",
    "score_clip",
    "score_run",
]
