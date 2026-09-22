"""The command line: the first thing a judge or a reviewer types.

Every command here is the boundary between "works on my machine" and "runs from a fresh clone",
so the assertions are about exit codes and files on disk rather than about returned objects --
that is all a reviewer can actually observe. Three of them matter most: ``run`` must exit
non-zero when an investigation was cut short (a silent partial report is the worst failure mode
this product has), ``probe`` must refuse politely when there is no key rather than traceback,
and asking for a hosted backend without a key must print the fix instead of a stack.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from vigil.agent.loop import Outcome
from vigil.cli import _backend, _select, app
from vigil.models.report import IncidentReport, ModelProvenance
from vigil.models.risk import RiskLedger
from vigil.models.trace import InvestigationTrace
from vigil.nebius import NebiusError
from vigil.video.source import VideoSource

from .helpers import CLIP_DIR, POLICY_DIR
from .helpers import ROOT as REPO_ROOT

RUNNER = CliRunner()


def invoke(*args: str, env: dict[str, str] | None = None) -> Any:
    """Run the CLI in a throwaway directory.

    ``typer.testing.CliRunner`` has no ``isolated_filesystem`` (unlike click's), and the
    app reads ``.env`` from the working directory, so the isolation is done by hand: a
    developer key sitting in the repo's ``.env`` must not decide whether a test passes.
    """
    with _throwaway_cwd():
        return RUNNER.invoke(app, list(args), env=env or {}, color=False)


@contextmanager
def _throwaway_cwd() -> Iterator[str]:
    old = Path.cwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            yield tmp
        finally:
            os.chdir(old)


def repo_env(**extra: str) -> dict[str, str]:
    """Point every path the CLI resolves from ``Settings`` back at the repo.

    The commands run in an isolated directory, so the defaults (``data/clips``,
    ``data/policy``) would resolve to nothing there.
    """
    base = {
        "VIGIL_CLIP_DIR": str(CLIP_DIR),
        "VIGIL_POLICY_DIR": str(POLICY_DIR),
        "VIGIL_ARTIFACT_DIR": str(REPO_ROOT / "artifacts"),
    }
    base.update(extra)
    return base


# ---------------------------------------------------------------------------------------- the root


def test_typing_vigil_with_no_arguments_shows_the_commands() -> None:
    """``no_args_is_help`` prints the command list and exits 2."""
    result = invoke()
    assert result.exit_code == 2, result.output
    for command in ("clips", "probe", "run", "demo", "eval"):
        assert command in result.output


def test_every_backend_name_is_rejected_except_the_four_documented() -> None:
    assert _backend("nebius") == "nebius"
    with pytest.raises(Exception, match="choose one of"):
        _backend("openai")
    result = invoke("run", "--perception", "quantum")
    assert result.exit_code != 0
    assert "auto, mock, nebius, cosmos" in result.output


# ------------------------------------------------------------------------------------------ probe


def test_probe_refuses_without_a_key_instead_of_tracing_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VIGIL_NEBIUS_API_KEY", raising=False)
    result = invoke("probe")
    assert result.exit_code == 2, result.output


def test_probe_prints_what_the_account_can_serve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_probe(settings: Any) -> dict[str, Any]:
        return {
            "base_url": settings.nebius_base_url,
            "model_count": 3,
            "models": ["nvidia/nemotron-3-super", "nvidia/nemotron-3-nano-omni", "meta/llama"],
            "picked_reasoning": "nvidia/nemotron-3-super",
            "picked_vision": "nvidia/nemotron-3-nano-omni",
        }

    monkeypatch.setattr("vigil.cli.probe_account", fake_probe)
    result = invoke("probe", env={"VIGIL_NEBIUS_API_KEY": "sk-test-key-123456"})
    assert result.exit_code == 0, result.output
    assert "sk-t…3456" in result.output, "the fingerprint has to be safe to paste into a ticket"
    assert "sk-test-key-123456" not in result.output
    assert "picked reasoning" in result.output
    assert "nemotron-3-nano-omni" in result.output
    assert "3 models" in result.output


def test_probe_reports_an_unreachable_account_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def broken(_settings: Any) -> dict[str, Any]:
        raise NebiusError("could not list models")

    monkeypatch.setattr("vigil.cli.probe_account", broken)
    result = invoke("probe", env={"VIGIL_NEBIUS_API_KEY": "sk-test-key-123456"})
    assert result.exit_code == 1
    assert "probe failed: could not list models" in result.output
    assert "Traceback" not in result.output


# -------------------------------------------------------------------------------------------- run


def test_run_writes_a_json_and_a_markdown_report_per_clip(tmp_path: Path) -> None:
    out = tmp_path / "reports"
    result = invoke(
        "run",
        "-v",
        "blind_corner_struck_by",
        "--perception",
        "mock",
        "--reasoning",
        "mock",
        "--out",
        str(out),
        env=repo_env(),
    )
    assert result.exit_code == 0, result.output
    assert "1 video(s)" in result.output
    assert "perception=mock" in result.output

    md = out / "blind_corner_struck_by.md"
    js = out / "blind_corner_struck_by.json"
    assert md.exists()
    assert js.exists()
    assert md.read_text(encoding="utf-8").startswith("# ")
    payload = json.loads(js.read_text(encoding="utf-8"))
    assert payload["video_id"] == "blind_corner_struck_by"
    assert payload["assessments"], "the scripted controller must find the authored collision"
    assert payload["trace"]["steps"], "--trace is on by default"
    assert "warning" in result.output
    assert "finding(s)" in result.output
    assert "scripted backends" in result.output


def test_no_trace_keeps_the_audit_trail_out_of_the_handoff_file(tmp_path: Path) -> None:
    out = tmp_path / "lean"
    result = invoke(
        "run",
        "-v",
        "blind_corner_struck_by",
        "--perception",
        "mock",
        "--reasoning",
        "mock",
        "--no-trace",
        "--out",
        str(out),
        env=repo_env(),
    )
    assert result.exit_code == 0, result.output
    payload = json.loads((out / "blind_corner_struck_by.json").read_text(encoding="utf-8"))
    assert "trace" not in payload
    assert payload["assessments"]


def test_demo_runs_the_whole_set_with_no_credentials(tmp_path: Path) -> None:
    result = invoke(
        "demo",
        "--out",
        str(tmp_path / "demo"),
        env=repo_env(),
    )
    assert result.exit_code == 0, result.output
    assert "8 video(s)" in result.output
    written = list((tmp_path / "demo").glob("*.md"))
    assert len(written) == 8, "every clip has to leave a report behind"
    controls = [p for p in written if p.stem.startswith("control_")]
    assert len(controls) == 2
    for path in controls:
        assert "No finding met the reporting bar" in path.read_text(encoding="utf-8")


def test_an_unknown_clip_names_the_ones_that_exist() -> None:
    result = invoke("run", "-v", "nope", env=repo_env())
    assert result.exit_code != 0
    assert "no clip 'nope' in the index" in result.output
    assert "blind_corner_struck_by" in result.output


def test_a_truncated_investigation_exits_non_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partial report must fail loudly: an all-clear that never finished looks identical."""
    report = IncidentReport(
        report_id="VIG-cut-1",
        video_id="blind_corner_struck_by",
        headline="Investigation ended early",
        narrative="one\ntruncated",
        provenance=ModelProvenance(
            perception_backend="mock",
            vision_model="scripted",
            reasoning_model="scripted",
        ),
    )
    outcome = Outcome(
        ledger=RiskLedger(video_id=report.video_id, summary="stopped early"),
        trace=InvestigationTrace(video_id=report.video_id),
        report=report,
        stopped="the token ceiling was reached before the agent called finish",
    )

    async def cut_short(*_args: Any, **_kwargs: Any) -> Outcome:
        return outcome

    monkeypatch.setattr("vigil.cli.investigate", cut_short)
    result = invoke(
        "run",
        "-v",
        "blind_corner_struck_by",
        "--perception",
        "mock",
        "--reasoning",
        "mock",
        "--out",
        str(tmp_path / "r"),
        env=repo_env(),
    )
    assert result.exit_code == 1, "a stopped run cannot report success"
    assert "incomplete: the token ceiling" in result.output
    assert "truncated" in result.output, "the narrative lines are echoed for the person watching"


@pytest.mark.parametrize("flag", ["--perception", "--reasoning"])
def test_naming_a_hosted_backend_without_a_key_is_an_instruction_not_a_traceback(
    flag: str,
) -> None:
    """The one failure a fresh clone hits, and the one place a traceback costs a run.

    The guard checks for a key and stops there -- it never silently downgrades to mock, because
    a demo that quietly ran on scripted data would be the worst thing this product could do.
    """
    result = invoke("run", flag, "nebius", env=repo_env(VIGIL_NEBIUS_API_KEY=""))
    assert result.exit_code == 2, result.output
    assert "VIGIL_NEBIUS_API_KEY in .env" in result.output
    assert "--perception mock" in result.output, "the message has to name the way out"
    assert "Traceback" not in result.output


def test_the_guard_steps_aside_once_a_key_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, str] = {}

    async def pretend(_video: Any, settings: Any, **_kw: Any) -> Any:
        seen["perception"] = settings.resolved_perception
        return Outcome(
            ledger=RiskLedger(video_id=_video.video_id, summary="done"),
            trace=InvestigationTrace(video_id=_video.video_id),
            report=IncidentReport(
                report_id="VIG-guard-1",
                video_id=_video.video_id,
                headline="Key present, so nothing to refuse",
                narrative="one\ntwo",
                provenance=ModelProvenance(
                    perception_backend="nebius",
                    vision_model="nvidia/nemotron-3-nano-omni-30b-a3b",
                    reasoning_model="nvidia/nemotron-3-super-120b-a12b",
                ),
            ),
        )

    monkeypatch.setattr("vigil.cli.investigate", pretend)
    result = invoke(
        "run",
        "-v",
        "blind_corner_struck_by",
        "--perception",
        "nebius",
        "--out",
        str(tmp_path / "r"),
        env=repo_env(VIGIL_NEBIUS_API_KEY="sk-looking-like-a-real-one"),
    )
    assert result.exit_code == 0, result.output
    assert seen == {"perception": "nebius"}
    assert "scripted backends" not in result.output


def test_selection_accepts_the_two_wildcards_and_one_clip(
    authored_videos: list[VideoSource],
) -> None:
    videos = authored_videos
    assert _select(videos, "all") == videos
    assert _select(videos, "*") == videos
    chosen = _select(videos, "blind_corner_struck_by")
    assert [v.video_id for v in chosen] == ["blind_corner_struck_by"]
    with pytest.raises(Exception, match="no clip 'nope'"):
        _select(videos, "nope")


# ------------------------------------------------------------------------------------------ clips


def test_clips_reports_the_dataset_without_redoing_finished_work(tmp_path: Path) -> None:
    """``vigil clips`` is the first command a reviewer runs; it must be re-runnable and quiet."""
    target = tmp_path / "clips"
    shutil.copytree(CLIP_DIR, target)
    before = {p.name: p.stat().st_mtime_ns for p in target.glob("*.mp4")}
    assert before, "the shipped dataset should contain clips"

    result = invoke("clips", "--out", str(target))
    assert result.exit_code == 0, result.output
    assert "8 clip(s)" in result.output
    assert "2 controls" in result.output
    for name, stamp in before.items():
        assert (target / name).stat().st_mtime_ns == stamp, f"{name} was re-rendered"


# ------------------------------------------------------------------------------------------- eval


def test_eval_refuses_to_present_a_scripted_run_as_a_measurement(tmp_path: Path) -> None:
    """The table is the one artefact that ends up in a README, so the guard has to be at the CLI.

    Mock backends reading mock-authored scenarios cannot produce a number about a model. The
    command still prints the whole board -- it is useful as a smoke test -- but it says what it
    is and exits non-zero, which is what keeps the number out of a slide by accident.
    """
    out = tmp_path / "eval"
    result = invoke(
        "eval",
        "--perception",
        "mock",
        "--reasoning",
        "mock",
        "--out",
        str(out),
        env=repo_env(),
    )
    assert result.exit_code == 1, result.output
    assert "NOT A MEASUREMENT" in result.output
    assert "--perception nebius" in result.output, "the message has to name the way out"

    payload = json.loads((out / "latest.json").read_text(encoding="utf-8"))
    assert payload["measured"] is False
    assert payload["exit_code"] == 1
    assert len(payload["summary"]["per_clip"]) == 8
    board = (out / "latest.md").read_text(encoding="utf-8")
    assert "NOT a measurement" in board
    for clip_id in ("blind_corner_struck_by", "control_housekeeping"):
        assert clip_id in board


def test_eval_scores_only_the_clips_it_was_handed(tmp_path: Path) -> None:
    """``--clips`` narrows the sweep, and the board has to shrink with it rather than lie."""
    out = tmp_path / "one"
    result = invoke(
        "eval",
        "--clips",
        "blind_corner_struck_by",
        "--perception",
        "mock",
        "--reasoning",
        "mock",
        "--out",
        str(out),
        env=repo_env(),
    )
    assert result.exit_code == 1, result.output
    assert "Clips scored: 1 (1 with authored hazards, 0 controls)" in result.output
    assert "1/1 anticipated" in result.output
    assert "control_housekeeping" not in result.output
    payload = json.loads((out / "latest.json").read_text(encoding="utf-8"))
    rows = payload["summary"]["per_clip"]
    assert [r["clip_id"] for r in rows] == ["blind_corner_struck_by"]
    assert rows[0]["truth"], "the authored clip is scored against its own sidecar"


def test_eval_names_a_clip_it_cannot_find_before_running_anything() -> None:
    """Failing after eight investigations is the difference between a typo and a wasted run."""
    result = invoke("eval", "--clips", "nope", env=repo_env())
    assert result.exit_code != 0
    assert "no clip(s) nope in the index" in result.output
    assert "blind_corner_struck_by" in result.output, "the message lists what is actually there"

# ------------------------------------------------------------------------------------------ serve


def test_serve_launches_the_judge_dashboard(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_run(app_name: str, **kwargs: object) -> None:
        seen["app"] = app_name
        seen.update(kwargs)

    monkeypatch.setattr("uvicorn.run", fake_run)
    result = invoke("serve", "--host", "127.0.0.1", "--port", "9876")
    assert result.exit_code == 0, result.output
    assert seen["app"] == "vigil.api.app:app"
    assert seen["host"] == "127.0.0.1"
    assert seen["port"] == 9876
    assert seen["reload"] is False
    assert "http://127.0.0.1:9876" in result.output
