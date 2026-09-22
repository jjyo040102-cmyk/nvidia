"""Judge-facing HTTP surface.

The API deliberately runs the same investigation entry point as the CLI.  These tests use the
bundled scripted backend so they exercise the whole boundary without a network credential.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from vigil.api.app import create_app
from vigil.config import Settings


def client(settings: Settings) -> TestClient:
    return TestClient(create_app(settings))


def test_health_and_status_explain_the_credential_free_mode(tmp_settings: Settings) -> None:
    web = client(tmp_settings)
    health = web.get("/healthz")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    status = web.get("/api/status").json()
    assert status["has_nebius_key"] is False
    assert status["default_mode"] == "demo"
    assert status["sample_count"] == 8
    assert status["demo_is_measurement"] is False


def test_samples_are_real_bundled_recordings(tmp_settings: Settings) -> None:
    response = client(tmp_settings).get("/api/samples")
    assert response.status_code == 200
    payload = response.json()
    ids = {sample["video_id"] for sample in payload["samples"]}
    assert "blind_corner_struck_by" in ids
    assert "control_housekeeping" in ids
    sample = next(
        item for item in payload["samples"] if item["video_id"] == "blind_corner_struck_by"
    )
    assert sample["duration_s"] > 0
    assert sample["camera_label"]


def test_demo_investigation_returns_trace_provenance_and_findings(tmp_settings: Settings) -> None:
    response = client(tmp_settings).post("/api/investigate/blind_corner_struck_by?mode=demo")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["mode"] == "demo"
    assert payload["measured"] is False
    report = payload["report"]
    assert report["assessments"]
    assert report["trace"]["steps"]
    assert report["provenance"]["perception_backend"] == "mock"
    assert "scripted" in report["provenance"]["reasoning_model"]
    assert "warning" in payload["summary"]


def test_unknown_sample_is_a_clean_404_with_available_ids(tmp_settings: Settings) -> None:
    response = client(tmp_settings).post("/api/investigate/not-a-real-camera?mode=demo")
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "not-a-real-camera" in detail
    assert "blind_corner_struck_by" in detail


def test_live_mode_without_a_key_refuses_instead_of_silently_using_mock(
    tmp_settings: Settings,
) -> None:
    response = client(tmp_settings).post("/api/investigate/blind_corner_struck_by?mode=live")
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "VIGIL_NEBIUS_API_KEY" in detail
    assert "never falls back" in detail


def test_dashboard_is_served_at_root(tmp_settings: Settings) -> None:
    response = client(tmp_settings).get("/")
    assert response.status_code == 200
    assert "See the accident before it happens" in response.text
    assert "Vigil" in response.text


def test_sample_media_streams_inline_for_the_dashboard_video(tmp_settings: Settings) -> None:
    response = client(tmp_settings).get("/media/blind_corner_struck_by")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("video/mp4")
    assert "attachment" not in response.headers.get("content-disposition", "").lower()


def test_dashboard_escapes_model_generated_text_before_inserting_html(
    tmp_settings: Settings,
) -> None:
    script = client(tmp_settings).get("/static/app.js").text
    assert "function escapeHtml" in script
    assert "escapeHtml(step.title)" in script
    assert "escapeHtml(rule.text)" in script
    assert "escapeHtml(v)" in script
