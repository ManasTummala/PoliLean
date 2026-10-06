"""Tests for the FastAPI service.

ELI5: these tests drive the web API like a browser would - they start the
app, ask it questions, and check the answers: the GUI page loads, /health
reports the model, /predict returns a valid lean + probabilities + axes,
and bad input (blank text, out-of-range threshold) is rejected politely.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from polilean.api import app


@pytest.fixture(scope="module")
def client(trained_classifier):
    # trained_classifier session fixture ensures a model file exists on disk
    with TestClient(app) as c:
        yield c


# ELI5: the GUI page itself must load and contain every interactive piece
# (button, bars, radar, dark theme, the 55% Undefined default).
def test_gui_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    html = r.text
    assert "PoliLean" in html
    assert "id=\"analyze\"" in html          # analyze button
    assert "id=\"prob-bars\"" in html        # percentile probability bars
    assert "id=\"axis-bars\"" in html        # value-axis diverging bars
    assert "id=\"radar\"" in html            # radar chart svg
    assert "fetch(\"/predict\"" in html      # posts to the predict endpoint
    # dark theme
    assert 'name="color-scheme" content="dark"' in html
    assert "--bg: #0b1220" in html
    # Undefined outcome for insufficient information (default threshold)
    assert 'id="threshold"' in html and 'value="55"' in html
    assert '"Undefined"' in html
    # professional Title-Case axis labels
    for title in ("Economic", "Social", "Authority", "Foreign Policy", "Environment"):
        assert title in html


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ELI5: the happy path - /health reports the model, /predict returns a real
# lean with probabilities that sum to 100% and evidence included by default.
def test_predict(client):
    r = client.post("/predict", json={"text": "Tax the rich to fund universal healthcare."})
    assert r.status_code == 200
    body = r.json()
    assert body["lean"] in {"left", "right", "centrist"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-3
    assert body["evidence"]  # explain defaults to True


# ELI5: bad input is rejected politely (blank text and out-of-range
# thresholds must come back as 422, not a crash).
def test_predict_empty_text(client):
    r = client.post("/predict", json={"text": "   "})
    assert r.status_code == 422


def test_predict_explain_off(client):
    r = client.post("/predict", json={"text": "Cut taxes", "explain": False})
    assert r.status_code == 200
    assert r.json()["evidence"] is None


# ELI5: threshold = "give up below this confidence" - nonsense text must come
# back as 'uncertain' while still carrying valid probabilities.
def test_predict_abstain_below_threshold(client):
    # OOV text -> near-prior probabilities; threshold above confidence abstains
    r = client.post(
        "/predict",
        json={"text": "asdf qwer zxcv plugh", "explain": False, "threshold": 0.9},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["lean"] == "uncertain"
    assert body["confidence"] < 0.9
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-3


def test_predict_threshold_out_of_range(client):
    r = client.post("/predict", json={"text": "hello", "threshold": 1.5})
    assert r.status_code == 422


# ELI5: every response must include all five value axes with in-range
# position/confidence, full probabilities, evidence, and a description.
def test_predict_includes_axes(client):
    r = client.post("/predict", json={"text": "Raise taxes to fund universal healthcare"})
    assert r.status_code == 200
    axes = r.json()["axes"]
    assert axes and {"economic", "social", "authority", "foreign", "environment"} <= set(axes)
    ax = axes["economic"]
    assert -1.0 <= ax["position"] <= 1.0
    assert 0.0 <= ax["confidence"] <= 1.0
    assert abs(sum(ax["probabilities"].values()) - 1.0) < 1e-3
    assert ax["evidence"]  # explain defaults to True
    assert ax["description"]
    assert ax["negative"] and ax["positive"]  # pole names for the GUI bars
