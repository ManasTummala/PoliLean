"""Tests for the FastAPI service."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from polilean.api import app


@pytest.fixture(scope="module")
def client(trained_classifier):
    # trained_classifier session fixture ensures a model file exists on disk
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict(client):
    r = client.post("/predict", json={"text": "Tax the rich to fund universal healthcare."})
    assert r.status_code == 200
    body = r.json()
    assert body["lean"] in {"left", "right", "centrist"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-3
    assert body["evidence"]  # explain defaults to True


def test_predict_empty_text(client):
    r = client.post("/predict", json={"text": "   "})
    assert r.status_code == 422


def test_predict_explain_off(client):
    r = client.post("/predict", json={"text": "Cut taxes", "explain": False})
    assert r.status_code == 200
    assert r.json()["evidence"] is None
