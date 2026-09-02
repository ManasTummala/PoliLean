"""Tests for the scikit-learn classifier."""

from __future__ import annotations

import pytest

from polilean.model import PoliticalLeanClassifier


def test_train_returns_metrics(trained_classifier):
    assert trained_classifier.pipeline is not None


def test_predict_returns_prediction(trained_classifier):
    pred = trained_classifier.predict("Raise taxes on billionaires to fund welfare")
    assert pred.lean in {"left", "right", "centrist"}
    assert 0.0 <= pred.confidence <= 1.0
    assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-6


def test_predict_dict(trained_classifier):
    d = trained_classifier.predict("Cut taxes and shrink government").to_dict()
    assert d["lean"] in {"left", "right", "centrist"}
    assert "probabilities" in d


def test_predict_with_evidence(trained_classifier):
    pred = trained_classifier.predict("Raise taxes on billionaires to fund welfare", explain=True)
    assert pred.evidence
    for _lean, feats in pred.evidence.items():
        assert all(isinstance(f, str) and isinstance(w, float) for f, w in feats)


def test_linear_svc_train_and_predict():
    """LinearSVC path: no predict_proba, softmax over decision function."""
    from polilean.model import PoliticalLeanClassifier

    svc = PoliticalLeanClassifier(classifier="linear_svc")
    svc.train(_small_df(), test_size=0.25)
    pred = svc.predict("universal healthcare for everyone", explain=True)
    assert pred.lean in {"left", "right", "centrist"}
    assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-3
    assert pred.evidence  # LinearSVC has coef_


def _small_df():
    import pandas as pd

    return pd.DataFrame({
        "text": [
            "raise taxes fund welfare programs",
            "universal healthcare free college",
            "strong unions working class",
            "cut taxes free market deregulate",
            "border security law order",
            "private enterprise innovation",
            "bipartisan compromise budget",
            "evidence based policy reform",
            "sensible regulation growth",
        ],
        "lean": ["left"] * 3 + ["right"] * 3 + ["centrist"] * 3,
    })


def test_predict_many(trained_classifier):
    preds = trained_classifier.predict_many(["free markets", "universal healthcare"])
    assert len(preds) == 2
    assert all(p.lean in {"left", "right", "centrist"} for p in preds)


def test_save_and_load(trained_classifier, tmp_path):
    path = trained_classifier.save(tmp_path / "model.pkl")
    assert path.exists()
    fresh = PoliticalLeanClassifier()
    fresh.load(path)
    assert fresh.pipeline is not None


def test_invalid_classifier_name():
    with pytest.raises(ValueError):
        PoliticalLeanClassifier(classifier="bogus")
