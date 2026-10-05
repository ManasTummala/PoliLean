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
    metrics = svc.train(_small_df(), test_size=0.25)
    assert 0.0 <= metrics["brier"] <= 2.0
    assert 0.0 <= metrics["ece"] <= 1.0
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


# ------------------------------------------------------------ calibration
def test_train_reports_calibration_metrics():
    clf = PoliticalLeanClassifier(classifier="logreg")
    metrics = clf.train(_small_df(), test_size=0.25)
    assert 0.0 <= metrics["brier"] <= 2.0  # multiclass Brier, lower is better
    assert 0.0 <= metrics["ece"] <= 1.0  # expected calibration error


def test_calibration_metrics_perfect_predictions():
    import numpy as np

    from polilean.model import calibration_metrics

    classes = ["left", "right", "centrist"]
    proba = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    m = calibration_metrics(["left", "right", "centrist"], proba, classes)
    assert m["brier"] == pytest.approx(0.0)
    assert m["ece"] == pytest.approx(0.0)


def test_calibration_metrics_detect_miscalibration():
    import numpy as np

    from polilean.model import calibration_metrics

    # Confident (0.9) but wrong -> both metrics must be > 0.
    proba = np.array([[0.9, 0.05, 0.05]])
    m = calibration_metrics(["right"], proba, ["left", "right", "centrist"])
    assert m["brier"] > 0.0
    assert m["ece"] > 0.0


# --------------------------------------------------------------- abstain
def test_abstain_below_threshold(trained_classifier):
    pred = trained_classifier.predict("Raise taxes on billionaires", threshold=2.0)
    assert pred.lean == "uncertain"
    # raw probabilities survive the abstain so callers see the near-tie
    assert pred.probabilities
    assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-6


def test_abstain_boundary_is_exact(trained_classifier):
    text = "The weather is nice today"
    base = trained_classifier.predict(text)
    just_over = trained_classifier.predict(text, threshold=base.confidence + 0.01)
    exactly_at = trained_classifier.predict(text, threshold=base.confidence)
    assert just_over.lean == "uncertain"  # confidence < threshold -> abstain
    assert exactly_at.lean in {"left", "right", "centrist"}  # equal -> label


def test_threshold_zero_never_abstains(trained_classifier):
    pred = trained_classifier.predict("Cut taxes and shrink government", threshold=0.0)
    assert pred.lean in {"left", "right", "centrist"}


def test_instance_default_threshold_and_override():
    clf = PoliticalLeanClassifier(classifier="logreg", threshold=2.0)
    clf.train(_small_df(), test_size=0.25)
    assert clf.predict("balanced budget compromise").lean == "uncertain"
    # per-call threshold overrides the instance default
    assert clf.predict("balanced budget compromise", threshold=0.0).lean in {
        "left", "right", "centrist",
    }


def test_predict_many_abstain(trained_classifier):
    preds = trained_classifier.predict_many(
        ["free markets", "universal healthcare"], threshold=2.0
    )
    assert all(p.lean == "uncertain" for p in preds)
