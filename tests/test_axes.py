"""Tests for value axes (per-dimension ideology scoring).

ELI5: every value axis must behave like a fair tug-of-war scoreboard -
the five axes exist with the right pole words, the training CSV loads and
validates, training produces honest accuracy numbers, predictions include
all five axes with sensible positions (public text leans public), evidence
is free of meaningless stop words, and saved models keep their axes.
"""

from __future__ import annotations

import pickle

import pytest

from polilean.axes import AXES, load_axes_dataset
from polilean.model import AxisScore, PoliticalLeanClassifier


# ELI5: the axis definitions themselves - 5 axes, each with a negative
# pole, a neutral middle, a positive pole, and a plain-English description.
def test_axes_spec_shape():
    assert len(AXES) == 5
    for name, spec in AXES.items():
        assert spec.name == name
        assert spec.negative != spec.positive
        assert "neutral" not in {spec.negative, spec.positive}
        assert spec.description


# ELI5: the training CSV must load cleanly and be rejected when malformed
# (missing file, wrong columns, labels that do not belong to an axis).
def test_load_axes_dataset():
    df = load_axes_dataset()
    assert set(df.columns) >= {"text", "axis", "label"}
    assert set(df["axis"]) == set(AXES)
    for axis, spec in AXES.items():
        labels = set(df.loc[df["axis"] == axis, "label"])
        assert labels == spec.labels  # every axis has both poles + neutral
        assert len(df[df["axis"] == axis]) >= 12


def test_load_axes_dataset_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_axes_dataset(tmp_path / "nope.csv")


def test_load_axes_dataset_invalid_values(tmp_path):
    bad_axis = tmp_path / "bad_axis.csv"
    bad_axis.write_text("text,axis,label\nsome text,economics,market\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown axes"):
        load_axes_dataset(bad_axis)

    bad_label = tmp_path / "bad_label.csv"
    bad_label.write_text("text,axis,label\nsome text,economic,marxist\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid labels"):
        load_axes_dataset(bad_label)

    missing_col = tmp_path / "missing.csv"
    missing_col.write_text("text,axis\nsome text,economic\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        load_axes_dataset(missing_col)


# ELI5: training must produce honest scores - accuracy above chance (0.33),
# calibration metrics in range, and clear errors for undersized datasets.
def test_train_axes_returns_metrics():
    clf = PoliticalLeanClassifier(classifier="logreg")
    metrics = clf.train_axes()
    assert set(metrics) == set(AXES)
    for axis, m in metrics.items():
        assert 0.45 <= m["accuracy"] <= 1.0, axis  # 3 classes, chance=0.33; guards regressions
        assert 0.0 <= m["brier"] <= 2.0, axis
        assert 0.0 <= m["ece"] <= 1.0, axis
        assert m["n"] >= 60 and m["folds"] == 5
    assert set(clf.axis_pipelines) == set(AXES)


def test_train_axes_validation():
    clf = PoliticalLeanClassifier(classifier="logreg")
    import pandas as pd

    with pytest.raises(ValueError, match="missing columns"):
        clf.train_axes(pd.DataFrame({"text": ["x"]}))
    with pytest.raises(ValueError, match="Unknown axes"):
        clf.train_axes(
            pd.DataFrame({"text": ["x"], "axis": ["bogus"], "label": ["neutral"]})
        )
    with pytest.raises(ValueError, match="Invalid labels"):
        clf.train_axes(
            pd.DataFrame({"text": ["x"], "axis": ["economic"], "label": ["bogus"]})
        )


# ELI5: predictions must carry all five axes with stop-word-free evidence,
# and the signs must make sense (public text leans public, market text
# leans market) - the actual ideology check.
def test_predict_includes_axes(trained_classifier):
    pred = trained_classifier.predict(
        "Raise taxes on billionaires to fund universal healthcare", explain=True
    )
    assert set(pred.axes) == set(AXES)
    for name, ax in pred.axes.items():
        assert isinstance(ax, AxisScore)
        assert ax.axis == name
        assert ax.label in ax.probabilities
        assert -1.0 <= ax.position <= 1.0
        assert 0.0 <= ax.confidence <= 1.0
        assert abs(sum(ax.probabilities.values()) - 1.0) < 1e-6
        assert ax.description
    # evidence is meaningful: content words only (no function-word noise)
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    assert pred.axes["economic"].evidence  # text is clearly about economics
    for ax in pred.axes.values():
        for feats in ax.evidence.values():
            assert all(f not in ENGLISH_STOP_WORDS for f, _ in feats)
    d = pred.to_dict()["axes"]
    assert set(d) == set(AXES)
    assert d["economic"]["label"] == pred.axes["economic"].label


def test_axis_directions(trained_classifier):
    public = trained_classifier.predict(
        "Nationalize the energy companies and fund universal healthcare with higher taxes"
    )
    market = trained_classifier.predict(
        "Deregulate the banks and cut taxes on corporations"
    )
    assert public.axes["economic"].position > 0  # toward the public pole
    assert market.axes["economic"].position < 0  # toward the market pole


def test_axes_present_in_predict_many(trained_classifier):
    preds = trained_classifier.predict_many(["free markets", "universal healthcare"])
    assert all(set(p.axes) == set(AXES) for p in preds)


def test_axes_empty_without_axis_training():
    import pandas as pd

    rows = {
        "left": ["tax the rich", "fund welfare programs", "raise minimum wage", "union rights now"],
        "right": ["cut taxes", "deregulate business", "strong borders", "law and order"],
        "centrist": ["balanced budget reform", "pragmatic compromise",
                    "evidence policy", "moderate growth"],
    }
    texts = [t for phrases in rows.values() for t in phrases]
    leans = [lean for lean, phrases in rows.items() for _ in phrases]
    clf = PoliticalLeanClassifier(classifier="logreg")
    clf.train(pd.DataFrame({"text": texts, "lean": leans}), test_size=0.25)
    pred = clf.predict("anything at all")
    assert pred.axes == {}


# ELI5: saved models must keep their axes - a fresh load round-trips them,
# and old version-1 pickles (bare pipeline, no axes) still open safely.
def test_save_load_roundtrip_preserves_axes(trained_classifier, tmp_path):
    path = trained_classifier.save(tmp_path / "m.pkl")
    fresh = PoliticalLeanClassifier()
    fresh.load(path)
    assert set(fresh.axis_pipelines) == set(AXES)
    assert fresh.predict("cut taxes").axes


def test_load_legacy_pipeline_format(trained_classifier, tmp_path):
    """Old pickles contain a bare Pipeline; they load without axes."""
    path = tmp_path / "legacy.pkl"
    with path.open("wb") as fh:
        pickle.dump(trained_classifier.pipeline, fh)
    fresh = PoliticalLeanClassifier()
    fresh.load(path)
    assert fresh.pipeline is not None
    assert fresh.axis_pipelines == {}
    assert fresh.predict("cut taxes").axes == {}
