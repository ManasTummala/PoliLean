"""Tests for the CLI."""

from __future__ import annotations

import json

from polilean.cli import main


def test_no_command_prints_help(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "usage" in out.lower()


def test_predict_missing_model(tmp_path, capsys):
    rc = main(["predict", "hello", "--model", str(tmp_path / "none.pkl")])
    assert rc == 1


def test_predict_json(trained_classifier, tmp_path, capsys):
    model_path = trained_classifier.save(tmp_path / "m.pkl")
    rc = main(["predict", "Tax the rich", "--json", "--model", str(model_path)])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["lean"] in {"left", "right", "centrist"}


def test_predict_abstain_threshold(trained_classifier, tmp_path, capsys):
    model_path = trained_classifier.save(tmp_path / "m.pkl")
    rc = main([
        "predict", "asdf qwer zxcv", "--json", "--model", str(model_path),
        "--threshold", "0.9",
    ])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["lean"] == "uncertain"
    assert data["confidence"] < 0.9


def test_predict_env_threshold(trained_classifier, tmp_path, capsys, monkeypatch):
    model_path = trained_classifier.save(tmp_path / "m.pkl")
    monkeypatch.setenv("POLILEAN_ABSTAIN_THRESHOLD", "0.9")
    rc = main(["predict", "asdf qwer", "--json", "--model", str(model_path)])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["lean"] == "uncertain"


def test_predict_invalid_threshold(tmp_path, capsys):
    rc = main([
        "predict", "hello", "--model", str(tmp_path / "none.pkl"),
        "--threshold", "1.5",
    ])
    assert rc == 2
    assert "between 0 and 1" in capsys.readouterr().err


def test_predict_json_includes_axes(trained_classifier, tmp_path, capsys):
    model_path = trained_classifier.save(tmp_path / "m.pkl")
    rc = main(["predict", "Tax the rich", "--json", "--model", str(model_path)])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    axes = data["axes"]
    assert {"economic", "social", "authority", "foreign", "environment"} <= set(axes)
    assert -1.0 <= axes["economic"]["position"] <= 1.0
    assert 0.0 <= axes["economic"]["confidence"] <= 1.0


def test_gui_help(capsys):
    import pytest

    with pytest.raises(SystemExit) as exc:
        main(["gui", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--port" in out and "--no-browser" in out


def test_train_cli(tmp_path, capsys, monkeypatch):
    import polilean.model as model_mod

    monkeypatch.setattr(model_mod, "MODELS_DIR", tmp_path)  # don't touch repo model
    rc = main(["train"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Calibration: Brier=" in out
    assert "Value axes trained (5 of 5)" in out
    assert (tmp_path / "logreg.pkl").exists()
