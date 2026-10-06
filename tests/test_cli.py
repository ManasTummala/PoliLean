"""Tests for the CLI.

ELI5: these tests pretend to type `polilean ...` commands and check what
comes out - JSON parses, thresholds abstain correctly, the train job
prints its accuracy report and writes a model file, and `gui --help`
lists its options. Nothing opens a real terminal here.
"""

from __future__ import annotations

import json

from polilean.cli import main


# ELI5: running with no command just prints the menu (exit 0), and a
# missing model file is reported as a clean error (exit 1), not a traceback.
def test_no_command_prints_help(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "usage" in out.lower()


def test_predict_missing_model(tmp_path, capsys):
    rc = main(["predict", "hello", "--model", str(tmp_path / "none.pkl")])
    assert rc == 1


# ELI5: `polilean predict --json` must print parseable JSON with a real lean
# (the machine-readable contract other tools build on).
def test_predict_json(trained_classifier, tmp_path, capsys):
    model_path = trained_classifier.save(tmp_path / "m.pkl")
    rc = main(["predict", "Tax the rich", "--json", "--model", str(model_path)])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["lean"] in {"left", "right", "centrist"}


# ELI5: --threshold makes the CLI abstain: nonsense text comes back as
# 'uncertain' below the threshold, the env-var default works too, and an
# out-of-range value is rejected before any model loads.
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


# ELI5: the JSON must also carry all five value axes with in-range numbers.
def test_predict_json_includes_axes(trained_classifier, tmp_path, capsys):
    model_path = trained_classifier.save(tmp_path / "m.pkl")
    rc = main(["predict", "Tax the rich", "--json", "--model", str(model_path)])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    axes = data["axes"]
    assert {"economic", "social", "authority", "foreign", "environment"} <= set(axes)
    assert -1.0 <= axes["economic"]["position"] <= 1.0
    assert 0.0 <= axes["economic"]["confidence"] <= 1.0


# ELI5: `polilean gui --help` must list its options (--port, --no-browser)
# without actually starting a server.
def test_gui_help(capsys):
    import pytest

    with pytest.raises(SystemExit) as exc:
        main(["gui", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--port" in out and "--no-browser" in out


# ELI5: the full train job from the command line - prints accuracy,
# calibration and per-axis scores, and really writes the model file
# (redirected to a temp folder so the repo's model is untouched).
def test_train_cli(tmp_path, capsys, monkeypatch):
    import polilean.model as model_mod

    monkeypatch.setattr(model_mod, "MODELS_DIR", tmp_path)  # don't touch repo model
    rc = main(["train"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Calibration: Brier=" in out
    assert "Value axes trained (5 of 5)" in out
    assert (tmp_path / "logreg.pkl").exists()
