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
