"""Tests for pandas dataset loading."""

from __future__ import annotations

import pandas as pd
import pytest

from polilean.data.dataset import load_dataset


def test_load_default_dataset():
    df = load_dataset()
    assert isinstance(df, pd.DataFrame)
    assert {"text", "lean"} <= set(df.columns)
    assert set(df["lean"].unique()) <= {"left", "right", "centrist"}
    assert len(df) > 0


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "nope.csv")


def test_missing_column_raises(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("text\nhello\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(path)


def test_invalid_label_raises(tmp_path):
    path = tmp_path / "bad2.csv"
    path.write_text("text,lean\nhello,communist\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid lean labels"):
        load_dataset(path)
