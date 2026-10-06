"""Tests for HF dataset source loaders (offline-safe parts).

ELI5: these tests never touch the network. They check the label-translation
rules and feed a hand-made fake AllSides.zip to the parser to prove the
folder-to-lean mapping works (and that junk folders are skipped).
"""

from __future__ import annotations

import pytest

from polilean.data import sources


def test_load_source_unknown():
    with pytest.raises(ValueError, match="Unknown source"):
        sources.load_source("does_not_exist")


def test_semeval_label_mapping():
    assert sources.SEMEVAL_TO_LEAN["right"] == "right"
    assert sources.SEMEVAL_TO_LEAN["left"] == "left"
    assert sources.SEMEVAL_TO_LEAN["least"] == "centrist"


def test_to_lean():
    assert sources._to_lean("left") == "left"
    assert sources._to_lean("center") == "centrist"
    assert sources._to_lean("right") == "right"
    assert sources._to_lean("unknown") is None


def test_load_allsides_from_local_zip(tmp_path, monkeypatch):
    """Build a fake AllSides.zip and verify the parser handles it."""
    import io
    import zipfile

    zip_path = tmp_path / "AllSides.zip"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("left/article1.txt", "Tax the rich and fund welfare.")
        zf.writestr("right/article2.txt", "Cut taxes and secure the border.")
        zf.writestr("center/article3.txt", "A balanced compromise on budget.")
        zf.writestr("junk/nested/article4.txt", "Should be skipped (unknown folder).")
    zip_path.write_bytes(buf.getvalue())

    monkeypatch.setattr(sources, "CACHE_DIR", tmp_path)
    df = sources.load_allsides()
    assert sorted(df["lean"].unique()) == ["centrist", "left", "right"]
    assert len(df) == 3
