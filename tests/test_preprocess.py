"""Tests for spaCy preprocessing."""

from __future__ import annotations

from polilean.preprocess import preprocess, preprocess_many


def test_empty_string():
    assert preprocess("") == ""


def test_whitespace_only():
    assert preprocess("   \n\t  ") == ""


def test_basic_lemmatization():
    result = preprocess("The workers are running and demanding better wages.")
    assert "worker" in result
    assert "run" in result
    assert "wage" in result


def test_stopwords_removed():
    result = preprocess("The quick brown fox jumps over the lazy dog")
    assert " the " not in f" {result} "
    assert "the" not in result.split()


def test_lowercase_output():
    assert preprocess("HELLO World") == preprocess("hello world")


def test_preprocess_many_matches_single():
    texts = ["Running workers demanded wages", "The government regulated banks"]
    batch = preprocess_many(texts)
    single = [preprocess(t) for t in texts]
    assert batch == single
