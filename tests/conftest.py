"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from polilean.model import PoliticalLeanClassifier


@pytest.fixture(scope="session")
def trained_classifier(tmp_path_factory) -> PoliticalLeanClassifier:
    """Train once per test session on a small synthetic dataset."""
    import pandas as pd

    df = pd.DataFrame({
        "text": [
            "raise taxes on the wealthy to fund welfare",
            "universal healthcare and free college for everyone",
            "strong unions protect the working class",
            "cut taxes and deregulate the free market",
            "border security and law and order now",
            "private enterprise drives innovation best",
            "we need bipartisan compromise on the budget",
            "evidence based policy beats partisan rhetoric",
            "sensible regulation balanced with growth",
            "increase the minimum wage for workers",
            "expand social safety net programs now",
            "nationalize the energy companies",
            "lower government spending and reduce debt",
            "gun rights protect law abiding citizens",
            "privatize services for efficiency",
            "practical moderate solutions work best",
            "compromise between both parties needed",
            "balanced approach to reform is wise",
        ],
        "lean": ["left"] * 6 + ["right"] * 6 + ["centrist"] * 6,
    })
    clf = PoliticalLeanClassifier(classifier="logreg")
    clf.train(df, test_size=0.25, random_state=1)
    return clf
