"""Shared pytest fixtures.

ELI5: one trained model, shared by the whole test session so the suite
stays fast - see the trained_classifier fixture below.
"""

from __future__ import annotations

import pytest

from polilean.model import PoliticalLeanClassifier


# ELI5: this fixture is a shared "test brain". pytest builds it once at the
# start of the whole test session (scope="session") so individual tests do
# not each pay the seconds-long training cost. The tiny 18-sentence dataset
# below is balanced (6 left / 6 right / 6 centrist) so the classifier has
# something fair to learn, then the five value-axis models are trained on
# the real data/axes.csv and every test reuses the finished object.
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
    from polilean.axes import load_axes_dataset

    clf.train_axes(load_axes_dataset())
    return clf
