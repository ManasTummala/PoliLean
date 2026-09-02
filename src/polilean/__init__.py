"""PoliLean — political leaning analyzer for arbitrary text.

Pipeline: spaCy preprocessing -> TF-IDF features -> scikit-learn
classifier trained on labeled political texts.
"""

from polilean.model import PoliticalLeanClassifier

__version__ = "0.2.0"

__all__ = ["PoliticalLeanClassifier", "__version__"]
