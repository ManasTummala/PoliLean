"""PoliLean — political leaning analyzer for arbitrary text.
"""

from .model import UNCERTAIN, PoliticalLeanClassifier

__version__ = "1.0.1"

__all__ = ["PoliticalLeanClassifier", "UNCERTAIN", "__version__"]
