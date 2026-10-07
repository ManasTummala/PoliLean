"""PoliLean — political leaning analyzer for arbitrary text.

Pipeline: spaCy preprocessing -> TF-IDF features -> scikit-learn
classifier trained on labeled political texts.
"""

# ELI5: the package's front door. Importing `polilean` gives you the main
# classifier class, the special "uncertain" abstain label, and the version.
from .model import UNCERTAIN, PoliticalLeanClassifier

__version__ = "0.2.0"

__all__ = ["PoliticalLeanClassifier", "UNCERTAIN", "__version__"]
