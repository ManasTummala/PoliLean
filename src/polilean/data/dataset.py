"""Dataset loading utilities backed by pandas."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# --- Default dataset location -------------------------------------------------
# ELI5: walk up from this file (data/ -> polilean/ -> src/ -> project root)
# so "data/train.csv" is found no matter where the command is run.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DATASET_PATH = DATA_DIR / "train.csv"

# The only three lean labels we accept anywhere in the pipeline.
VALID_LEANS = {"left", "right", "centrist"}

# ELI5: read the training CSV into a pandas table and refuse anything weird:
# missing columns, blank rows, or lean words outside {left, right, centrist}.
# Normalizing (trim + lowercase) happens here so one typo cannot poison
# training. Returns a tidy DataFrame with columns [text, lean].
def load_dataset(path: Path | str | None = None) -> pd.DataFrame:
    """Load a labeled dataset with columns [text, lean]."""
    path = Path(path) if path else DEFAULT_DATASET_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at '{path}'.")
    df = pd.read_csv(path)
    required = {"text", "lean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    df["lean"] = df["lean"].str.strip().str.lower()
    bad = set(df["lean"].unique()) - VALID_LEANS
    if bad:
        raise ValueError(f"Invalid lean labels {sorted(bad)}; must be one of {sorted(VALID_LEANS)}")
    return df.dropna(subset=["text", "lean"]).reset_index(drop=True)


# ELI5: cut a table into "study part" and "exam part" fairly - stratify=True
# keeps the same left/right/centrist ratio in both halves. random_state makes
# the split reproducible (same shuffle every run).
def train_test_split_frame(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Deterministic stratified split of the DataFrame (for data inspection)."""
    from sklearn.model_selection import train_test_split

    return train_test_split(df, test_size=test_size, random_state=random_state, stratify=df["lean"])
