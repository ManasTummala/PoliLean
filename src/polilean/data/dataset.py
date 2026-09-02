"""Dataset loading utilities backed by pandas."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DATASET_PATH = DATA_DIR / "train.csv"

VALID_LEANS = {"left", "right", "centrist"}


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


def train_test_split_frame(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Deterministic stratified split of the DataFrame (for data inspection)."""
    from sklearn.model_selection import train_test_split

    return train_test_split(df, test_size=test_size, random_state=random_state, stratify=df["lean"])
