"""Value-axis definitions and dataset loading.

Beyond the left/center/right lean, PoliLean rates text on independent
value axes: economic, social, authority, foreign policy, and
environment. Each axis is trained as a three-way classification
(negative pole / neutral / positive pole); the signed position reported
at prediction time is derived from the class probabilities:

    position = P(positive pole) - P(negative pole)   in [-1, +1]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# --- Where the data files live ---------------------------------------------
# ELI5: these three lines just build file paths. parents[2] walks up from
# this file's folder to the project root, so "data/axes.csv" resolves the
# same way no matter where you run PoliLean from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_AXES_PATH = DATA_DIR / "axes.csv"

# The middle class label: a text can also sit on the fence (neutral).
NEUTRAL = "neutral"

REQUIRED_COLUMNS = {"text", "axis", "label"}


@dataclass(frozen=True)
class AxisSpec:
    """One bipolar value axis."""

    name: str
    negative: str  # class label at the negative pole
    positive: str  # class label at the positive pole
    description: str

    @property
    def labels(self) -> set[str]:
        return {self.negative, NEUTRAL, self.positive}


# --- The five value axes (the heart of the feature) ------------------------
# ELI5: every axis is a two-sided argument with a middle option, like a
# seesaw: market (private) <-> neutral <-> public (state), etc. Each entry
# says the axis name, the word for each end, and a plain-English description
# shown in the GUI. The dict keys double as the CSV's "axis" column values.
AXES: dict[str, AxisSpec] = {
    "economic": AxisSpec(
        name="economic",
        negative="market",
        positive="public",
        description="private/free-market economy vs public/state-led economy",
    ),
    "social": AxisSpec(
        name="social",
        negative="traditionalist",
        positive="progressive",
        description="traditionalist vs progressive on social issues",
    ),
    "authority": AxisSpec(
        name="authority",
        negative="authoritarian",
        positive="libertarian",
        description="strong-state authority vs individual liberty",
    ),
    "foreign": AxisSpec(
        name="foreign",
        negative="nationalist",
        positive="internationalist",
        description="national sovereignty vs international cooperation",
    ),
    "environment": AxisSpec(
        name="environment",
        negative="growth",
        positive="green",
        description="economic-growth-first vs environmental protection",
    ),
}


# --- Load + sanity-check the axes training CSV ------------------------------
# ELI5: read the CSV into a pandas table and refuse anything weird: missing
# columns (text/axis/label), unknown axis names, or labels that do not belong
# to that axis (e.g. "green" on the economic axis). Failing loudly here beats
# training on garbage later. Returns a tidy DataFrame ready for train_axes().
def load_axes_dataset(path: Path | str | None = None) -> pd.DataFrame:
    """Load the value-axes dataset with columns [text, axis, label]."""
    path = Path(path) if path else DEFAULT_AXES_PATH
    if not path.exists():
        raise FileNotFoundError(f"Axes dataset not found at '{path}'.")
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Axes dataset is missing required columns: {sorted(missing)}")
    df = df.dropna(subset=list(REQUIRED_COLUMNS)).copy()
    df["axis"] = df["axis"].str.strip().str.lower()
    df["label"] = df["label"].str.strip().str.lower()

    bad_axes = set(df["axis"].unique()) - set(AXES)
    if bad_axes:
        raise ValueError(
            f"Unknown axes {sorted(bad_axes)}; must be one of {sorted(AXES)}"
        )
    for axis, spec in AXES.items():
        bad_labels = set(df.loc[df["axis"] == axis, "label"].unique()) - spec.labels
        if bad_labels:
            raise ValueError(
                f"Invalid labels {sorted(bad_labels)} for axis '{axis}'; "
                f"must be one of {sorted(spec.labels)}"
            )
    return df.reset_index(drop=True)
