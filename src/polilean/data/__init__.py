"""Data loading and dataset utilities (pandas-based)."""

from polilean.data.dataset import (
    DATA_DIR,
    DEFAULT_DATASET_PATH,
    load_dataset,
    train_test_split_frame,
)

__all__ = [
    "DATA_DIR",
    "DEFAULT_DATASET_PATH",
    "load_dataset",
    "train_test_split_frame",
]
