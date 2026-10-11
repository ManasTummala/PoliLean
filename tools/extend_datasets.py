"""
Merge new seed CSVs into the main datasets
"""

from __future__ import annotations

import pandas as pd


def _merge(target: str, seed: str, keys: list[str]) -> tuple[int, int]:
    """Append rows from ``seed`` CSV not already in ``target`` CSV."""
    dst = pd.read_csv(target)
    src = pd.read_csv(seed)
    present = set(dst[keys[0]].astype(str).str.strip())
    fresh = src[~src[keys[0]].astype(str).str.strip().isin(present)]
    if len(fresh):
        pd.concat([dst, fresh], ignore_index=True).to_csv(target, index=False)
        dst = pd.read_csv(target)
    return len(dst), len(fresh)


def main() -> None:
    # both merges are idempotent - already-copied rows are detected by their text and skipped, so CI/Docker can run this without duplicating.
    total, added = _merge("data/train.csv", "data/extremist_lean.csv", ["text"])
    print(f"train.csv: {total} rows ({added} extremist)")

    total, added = _merge("data/train.csv", "data/neutral_seed.csv", ["text"])
    print(f"train.csv: {total} rows ({added} neutral)")

    total, added = _merge("data/train.csv", "data/slang_lean.csv", ["text"])
    print(f"train.csv: {total} rows ({added} slang)")
    print(pd.read_csv("data/train.csv")["lean"].value_counts().to_dict())

    total, added = _merge("data/axes.csv", "data/extremist_axes.csv", ["text"])
    print(f"axes.csv: {total} rows ({added} added)")

    total, added = _merge("data/axes.csv", "data/slang_axes.csv", ["text"])
    print(f"axes.csv: {total} rows ({added} slang)")


if __name__ == "__main__":
    main()
