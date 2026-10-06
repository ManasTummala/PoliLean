"""Merge the extra seed CSVs (extremist + slang) into the main datasets.

ELI5: the bundled seed data used to contain only polite opinions, so the
model froze (returned "uncertain") on radical-sounding text and couldn't
decode internet slang. This script copies the balanced extremist rows from
data/extremist_lean.csv / data/extremist_axes.csv and the internet-slang
rows from data/slang_lean.csv / data/slang_axes.csv into data/train.csv and
data/axes.csv so every training run (CLI, Docker, tests) sees them. Rows
already present are skipped, so running it twice is safe.

The rows themselves live in the seed CSVs (a data file is the right home
for training sentences, and it keeps this script readable).

Run: python tools/extend_datasets.py
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
    # ELI5: both merges are idempotent - already-copied rows are detected by
    # their text and skipped, so CI/Docker can run this without duplicating.
    total, added = _merge("data/train.csv", "data/extremist_lean.csv", ["text"])
    print(f"train.csv: {total} rows ({added} extremist)")

    # Neutral rows: the SAME everyday sentence is labeled left, right, AND
    # centrist so those words cancel out in training - off-topic input then
    # scores near-uniform and abstains as Undefined instead of guessing.
    total, added = _merge("data/train.csv", "data/neutral_seed.csv", ["text"])
    print(f"train.csv: {total} rows ({added} neutral)")

    # Slang rows: internet register (tbh/fr/smh, based/cope, memes) so the
    # model can dissect online posts, not just formal prose. Includes
    # everyday-slang sentences under ALL 3 labels so non-political slang
    # cancels out and pure meme text abstains instead of guessing.
    total, added = _merge("data/train.csv", "data/slang_lean.csv", ["text"])
    print(f"train.csv: {total} rows ({added} slang)")
    print(pd.read_csv("data/train.csv")["lean"].value_counts().to_dict())

    total, added = _merge("data/axes.csv", "data/extremist_axes.csv", ["text"])
    print(f"axes.csv: {total} rows ({added} added)")

    total, added = _merge("data/axes.csv", "data/slang_axes.csv", ["text"])
    print(f"axes.csv: {total} rows ({added} slang)")


if __name__ == "__main__":
    main()
