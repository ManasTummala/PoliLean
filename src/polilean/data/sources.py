"""HuggingFace dataset sources for political-leaning training data.

Sources
-------
allside : valurank/PoliticalBias_AllSides_Txt
    ~20k articles labeled left/right/center by AllSides editors. The repo
    stores one .txt file per article inside left/, right/, center/ folders of
    AllSides.zip; we download the zip and parse the folder structure.

semeval : SemEvalWorkshop/hyperpartisan_news_detection (bypublisher config)
    SemEval-2019 Task 4. 600k train articles with `bias` class labels
    (0=right, 1=right-center, 2=least, 3=left-center, 4=left). We keep the
    clear-cut labels (right, least, left) and drop the center-leaning
    variants (right-center, left-center) for a clean 3-class problem.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

# --- Local cache + label translation tables ---------------------------------
# ELI5: downloads live in ~/.cache/polilean so we fetch each corpus only once.
# The two tables below translate foreign label names into our three leans:
# folder names for the AllSides zip, numeric bias ids for SemEval.
CACHE_DIR = Path.home() / ".cache" / "polilean"

# ELI5: SemEval numbers its bias levels 0..4; we only keep the clear-cut
# ones (right/left/least -> centrist) and drop the mushy middle variants.
SEMEVAL_BIAS_NAMES = {0: "right", 1: "right-center", 2: "least", 3: "left-center", 4: "left"}
SEMEVAL_TO_LEAN = {"right": "right", "left": "left", "least": "centrist"}


# ELI5: AllSides ships files in folders named after their lean; map each
# folder name (case/space tolerant) onto left/right/centrist.
FOLDER_TO_LEAN = {
    "left": "left",
    "left data": "left",
    "right": "right",
    "right data": "right",
    "center": "centrist",
    "center data": "centrist",
    "least": "centrist",
}


# ELI5: one-line helper: folder name -> lean, or None if unrecognized.
def _to_lean(raw_label: str) -> str | None:
    return FOLDER_TO_LEAN.get(raw_label.strip().lower())


# ELI5: fetch AllSides.zip (once), open it like a lunchbox, and read every
# .txt file whose parent folder tells us the lean. Files that fail to decode
# or are empty are skipped. --max-samples splits the cap evenly across classes
# so one giant folder cannot eat the whole budget.
def load_allsides(max_samples: int | None = None) -> pd.DataFrame:
    """Load the AllSides article corpus (~20k articles) from valurank."""
    import requests

    zip_path = CACHE_DIR / "AllSides.zip"
    if not zip_path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        url = "https://huggingface.co/datasets/valurank/PoliticalBias_AllSides_Txt/resolve/main/AllSides.zip"
        print(f"Downloading AllSides.zip from {url} ...")
        resp = requests.get(url, timeout=600)
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)

    rows: list[dict[str, str]] = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            parts = Path(name).parts
            if len(parts) < 2 or not name.lower().endswith((".txt", ".TXT")):
                continue
            label = _to_lean(parts[-2].lower())
            if label is None:
                continue
            try:
                text = zf.read(name).decode("utf-8", errors="ignore")
            except Exception:
                continue
            if text.strip():
                rows.append({"text": text.strip(), "lean": label})

    df = pd.DataFrame(rows)
    if max_samples and len(df) > max_samples:
        # per-class quota so a single folder cannot consume the whole cap
        quota = max_samples // df["lean"].nunique()
        df = df.groupby("lean", group_keys=False).head(quota).reset_index(drop=True)
    if df.empty:
        raise ValueError("No articles parsed from AllSides.zip — archive layout may have changed.")
    print(f"AllSides: {len(df)} articles ({df['lean'].value_counts().to_dict()})")
    return df


# ELI5: pull SemEval-2019 news articles from HuggingFace, keep only the
# unambiguous bias labels (right/left/least), rename 'least' to 'centrist',
# and drop blank articles. trust_remote_code is required by this older
# dataset's loader script.
def load_semeval(max_samples: int | None = None) -> pd.DataFrame:
    """Load SemEval-2019 Task 4 bypublisher articles, keeping clear labels."""
    from datasets import load_dataset

    ds = load_dataset(
        "SemEvalWorkshop/hyperpartisan_news_detection",
        "bypublisher",
        split="train",
        trust_remote_code=True,
    )
    frames = []
    for raw_label in ("right", "left", "least"):
        keep_ids = [k for k, v in SEMEVAL_BIAS_NAMES.items() if v == raw_label]
        subset = ds.filter(lambda ex, ids=keep_ids: ex["bias"] in ids, num_proc=1)
        sub_df = subset.to_pandas()[["text", "bias"]]
        sub_df["lean"] = sub_df["bias"].map(lambda b: SEMEVAL_TO_LEAN[SEMEVAL_BIAS_NAMES[b]])
        frames.append(sub_df[["text", "lean"]])
        if max_samples:
            frames[-1] = frames[-1].head(max_samples // 3)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["text"].astype(str).str.strip() != ""].reset_index(drop=True)
    print(f"SemEval bypublisher: {len(df)} articles ({df['lean'].value_counts().to_dict()})")
    return df


# ELI5: a name->function phone book so `--source allsides` etc. can find
# the right loader without a big if/else ladder.
LOADERS = {
    "allsides": load_allsides,
    "semeval": load_semeval,
}


# ELI5: public entry point - look up the loader by name and run it.
def load_source(source: str, max_samples: int | None = None) -> pd.DataFrame:
    """Load one of the registered HF sources as a [text, lean] DataFrame."""
    if source not in LOADERS:
        raise ValueError(f"Unknown source '{source}'; available: {sorted(LOADERS)}")
    return LOADERS[source](max_samples)
