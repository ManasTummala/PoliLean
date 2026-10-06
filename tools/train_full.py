"""Train PoliLean on the real news corpora (AllSides + SemEval + seed CSV).

ELI5: the bundled ~500-example CSV is too small, so the model memorizes it.
This script builds a much bigger, balanced training set from the cached
HuggingFace corpora, cleans the text ONCE (spaCy is the slow part - the
result is cached in .cache/), and sweeps hyperparameters with 3-fold
cross-validation to find settings that GENERALIZE instead of memorize.

Anti-memorization measures:
  * articles truncated to their first MAX_CHARS (headline+lede carry the
    lean signal; bodies full of one-off names only invite memorization)
  * size-aware min_df prunes terms seen in <N documents
  * the short seed-opinion rows are UP-SAMPLED inside each training fold
    only (never in the validation fold), so the model keeps its feel for
    short opinion texts without the CV score becoming a lie
  * the final holdout split is never up-sampled - reported numbers stay honest

Run:  python tools/train_full.py            (full run)
      python tools/train_full.py --fast     (quick smoke run)
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polilean.data.dataset import load_dataset  # noqa: E402
from polilean.data.sources import load_allsides, load_semeval  # noqa: E402
from polilean.model import PoliticalLeanClassifier, calibration_metrics  # noqa: E402
from polilean.preprocess import preprocess_many  # noqa: E402

# News articles are long; the lean signal sits mostly in the headline+lede,
# and truncating cuts spaCy time ~4x while reducing memorization of article
# bodies (names, one-off events).
MAX_CHARS = 1500

# Repeat each seed row this many times inside training folds. The seed set is
# only ~2% of the raw corpus; without this its short-opinion style gets drowned
# out by news prose and the model loses its grip on short texts.
SEED_UPSAMPLE = 8

CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache"


def build_corpus(fast: bool) -> pd.DataFrame:
    """Combine seed CSV + AllSides + SemEval into one balanced frame.

    A ``source`` column marks which rows came from the small seed set so the
    training step can up-sample exactly those rows.
    """
    seed = load_dataset()
    seed["source"] = "seed"
    frames = [seed]
    caps = (1500, 600) if fast else (9000, 9000)
    t0 = time.time()
    sides = load_allsides(max_samples=caps[0])
    sides["source"] = "allsides"
    sem = load_semeval(max_samples=caps[1])
    sem["source"] = "semeval"
    frames += [sides, sem]
    df = pd.concat(frames, ignore_index=True)
    df["text"] = df["text"].astype(str).str.slice(0, MAX_CHARS)
    # Dedup on (text, lean), NOT text alone: neutral seed rows repeat the same
    # sentence under all 3 labels on purpose (that is what cancels their lean
    # signal), and text-only dedup would keep just the first label.
    df = df.dropna().drop_duplicates(subset=["text", "lean"]).reset_index(drop=True)
    print(f"corpus: {len(df)} docs in {time.time() - t0:.0f}s "
          f"{df['lean'].value_counts().to_dict()} "
          f"(seed rows: {int((df['source'] == 'seed').sum())})", flush=True)
    return df


def clean_once(df: pd.DataFrame) -> list[str]:
    """spaCy-clean every doc exactly once, cached in .cache/.

    ELI5: cleaning is the 5-minute part, so we save the result keyed by a
    fingerprint of the inputs - reruns with unchanged data skip it entirely.
    """
    fingerprint_src = "|".join([
        str(MAX_CHARS),
        str(len(df)),
        hashlib.md5("".join(df["text"]).encode(), usedforsecurity=False).hexdigest(),
    ])
    digest = hashlib.md5(fingerprint_src.encode(), usedforsecurity=False).hexdigest()[:16]
    cache_path = CACHE_DIR / f"clean_{digest}.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as fh:
            cleaned = pickle.load(fh)
        print(f"preprocessed {len(cleaned)} docs (cached)", flush=True)
        return cleaned
    t0 = time.time()
    cleaned = preprocess_many(df["text"].tolist())
    print(f"preprocessed {len(cleaned)} docs in {time.time() - t0:.0f}s", flush=True)
    CACHE_DIR.mkdir(exist_ok=True)
    with cache_path.open("wb") as fh:
        pickle.dump(cleaned, fh)
    return cleaned


def fold_train(
    texts: list[str], labels: list[str], is_seed: list[bool],
    c: float, min_df: int,
):
    """Build one training pipeline, up-sampling seed rows beforehand."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    up_texts = list(texts)
    up_labels = list(labels)
    for t, y, s in zip(texts, labels, is_seed, strict=True):
        if s:
            up_texts.extend([t] * (SEED_UPSAMPLE - 1))
            up_labels.extend([y] * (SEED_UPSAMPLE - 1))
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=c)),
    ])
    pipe.fit(up_texts, up_labels)
    return pipe


def sweep(
    cleaned: list[str], labels: list[str], is_seed: list[bool], fast: bool,
) -> dict:
    """3-fold CV over regularization (C) x noise pruning (min_df).

    ELI5: try a few settings, grade each on a fold it has NOT seen, and keep
    whichever scores best on unseen text - that is the anti-memorization test.
    Seed up-sampling happens INSIDE each training fold only, so validation
    folds stay untouched and the score cannot be inflated by duplicates.
    """
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import StratifiedKFold

    grid = [(1.0, 5), (4.0, 5), (4.0, 10), (1.0, 10)] if not fast else [(4.0, 5)]
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    best, best_score = None, -1.0
    print(f"{'C':>5} {'min_df':>7} {'cv_acc':>7}", flush=True)
    for c, min_df in grid:
        accs = []
        for tr, va in skf.split(cleaned, labels):
            pipe = fold_train(
                [cleaned[i] for i in tr],
                [labels[i] for i in tr],
                [is_seed[i] for i in tr],
                c, min_df,
            )
            pred = pipe.predict([cleaned[i] for i in va])
            accs.append(accuracy_score([labels[i] for i in va], pred))
        mean = float(np.mean(accs))
        print(f"{c:>5} {min_df:>7} {mean:>7.3f}", flush=True)
        if mean > best_score:
            best, best_score = {"c": c, "min_df": min_df}, mean
    print(f"best: C={best['c']} min_df={best['min_df']} cv_acc={best_score:.3f}", flush=True)
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description="train on real corpora")
    ap.add_argument("--fast", action="store_true", help="small smoke run")
    args = ap.parse_args()

    df = build_corpus(args.fast)
    cleaned = clean_once(df)
    labels = df["lean"].astype(str).tolist()
    is_seed = (df["source"] == "seed").tolist()

    best = sweep(cleaned, labels, is_seed, args.fast)

    # --- honest holdout: split FIRST, up-sample train side only -------------
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(cleaned))
    tr_idx, te_idx = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=labels
    )
    clf = PoliticalLeanClassifier(classifier="logreg")
    clf.pipeline = fold_train(
        [cleaned[i] for i in tr_idx],
        [labels[i] for i in tr_idx],
        [is_seed[i] for i in tr_idx],
        best["c"], best["min_df"],
    )

    from sklearn.metrics import classification_report

    x_te = [cleaned[i] for i in te_idx]
    y_te = [labels[i] for i in te_idx]
    x_tr = [cleaned[i] for i in tr_idx]
    y_tr = [labels[i] for i in tr_idx]
    test_acc = float(np.mean(clf.pipeline.predict(x_te) == np.asarray(y_te)))
    # train score on the ORIGINAL (non-up-sampled) train rows: comparable
    # to the test score, so the printed gap is meaningful.
    train_acc = float(np.mean(clf.pipeline.predict(x_tr) == np.asarray(y_tr)))
    classes, proba = clf._predict_proba(x_te)
    cal = calibration_metrics(y_te, proba, classes)
    print(f"\nholdout accuracy : {test_acc:.3f}")
    print(f"train accuracy   : {train_acc:.3f} (memorization gap={train_acc - test_acc:.3f})")
    print(f"calibration      : Brier={cal['brier']:.4f} ECE={cal['ece']:.4f}")
    print(classification_report(y_te, clf.pipeline.predict(x_te), zero_division=0))

    print("training value axes ...", flush=True)
    from polilean.axes import load_axes_dataset

    axis_metrics = clf.train_axes(load_axes_dataset())
    for axis, m in sorted(axis_metrics.items()):
        print(f"  {axis:<12} accuracy={m['accuracy']:.3f}")

    path = clf.save()
    print(f"saved: {path}")

    # Extremist-language probes: must classify, not abstain.
    probes = [
        ("Seize the means of production by force and end capitalist exploitation "
         "in violent revolution", "left"),
        ("Round up every illegal alien and deport them tomorrow, no exceptions", "right"),
        ("Hang the traitors who stole the election, revolution is coming", "right"),
        ("Expropriate every factory without compensation, private ownership must die", "left"),
        ("News analysis: researchers warn extremist rhetoric rises on both flanks", "centrist"),
    ]
    print("\nextremist probes (threshold 0.55):")
    misses = 0
    for text, expected in probes:
        p = clf.predict(text, threshold=0.55, explain=False)
        ok = p.lean == expected
        misses += not ok
        print(f"  {'OK  ' if ok else 'MISS'} {p.lean:<9} {p.confidence:.2f} (expected {expected})")

    # Slang probes: internet register must classify like formal text does.
    slang_probes = [
        ("billionaires are hoarding while we can't pay rent, eat the rich fr 💀", "left"),
        ("based and redpilled, close the borders already and enforce the laws", "right"),
        ("both sides are chronically online and need to touch grass imo", "centrist"),
        ("free college pls, education shouldn't be a debt sentence tbh", "left"),
        ("drain the swamp already, the deep state runs everything lol", "right"),
    ]
    print("\nslang probes (threshold 0.55):")
    for text, expected in slang_probes:
        p = clf.predict(text, threshold=0.55, explain=False)
        ok = p.lean == expected
        misses += not ok
        print(f"  {'OK  ' if ok else 'MISS'} {p.lean:<9} {p.confidence:.2f} (expected {expected})")

    # Off-topic probes: must still abstain (formal AND slang non-political).
    print("\noff-topic probes (threshold 0.55, want uncertain):")
    for text in ("The weather is nice today and we walked the dog.",
                 "This movie was great, we watched it last night",
                 "ngl this weather has me staying inside all weekend",
                 "my wifi keeps lagging and im about to lose it fr"):
        p = clf.predict(text, threshold=0.55, explain=False)
        ok = p.lean == "uncertain"
        misses += not ok
        print(f"  {'OK  ' if ok else 'MISS'} {p.lean:<9} {p.confidence:.2f}")
    print(f"\nmisses: {misses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
