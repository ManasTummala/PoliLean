"""Command-line interface for PoliLean."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from polilean import __version__
from polilean.model import AVAILABLE_CLASSIFIERS, PoliticalLeanClassifier


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="polilean",
        description="Analyze the political leaning of a text (spaCy + scikit-learn).",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command")

    # train
    tp = sub.add_parser("train", help="Train the classifier on a dataset.")
    tp.add_argument("--source", default="csv",
                    help="csv (data/train.csv), allsides (HF valurank AllSides), "
                         "semeval (HF SemEval-2019 bypublisher)")
    tp.add_argument("--dataset", type=Path, default=None,
                    help="CSV path when --source=csv (default data/train.csv)")
    tp.add_argument("--max-samples", type=int, default=None,
                    help="Cap samples per source (useful for quick runs)")
    tp.add_argument("--classifier", choices=list(AVAILABLE_CLASSIFIERS), default="logreg")
    tp.add_argument("--test-size", type=float, default=0.2)

    # predict
    pp = sub.add_parser("predict", help="Predict the lean of a text.")
    pp.add_argument("text", nargs="*", help="Text to analyze (or omit to read stdin).")
    pp.add_argument("--model", type=Path, default=None, help="Path to trained model .pkl")
    pp.add_argument("--classifier", choices=list(AVAILABLE_CLASSIFIERS), default="logreg")
    pp.add_argument("--json", action="store_true", help="Emit JSON output.")
    pp.add_argument("--explain", action="store_true",
                    help="Show top n-grams that drove the prediction.")
    pp.add_argument("--file", type=Path, default=None, help="Read text from a file instead.")
    return p


def _load_training_frame(args: argparse.Namespace):
    import pandas as pd

    from polilean.data.dataset import load_dataset
    from polilean.data.sources import load_source

    if args.source == "csv":
        return load_dataset(args.dataset)
    if args.source == "both":
        frames = [load_dataset()]
        for name in ("allsides", "semeval"):
            try:
                frames.append(load_source(name, args.max_samples))
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"Warning: could not load source '{name}': {exc}", file=sys.stderr)
        return pd.concat(frames, ignore_index=True)
    return load_source(args.source, args.max_samples)


def _cmd_train(args: argparse.Namespace) -> int:
    df = _load_training_frame(args)
    clf = PoliticalLeanClassifier(classifier=args.classifier)
    print(f"Training {args.classifier} on {len(df)} examples ...")
    metrics = clf.train(df, test_size=args.test_size)
    path = clf.save()
    print(
        f"Accuracy: {metrics['accuracy']:.3f}  "
        f"(train={metrics['train_size']}, test={metrics['test_size']})"
    )
    for lean, stats in sorted(metrics["report"].items()):
        if isinstance(stats, dict) and "f1-score" in stats:
            print(f"  {lean:<10} precision={stats['precision']:.3f} "
                  f"recall={stats['recall']:.3f} f1={stats['f1-score']:.3f}")
    print(f"Model saved to: {path}")
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    if args.file:
        text = args.file.read_text(encoding="utf-8")
    elif args.text:
        text = " ".join(args.text)
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        print("Error: provide text, --file, or pipe text on stdin.", file=sys.stderr)
        return 2

    clf = PoliticalLeanClassifier(classifier=args.classifier, model_path=args.model)
    try:
        clf.load()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    pred = clf.predict(text, explain=args.explain)
    if args.json:
        print(json.dumps(pred.to_dict(), indent=2))
    else:
        print("=" * 60)
        print("  PoliLean - Political Leaning Analyzer")
        print("=" * 60)
        print(f"  Text         : {text[:70]}{'...' if len(text) > 70 else ''}")
        print(f"  Lean         : {pred.lean}")
        print(f"  Confidence   : {pred.confidence:.1%}")
        print("  Probabilities:")
        for lean, p in sorted(pred.probabilities.items(), key=lambda kv: -kv[1]):
            bar = "#" * int(p * 30)
            print(f"    {lean:<10}: {p:6.1%}  {bar}")
        if pred.evidence:
            print("  Evidence (top n-grams pushing toward each lean):")
            for lean, feats in pred.evidence.items():
                top = ", ".join(f"{f} ({w:+.2f})" for f, w in feats[:5] if w != 0)
                if top:
                    print(f"    {lean:<10}: {top}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        return _cmd_train(args)
    if args.command == "predict":
        return _cmd_predict(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
