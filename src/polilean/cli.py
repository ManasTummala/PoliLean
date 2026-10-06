"""Command-line interface for PoliLean."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from polilean import __version__
from polilean.model import AVAILABLE_CLASSIFIERS, PoliticalLeanClassifier

# --- Command-line entry point -------------------------------------------------
# ELI5: this file turns user commands into actions. `polilean <thing> ...`
# offers three "things": train (teach the model), predict (analyze text),
# gui (open the web page). argparse below builds the menu of options.


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="polilean",
        description="Analyze the political leaning of a text (spaCy + scikit-learn).",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    # Sub-commands: everything after `polilean` picks which job to run.
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
    tp.add_argument("--axes-dataset", type=Path, default=None,
                    help="CSV with columns text,axis,label (default data/axes.csv)")
    tp.add_argument("--skip-axes", action="store_true",
                    help="Train only the lean classifier, no value axes.")

    # predict
    pp = sub.add_parser("predict", help="Predict the lean of a text.")
    pp.add_argument("text", nargs="*", help="Text to analyze (or omit to read stdin).")
    pp.add_argument("--model", type=Path, default=None, help="Path to trained model .pkl")
    pp.add_argument("--classifier", choices=list(AVAILABLE_CLASSIFIERS), default="logreg")
    pp.add_argument("--json", action="store_true", help="Emit JSON output.")
    pp.add_argument("--explain", action="store_true",
                    help="Show top n-grams that drove the prediction.")
    pp.add_argument("--file", type=Path, default=None, help="Read text from a file instead.")
    pp.add_argument("--threshold", type=float, default=None,
                    help="Abstain (return 'uncertain') when max probability is below "
                         "this value in [0,1]. Defaults to POLILEAN_ABSTAIN_THRESHOLD "
                         "if that env var is set.")

    # gui
    gp = sub.add_parser("gui", help="Launch the web GUI (percentile charts + axis radar).")
    gp.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1).")
    gp.add_argument("--port", type=int, default=8000, help="Port (default 8000).")
    gp.add_argument("--no-browser", action="store_true",
                    help="Do not open the browser automatically.")
    return p


# --- Training helpers --------------------------------------------------------
# ELI5: choose which CSV/network source to learn from. "csv" = the bundled
# seed file; "allside"/"semeval" = big corpora from HuggingFace; "both" =
# merge the bundled CSV with whatever network sources load successfully.
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


# ELI5: the `train` job. Load data -> fit the lean classifier -> print how
# accurate and how well-calibrated it is -> (unless --skip-axes) fit the five
# value-axis models and print their cross-validation scores -> save everything
# into one .pkl file that predict/gui/API later load.
def _cmd_train(args: argparse.Namespace) -> int:
    from polilean.axes import load_axes_dataset

    df = _load_training_frame(args)
    clf = PoliticalLeanClassifier(classifier=args.classifier)
    print(f"Training {args.classifier} on {len(df)} examples ...")
    metrics = clf.train(df, test_size=args.test_size)
    gap = metrics["train_accuracy"] - metrics["accuracy"]
    print(
        f"Accuracy: test={metrics['accuracy']:.3f}  "
        f"train={metrics['train_accuracy']:.3f}  "
        f"(memorization gap={gap:.3f}, train={metrics['train_size']}, "
        f"test={metrics['test_size']})"
    )
    if gap > 0.15:
        print(
            "  Note: large train/test gap = the model leans toward memorizing. "
            "Train on more data (--source both) to generalize better."
        )
    for lean, stats in sorted(metrics["report"].items()):
        if isinstance(stats, dict) and "f1-score" in stats:
            print(f"  {lean:<10} precision={stats['precision']:.3f} "
                  f"recall={stats['recall']:.3f} f1={stats['f1-score']:.3f}")
    print(
        f"Calibration: Brier={metrics['brier']:.4f}  ECE={metrics['ece']:.4f} "
        f"(0 = perfectly calibrated; lower is better)"
    )

    if args.skip_axes:
        print("Value axes: skipped (--skip-axes)")
    else:
        try:
            axes_df = load_axes_dataset(args.axes_dataset)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        axis_metrics = clf.train_axes(axes_df)
        print(f"Value axes trained ({len(axis_metrics)} of 5):")
        for axis, m in sorted(axis_metrics.items()):
            print(f"  {axis:<12} accuracy={m['accuracy']:.3f} "
                  f"brier={m['brier']:.3f} ece={m['ece']:.3f}")

    path = clf.save()
    print(f"Model saved to: {path}")
    return 0


# ELI5: the `predict` job. Read text from args/file/stdin, validate the
# abstain threshold, load the model, then print either JSON (--json) or a
# friendly report: lean + confidence bar + evidence + the value axes.
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

    threshold = args.threshold
    if threshold is None:
        env = os.environ.get("POLILEAN_ABSTAIN_THRESHOLD")
        if env:
            try:
                threshold = float(env)
            except ValueError:
                print(
                    f"Warning: ignoring invalid POLILEAN_ABSTAIN_THRESHOLD={env!r}",
                    file=sys.stderr,
                )
    if threshold is not None and not 0.0 <= threshold <= 1.0:
        print("Error: threshold must be between 0 and 1.", file=sys.stderr)
        return 2

    clf = PoliticalLeanClassifier(classifier=args.classifier, model_path=args.model)
    try:
        clf.load()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    pred = clf.predict(text, explain=args.explain, threshold=threshold)
    if args.json:
        print(json.dumps(pred.to_dict(), indent=2))
    else:
        print("=" * 60)
        print("  PoliLean - Political Leaning Analyzer")
        print("=" * 60)
        print(f"  Text         : {text[:70]}{'...' if len(text) > 70 else ''}")
        print(f"  Lean         : {pred.lean}")
        if pred.lean == "uncertain" and threshold is not None:
            print(f"  Abstained    : yes - confidence below {threshold:.0%} threshold")
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
        if pred.axes:
            print("  Value Axes (position: negative pole <- 0 -> positive pole):")
            for name, ax in pred.axes.items():
                print(f"    {name:<12}: {ax.label}  (position {ax.position:+.2f}, "
                      f"confidence {ax.confidence:.1%})")
                if args.explain:
                    top = ", ".join(
                        f"{f} ({w:+.2f})"
                        for f, w in ax.evidence.get(ax.label, [])[:4] if w != 0
                    )
                    if top:
                        print(f"        {ax.label}: {top}")
    return 0


# ELI5: the `gui` job. Warm-load the model (so the first click is fast),
# print the URL, optionally pop a browser tab after 1 second, then hand the
# process to uvicorn - a production web server - until Ctrl+C.
def _cmd_gui(args: argparse.Namespace) -> int:
    import threading
    import webbrowser

    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is required for the GUI (pip install uvicorn).", file=sys.stderr)
        return 1

    from polilean.api import app, get_classifier

    try:
        get_classifier()  # warm-load the model before the browser opens
    except RuntimeError as exc:
        print(f"Warning: {exc}", file=sys.stderr)

    url = f"http://{args.host}:{args.port}/"
    print(f"PoliLean GUI at {url}  (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    except KeyboardInterrupt:
        print("\nGUI stopped.")
    return 0


# --- Router ------------------------------------------------------------------
# ELI5: parse the command line, jump to the matching job, and fall back to
# printing help when no (or an unknown) command was given.
def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        return _cmd_train(args)
    if args.command == "predict":
        return _cmd_predict(args)
    if args.command == "gui":
        return _cmd_gui(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
