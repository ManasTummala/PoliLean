# PoliLean

Political leaning analyzer for arbitrary text. Classifies text as
**left / right / centrist** with confidence scores and *evidence* — the exact
n-grams that drove each prediction. Built on a deliberately interpretable
classical-ML stack: **spaCy** preprocessing, **scikit-learn** classification,
**pandas** data handling.

## Quick Start

```bash
pip install -e .
python -m spacy download en_core_web_sm

# Train on the seed dataset
polilean train

# Analyze a string (with evidence)
polilean predict "We must tax the wealthy to fund universal healthcare." --explain
```

## Training Data Sources

```bash
# Seed CSV (300 examples, bundled)
polilean train --source csv

# AllSides corpus: ~17k news articles labeled by AllSides editors
polilean train --source allsides

# SemEval-2019 Task 4 (bypublisher): 600k labeled news articles
polilean train --source semeval

# Combine all available sources
polilean train --source both

# Quick run with a per-class sample cap
polilean train --source allsides --max-samples 600
```

Source mapping: AllSides `left/right/center` and SemEval
`right/left/least` map to `left/right/centrist`; SemEval's mixed
`right-center`/`left-center` labels are dropped to keep classes clean.

## Classifiers

```bash
polilean train --classifier logreg       # LogisticRegression (default; probabilities)
polilean train --classifier linear_svc   # LinearSVC (fast, exact coefficients)
polilean train --classifier naive_bayes  # MultinomialNB
```

## Explainability

Every prediction can show which features pushed it toward each lean:

```bash
polilean predict "We must tax the wealthy to fund universal healthcare." --explain --json
```

```json
{
  "lean": "left",
  "confidence": 0.73,
  "probabilities": {"left": 0.73, "centrist": 0.15, "right": 0.12},
  "evidence": {
    "left":   [{"feature": "universal healthcare", "weight": 0.22}, ...],
    "right":  [{"feature": "healthcare", "weight": -0.16}, ...]
  }
}
```

This is the core reason for choosing TF-IDF + linear models: for a system
that makes claims about political leaning, you can always answer *why*.

## Calibration & Abstention

`train` reports two calibration metrics on the held-out test set — how
much you can trust the confidence scores themselves:

- **Brier score** — mean squared error between the predicted class
  probabilities and the outcome (multiclass; 0 = perfect, max 2.0).
- **ECE** — expected calibration error over 10 equal-width confidence
  bins (0 = perfect, max 1.0).

Both are lower-is-better:

```
Calibration: Brier=0.0482  ECE=0.1715 (0 = perfectly calibrated; lower is better)
```

When a forced label is worse than no label, let the model abstain: any
prediction whose confidence falls below the threshold returns
`"uncertain"` instead. Raw probabilities and evidence are still
returned, so callers can see the near-tie:

```bash
# Abstain below 80% confidence
polilean predict "some ambiguous text" --threshold 0.8

# Default threshold for every predict call (CLI and API)
export POLILEAN_ABSTAIN_THRESHOLD=0.8
```

```json
{
  "lean": "uncertain",
  "confidence": 0.647,
  "probabilities": {"left": 0.647, "centrist": 0.201, "right": 0.152}
}
```

Programmatically: `clf.predict(text, threshold=0.8)` (or pass
`threshold=` to the constructor). `threshold=0.0` always emits a label.

## HTTP API + Docker

The repo ships a FastAPI service and a Docker image with a model baked in:

```bash
docker compose up --build          # API on http://localhost:8000

curl -X POST localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Cut taxes and secure the border.", "explain": true}'

# Or one-off CLI inside the container
docker run --rm polilean predict "some text" --explain
```

Endpoints: `GET /health`, `POST /predict`
(body: `{"text": ..., "explain": bool, "threshold"?: 0..1}` — omit
`threshold` to use `POLILEAN_ABSTAIN_THRESHOLD`, or neither to never
abstain).
Interactive docs at `http://localhost:8000/docs`.

## Architecture

```
src/polilean/
├── __init__.py        # Public API
├── preprocess.py      # spaCy pipeline (lemma + stop-word removal)
├── model.py           # TF-IDF + logreg/LinearSVC/NB, evidence extraction
├── api.py             # FastAPI service
├── cli.py             # train / predict commands
├── data/
│   ├── dataset.py     # pandas CSV loading + validation
│   └── sources.py     # HF sources: AllSides + SemEval bypublisher
└── models/trained/    # saved .pkl models
data/train.csv         # seed training dataset
Dockerfile / docker-compose.yml
```

## Development

```bash
pip install -e ".[dev]"
pytest          # 42 tests
ruff check src tests
```

## License

MIT — see [LICENSE](LICENSE).
