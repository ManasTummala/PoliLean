# PoliLean

Political leaning analyzer for arbitrary text. Classifies text as
**left / right / centrist** *and* rates it on five **value axes** —
economic, social, authority, foreign policy, environment — each with a
signed position, a confidence percentage, and *evidence*: the exact
n-grams that drove the prediction. Built on a deliberately interpretable
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
# Seed CSV (549 examples, bundled - includes extremist, neutral, and slang rows)
polilean train --source csv

# AllSides corpus: ~17k news articles labeled by AllSides editors
polilean train --source allsides

# SemEval-2019 Task 4 (bypublisher): 600k labeled news articles
polilean train --source semeval

# Combine all available sources (recommended - this is what generalizes)
polilean train --source both

# Quick run with a per-class sample cap
polilean train --source allsides --max-samples 600
```

For the strongest model, `tools/train_full.py` combines the seed CSV with
both cached HuggingFace corpora, truncates articles to their first 1,500
characters (headline + lede carry the lean signal, and shorter texts are
harder to memorize), preprocesses once, and picks `C`/`min_df` by 3-fold
cross-validation:

```bash
python tools/train_full.py          # full run on AllSides + SemEval + seed
python tools/train_full.py --fast   # small smoke run
```

`train` prints a **memorization gap** (train accuracy − test accuracy):
a gap near zero means the model generalizes; a large gap means it is
remembering its study material. The size-aware `min_df` in
`_fit_pipeline` automatically prunes rare terms on large corpora so
one-off names cannot become shortcuts.

Source mapping: AllSides `left/right/center` and SemEval
`right/left/least` map to `left/right/centrist`; SemEval's mixed
`right-center`/`left-center` labels are dropped to keep classes clean.

### Extremist-language coverage

Polite-opinion-only training data makes the model abstain on radical
rhetoric. The seed datasets therefore include balanced extremist-language
rows (revolutionary/abolish rhetoric on the left, ethno-nationalist/
strongman rhetoric on the right, and neutral reporting *about* extremism
as centrist), plus extremist-flavored rows for every value axis. They live
in `data/extremist_lean.csv` and `data/extremist_axes.csv` and are merged
into the main CSVs by:

```bash
python tools/extend_datasets.py   # idempotent - safe to run twice
```

With the combined corpus, militant phrasing classifies with confidence
instead of returning Undefined.

### Internet-slang coverage

Real online political discourse is lowercase, abbreviated, and meme-flavored
(`tbh`, `fr`, `smh`, `based`, `cope`, `touch grass`, 💀). Formal-only seed
data makes the model miss or abstain on such posts, so the seed set also
includes balanced slang rows for every lean - left populism, right
culture-war slang, and both-sides chronically-online posts as centrist -
as well as slang-flavored rows for every value axis. Everyday (non-political)
slang sentences appear under **all three** labels so pure meme text with no
political signal cancels out and still returns Undefined instead of a guess.
They live in `data/slang_lean.csv` and `data/slang_axes.csv` and are merged
by the same idempotent `python tools/extend_datasets.py`. `tools/train_full.py`
prints a dedicated **slang probe battery** (plus slang off-topic probes that
must abstain) so regressions are visible in every training run.

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

## Value Axes

Beyond left/center/right, every prediction scores the text on five
independent value axes. Each axis is its own 3-class model (negative
pole / neutral / positive pole):

| Axis          | Negative pole    | Positive pole       | Measures |
|---------------|------------------|---------------------|----------|
| `economic`    | `market`         | `public`            | private/free-market vs state-led economy |
| `social`      | `traditionalist` | `progressive`       | social traditionalism vs progressivism |
| `authority`   | `authoritarian`  | `libertarian`       | strong-state authority vs individual liberty |
| `foreign`     | `nationalist`    | `internationalist`  | sovereignty-first vs international cooperation |
| `environment` | `growth`         | `green`             | growth-first vs environmental protection |

Each axis reports:

- **position** in `[-1, +1]` — `P(positive pole) − P(negative pole)`; `0` = balanced,
  `+1` = entirely at the positive pole
- **confidence** — probability of the dominant label (the percentage score)
- **probabilities** for all three classes, a **description**, and — with
  `--explain` — the top content-word evidence per class

```bash
polilean predict "Nationalize energy and fund universal healthcare" --explain
```

```
  Value Axes (position: negative pole <- 0 -> positive pole):
    economic    : public  (position +0.72, confidence 81.1%)
        public: healthcare (+0.20), funded (+0.17), energy (+0.15)
    environment : green   (position +0.12, confidence 44.4%)
    ...
```

JSON shape (abstention and axes compose: a below-threshold lean still
reports its axes):

```json
"axes": {
  "economic": {
    "label": "public",
    "position": 0.6211,
    "confidence": 0.7693,
    "probabilities": {"market": 0.1482, "neutral": 0.0825, "public": 0.7693},
    "description": "private/free-market economy vs public/state-led economy",
    "evidence": [{"feature": "healthcare", "weight": 0.2}]
  }
}
```

Training uses the bundled `data/axes.csv` (300 hand-written examples,
20 per class per axis) and happens automatically with the lean model:

```bash
polilean train                    # lean + all 5 axes
polilean train --skip-axes        # lean only
polilean train --axes-dataset path/to/custom_axes.csv
```

`train` reports 5-fold cross-validated accuracy and calibration per
axis (3-class chance = 0.33), then fits each final axis pipeline on all
rows:

```
Value axes trained (5 of 5):
  authority    accuracy=0.483 brier=0.610 ece=0.070
  economic     accuracy=0.550 brier=0.596 ece=0.076
  environment  accuracy=0.583 brier=0.555 ece=0.140
  foreign      accuracy=0.717 brier=0.521 ece=0.242
  social       accuracy=0.700 brier=0.508 ece=0.198
```

Axis models consume **raw** text (no spaCy cleanup): stop words and
negations carry stance signal — confirmed by cross-validation (raw
0.58 vs cleaned 0.41 accuracy). Stop words are still filtered out of
the *evidence* so explanations stay meaningful.

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

## Web GUI

A dependency-free **dark-themed** web GUI ships with the API — percentile
probability bars, a colored lean badge, diverging value-axis bars, and an
SVG radar chart of the five axes:

```bash
polilean gui                # serves http://127.0.0.1:8000/ and opens a browser
polilean gui --port 9000 --no-browser
```

Or open `http://localhost:8000/` when running the API (`uvicorn
polilean.api:app` or Docker). Type any text (or click an example),
optionally set the Undefined threshold (default **55%** — below it the
result is reported as **Undefined** rather than left/center/right, e.g.
for off-topic or too-short text), and press **Analyze** (Ctrl+Enter
works too). Results show:

- **Lean badge** — Left / Centrist / Right / **Undefined**, color-coded,
  with the top-class confidence as a percentage.
- **Probability bars** — percentile bars per lean class.
- **Value axes** — each axis (Economic, Social, Authority, Foreign
  Policy, Environment) shown as a diverging bar
  (`negative pole ← neutral → positive pole`) with position %, dominant
  label chip, and confidence %.
- **Radar chart** — the five axis positions plotted as a pentagon;
  distance from center = strength, dot color = pole direction.
- **Evidence** — the top feature chips (green = pushes toward, red =
  away) for the lean and each axis.

## HTTP API

The repo ships a FastAPI service:

```bash
uvicorn polilean.api:app --port 8000    # or: polilean gui
curl -X POST localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Cut taxes and secure the border.", "explain": true}'
```

Endpoints: `GET /` (web GUI), `GET /health`, `POST /predict`
(body: `{"text": ..., "explain": bool, "threshold"?: 0..1}` — omit
`threshold` to use `POLILEAN_ABSTAIN_THRESHOLD`, or neither to never
abstain). Responses carry `lean`, `confidence`, `probabilities`,
optional `evidence`, and `axes` (the value-axis scores).
Interactive docs at `http://localhost:8000/docs`.

## Docker

The image bakes in the dependencies, the spaCy model, **and a model
trained on the bundled CSVs**, so it works the moment it starts.

### Option A — compose (recommended)

```bash
docker compose up --build      # build the image and start the container
open http://localhost:8000/    # the dark web GUI
```

### Option B — plain docker

```bash
docker build -t polilean .            # cook the image (first run: a few minutes)
docker run --rm -p 8000:8000 polilean # start it; API + GUI on port 8000
open http://localhost:8000/
```

`-p 8000:8000` is the bridge: the server listens on port 8000 *inside*
the container, and that flag mirrors it to port 8000 on your machine.

### Day-to-day container commands

```bash
docker ps                          # is it running? (name: polilean-api)
docker logs -f polilean-api        # watch output; Ctrl+C stops watching, not the app
docker exec -it polilean-api bash  # open a shell INSIDE the container
docker stop polilean-api           # stop it   (compose: docker compose down)
```

### One-off commands inside the container

The image's default command starts the web server; putting your own
command after the image name replaces it:

```bash
# CLI prediction (model is already baked in)
docker run --rm polilean python -m polilean.cli predict "some text" --explain

# Run the test suite against the installed package
docker run --rm polilean python -m pytest tests/ -q
```

> The container runs as a non-root user (`appuser`), the trained model
> lives at `src/polilean/models/trained/logreg.pkl` inside the image, and
> the HuggingFace download cache is kept in named volumes so rebuilds do
> not re-download corpora.

## Architecture

```
src/polilean/
├── __init__.py        # Public API
├── preprocess.py      # spaCy pipeline (lemma + stop-word removal)
├── model.py           # TF-IDF + logreg/LinearSVC/NB, axes, evidence
├── axes.py            # value-axis specs + axes dataset loader
├── api.py             # FastAPI service (serves the web GUI at /)
├── cli.py             # train / predict / gui commands
├── static/index.html  # web GUI (percentile bars + radar chart)
├── data/
│   ├── dataset.py     # pandas CSV loading + validation
│   └── sources.py     # HF sources: AllSides + SemEval bypublisher
└── models/trained/    # saved .pkl models (lean + axes bundle)
data/train.csv         # seed training dataset (549 rows)
data/axes.csv          # value-axes seed dataset (365 rows)
data/slang_lean.csv    # internet-slang lean seed rows (merged by extend_datasets)
data/slang_axes.csv    # internet-slang value-axis seed rows
Dockerfile / docker-compose.yml
```

## Development

```bash
pip install -e ".[dev]"
pytest          # 59 tests
ruff check src tests
```

## License

MIT — see [LICENSE](LICENSE).
