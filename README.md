# PoliLean

**Political leaning analyzer for arbitrary text.** Classifies text as
**left / right / centrist** *and* rates it on five **value axes** —
economic, social, authority, foreign policy, environment — each with a
signed position, a confidence percentage, and *evidence*: the exact
n-grams that drove the prediction.

Built on a deliberately interpretable classical-ML stack: **spaCy**
preprocessing, **scikit-learn** classification, **pandas** data handling.
For a system that makes claims about political leaning, you can always
answer *why*.

[![CI](https://github.com/ManasTummala/PoliLean/actions/workflows/ci.yml/badge.svg)](https://github.com/ManasTummala/PoliLean/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [CLI](#cli)
  - [Web GUI](#web-gui)
  - [HTTP API](#http-api)
  - [Docker](#docker)
- [Deploying to Vercel](#deploying-to-vercel)
- [Value Axes](#value-axes)
- [Calibration & Abstention](#calibration--abstention)
- [Explainability](#explainability)
- [Training Data](#training-data)
  - [Sources](#sources)
  - [Extremist-language coverage](#extremist-language-coverage)
  - [Internet-slang coverage](#internet-slang-coverage)
- [Classifiers](#classifiers)
- [Development](#development)
- [Project Structure](#project-structure)
- [Generative AI Statement](#generative-ai-statement)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Three-way lean classification** — left / right / centrist with per-class
  probabilities.
- **Five value axes** — economic, social, authority, foreign, environment;
  each an independent 3-class model with a signed position in `[-1, +1]`.
- **Evidence for every prediction** — the top n-grams that pushed the model
  toward (and away from) each class, for the lean *and* every axis.
- **Calibrated confidence + abstention** — Brier/ECE reporting, and a
  configurable confidence threshold below which the model returns
  `uncertain` instead of a forced label.
- **Robust to real-world text** — training data covers extremist rhetoric
  and internet slang (`tbh`, `based`, `touch grass`, 💀), so militant or
  meme-flavored posts classify instead of abstaining.
- **Three interfaces** — CLI (`polilean`), a dependency-free dark-themed
  web GUI with radar chart, and a FastAPI HTTP service — deployable with
  Docker or serverless on Vercel.
- **Interpretable by construction** — TF-IDF + linear models; no black box.

## Tech Stack

Every framework, tool, and library used by PoliLean, and what it is for.

### Runtime — language & core ML

| Technology | Version | Role |
|---|---|---|
| [Python](https://www.python.org/) | 3.10–3.13 | implementation language (3.13 pinned for Vercel) |
| [spaCy](https://spacy.io/) | `>=3.8,<3.9` | tokenization, lemmatization, stop-word removal |
| [`en_core_web_sm`](https://github.com/explosion/spacy-models) | 3.8.0 | spaCy English pipeline (installed automatically as a pinned dependency) |
| [scikit-learn](https://scikit-learn.org/) | `>=1.7.2,<2` | TF-IDF features, classifiers, pipelines |
| [NumPy](https://numpy.org/) | `>=1.26` | probability arrays and numeric plumbing |
| [pandas](https://pandas.pydata.org/) | `>=2.0` | CSV datasets and label tables |
| [FastAPI](https://fastapi.tiangolo.com/) | `>=0.111` | HTTP API (`/`, `/health`, `/predict`) |
| [Pydantic](https://docs.pydantic.dev/) | v2 (transitive) | request/response schemas and validation |
| [Starlette](https://www.starlette.io/) | transitive | ASGI framework underneath FastAPI |
| [Uvicorn](https://www.uvicorn.org/) | `>=0.30` | ASGI server |

scikit-learn estimators used: `TfidfVectorizer`, `LogisticRegression`
(default), `LinearSVC`, `MultinomialNB`, `Pipeline`. Model persistence
uses the standard-library `pickle` (plus `argparse`, `functools`,
`pathlib`, `zipfile`, `re`, `json` from the standard library).

### Data acquisition — optional `train` extra

| Technology | Version | Role |
|---|---|---|
| [HuggingFace Datasets](https://huggingface.co/docs/datasets) | `>=2.19,<3.0` | streams the AllSides and SemEval-2019 corpora |
| [requests](https://requests.readthedocs.io/) | `>=2.31` | downloads the AllSides zip corpus |

Install with `pip install -e ".[train]"` — kept out of the runtime
requirements so serverless bundles stay small.

### Web GUI

| Technology | Role |
|---|---|
| HTML5 / CSS3 / vanilla JavaScript | the dark-themed GUI at `src/polilean/static/index.html` — no frontend framework, no build step |
| SVG | hand-rolled radar chart of the five value axes |

### Development, testing & CI

| Technology | Version | Role |
|---|---|---|
| [pytest](https://docs.pytest.org/) | `>=8` | 60-test suite |
| [pytest-cov](https://pytest-cov.readthedocs.io/) / coverage.py | `>=5` | coverage reporting (~80% total) |
| [ruff](https://docs.astral.sh/ruff/) | `>=0.5` | linting (rules E, F, W, I, N, UP, B; line length 100) |
| [Hatchling](https://hatch.pypa.io/) | — | PEP 517/621 build backend |
| [pip](https://pip.pypa.io/) / venv | — | dependency management |
| [Git](https://git-scm.com/) / [GitHub](https://github.com/) | — | version control and hosting |
| [GitHub Actions](https://github.com/features/actions) | — | CI matrix on Python 3.10–3.13 (ruff + pytest + coverage) |

### Deployment

| Technology | Role |
|---|---|
| [Docker](https://www.docker.com/) | container image baking dependencies, the spaCy model, and the trained model |
| [Docker Compose](https://docs.docker.com/compose/) | one-command local deployment |
| [Vercel](https://vercel.com/) | serverless deployment (Python runtime + FastAPI entrypoint) |

### Key transitive dependencies

Installed automatically underneath the direct dependencies:

- **spaCy stack** — `thinc`, `blis`, `murmurhash`, `cymem`, `preshed`,
  `srsly`, `catalogue`, `confection`, `weasel`, `typer`, `click`,
  `jinja2`, `cloudpathlib`, `smart-open`, `wasabi`, `tqdm`
- **scikit-learn stack** — `scipy`, `joblib`, `threadpoolctl`
- **FastAPI stack** — `pydantic-core`, `starlette`, `httpx`, `h11`,
  `anyio`, `certifi`

## Installation

Requires **Python 3.10–3.13** and **scikit-learn ≥ 1.7.2**. The spaCy
model (`en_core_web_sm` 3.8.0) installs automatically as a pinned
dependency — no separate `spacy download` step is needed.

```bash
git clone https://github.com/ManasTummala/PoliLean.git
cd PoliLean
pip install -e .
```

Optional extras:

```bash
pip install -e ".[dev]"    # tests + lint
pip install -e ".[train]"  # corpus downloaders (HuggingFace Datasets, requests)
```

A pre-trained model on the bundled seed dataset ships with the repo, so
`polilean predict` works immediately after install. Train your own with
`polilean train` (see [Training Data](#training-data)).

## Quick Start

```bash
# Analyze a string (with evidence)
polilean predict "We must tax the wealthy to fund universal healthcare." --explain

# Train on the bundled seed dataset
polilean train

# Launch the dark web GUI
polilean gui
```

## Usage

### CLI

```bash
polilean predict "Some political text" --explain --json   # machine-readable output
polilean predict "Some ambiguous text" --threshold 0.8    # abstain below 80% confidence
polilean train                                            # lean + all 5 axes
polilean gui --port 9000 --no-browser                     # web GUI on a custom port
```

### Web GUI

A dependency-free **dark-themed** web GUI ships with the API — percentile
probability bars, a colored lean badge, diverging value-axis bars, and an
SVG radar chart of the five axes:

```bash
polilean gui                # serves http://127.0.0.1:8000/ and opens a browser
polilean gui --port 9000 --no-browser
```

Or open `http://localhost:8000/` when running the API (`uvicorn
polilean.api:app` or Docker). Type any text (or click an example),
optionally set the abstain threshold (default **55%** — below it the
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

### HTTP API

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

Example response:

```json
{
  "lean": "left",
  "confidence": 0.73,
  "probabilities": {"left": 0.73, "centrist": 0.15, "right": 0.12},
  "evidence": {
    "left":   [{"feature": "universal healthcare", "weight": 0.22}],
    "right":  [{"feature": "healthcare", "weight": -0.16}]
  }
}
```

Python API:

```python
from polilean import PoliticalLeanClassifier

clf = PoliticalLeanClassifier(classifier="logreg").load()
result = clf.predict("We must tax the wealthy to fund universal healthcare.",
                     explain=True, threshold=0.55)
```

### Docker

The image bakes in the dependencies, the spaCy model, **and a model
trained on the bundled CSVs**, so it works the moment it starts.

**Option A — compose (recommended):**

```bash
docker compose up --build      # build the image and start the container
open http://localhost:8000/    # the dark web GUI
```

**Option B — plain docker:**

```bash
docker build -t polilean .            # cook the image (first run: a few minutes)
docker run --rm -p 8000:8000 polilean # start it; API + GUI on port 8000
open http://localhost:8000/
```

`-p 8000:8000` is the bridge: the server listens on port 8000 *inside*
the container, and that flag mirrors it to port 8000 on your machine.

Day-to-day container commands:

```bash
docker ps                          # is it running? (name: polilean-api)
docker logs -f polilean-api        # watch output; Ctrl+C stops watching, not the app
docker exec -it polilean-api bash  # open a shell INSIDE the container
docker stop polilean-api           # stop it   (compose: docker compose down)
```

One-off commands inside the container (the default command starts the
web server; your own command replaces it):

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

## Deploying to Vercel

PoliLean deploys as a serverless FastAPI app on Vercel with no extra
configuration — the repo carries everything Vercel's Python runtime
needs:

- **Entrypoint** — `pyproject.toml` declares
  `[tool.vercel] entrypoint = "src.polilean.api:app"`, which Vercel's
  FastAPI builder uses to locate the ASGI app.
- **Python version** — `.python-version` pins the runtime to
  **3.13** (Vercel supports 3.12–3.14).
- **Dependencies** — installed from `[project] dependencies` in
  `pyproject.toml`, including the `en_core_web_sm` spaCy model wheel, so
  no separate model-download step is needed.
- **Bundle** — `.vercelignore` keeps tests, tools, datasets, and caches
  out of the function bundle; the trained model and web GUI ship in it.

Steps:

1. Push the repo to GitHub and import it at
   [vercel.com/new](https://vercel.com/new) — Vercel auto-detects the
   FastAPI entrypoint from `pyproject.toml`.
2. Deploy. No `vercel.json` is required.
3. Open the deployment URL: `GET /` serves the web GUI, `GET /health`
   reports model status, and `POST /predict` returns the full JSON
   prediction (same contract as [HTTP API](#http-api)).

The environment variables `POLILEAN_MODEL_PATH`, `POLILEAN_CLASSIFIER`,
and `POLILEAN_ABSTAIN_THRESHOLD` can be set in **Project Settings →
Environment Variables**.

> The first request after a cold start loads spaCy and unpickles the
> model, which adds a second or two of latency.

All intra-package imports are relative, so the app imports correctly
both as an installed package (`polilean.api:app` — Docker, CLI) and
straight from the checkout (`src.polilean.api:app` — Vercel).

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

Axis training uses the bundled `data/axes.csv` (365 curated
examples, 73 per axis — see the [Generative AI Statement](#generative-ai-statement)) and happens automatically with the lean model:

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

## Explainability

Every prediction can show which features pushed it toward each lean:

```bash
polilean predict "We must tax the wealthy to fund universal healthcare." --explain --json
```

This is the core reason for choosing TF-IDF + linear models: for a system
that makes claims about political leaning, you can always answer *why*.
Evidence is available for the lean classification and for every value
axis (see [Value Axes](#value-axes)).

## Training Data

### Sources

The bundled seed dataset (`data/train.csv`) contains **549 labeled
examples** — 183 per class — with balanced extremist-language and
internet-slang rows (see below). It trains out of the box:

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
instead of returning `uncertain`.

### Internet-slang coverage

Real online political discourse is lowercase, abbreviated, and meme-flavored
(`tbh`, `fr`, `smh`, `based`, `cope`, `touch grass`, 💀). Formal-only seed
data makes the model miss or abstain on such posts, so the seed set also
includes balanced slang rows for every lean — left populism, right
culture-war slang, and both-sides chronically-online posts as centrist —
as well as slang-flavored rows for every value axis. Everyday (non-political)
slang sentences appear under **all three** labels so pure meme text with no
political signal cancels out and still returns `uncertain` instead of a guess.
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

## Development

```bash
pip install -e ".[dev]"
pytest                      # 60 tests
ruff check src tests        # lint (ruff: E,F,W,I,N,UP,B; line-length 100)
```

CI runs on every push and pull request across **Python 3.10–3.13**
(ruff + pytest with coverage). The committed model artifact is loaded
through a small compatibility shim in `src/polilean/model.py`
(`_patch_sklearn_compat`) so pickles trained on a newer scikit-learn keep
working on the oldest supported version.

## Project Structure

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
pyproject.toml         # dependencies, tool config, Vercel entrypoint
.python-version / .vercelignore  # Vercel runtime pin + bundle excludes
Dockerfile / docker-compose.yml
```

## Generative AI Statement

Generative AI was used throughout the design, implementation, and
documentation of PoliLean **for the specific purpose of removing human
bias**.

Political-lean classification is exactly the kind of task where a single
author's worldview leaks into the data: which examples count as "left"
or "right", which phrasings feel "neutral", how value-axis poles get
named. To counter that, generative AI assistance was used to draft,
balance, and cross-check the labeled training data (lean + all five value
axes, including the extremist-language, neutral-seed, and internet-slang
rows), the probe batteries that audit each training run, the source
code, and this README — so the corpus reflects symmetric coverage of
left, centrist, and right language rather than one person's framing.

Additional bias controls, independent of AI assistance:

- **Balanced classes** — 183 examples per lean; 73 per value axis.
- **Deliberate counter-examples** — everyday slang and neutral reporting
  *about* extremism appear under all labels so no single vocabulary
  becomes a shortcut.
- **Auditable design** — TF-IDF + linear models with per-prediction
  evidence, so every classification can be challenged on its features.
- **Calibrated abstention** — the model returns `uncertain` instead of
  guessing when confidence is low.

Generative AI produced drafts; the maintainer reviewed, tested, and
takes responsibility for the final result.

## Contributing

Contributions are welcome — especially labeled training examples, new
data sources, and calibration improvements.

1. Fork and create a feature branch.
2. `pip install -e ".[dev]"` and make your change with tests.
3. Run `pytest` and `ruff check src tests` — both must pass.
4. Open a pull request describing the change and its effect on model
   behavior (probe results from `tools/train_full.py` are appreciated).

## License

MIT — see [LICENSE](LICENSE).
