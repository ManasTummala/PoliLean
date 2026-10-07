"""FastAPI service exposing the PoliLean analyzer over HTTP."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from . import __version__
from .model import AVAILABLE_CLASSIFIERS, PoliticalLeanClassifier

# --- Environment knobs -------------------------------------------------------
# ELI5: read optional settings from environment variables before the app
# starts. POLILEAN_MODEL_PATH points at a specific model file;
# POLILEAN_CLASSIFIER picks the brain (logreg by default);
# POLILEAN_ABSTAIN_THRESHOLD is the global "give up below this confidence"
# number. Bad threshold values fail fast at startup instead of silently
# never abstaining.
_model_path_env = os.environ.get("POLILEAN_MODEL_PATH")
MODEL_PATH = Path(_model_path_env) if _model_path_env else None
CLASSIFIER = os.environ.get("POLILEAN_CLASSIFIER", "logreg")


def _env_threshold() -> float | None:
    """Optional default abstain threshold from POLILEAN_ABSTAIN_THRESHOLD."""
    raw = os.environ.get("POLILEAN_ABSTAIN_THRESHOLD")
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"POLILEAN_ABSTAIN_THRESHOLD={raw!r} is not a number") from exc
    if not 0.0 <= value <= 1.0:
        raise RuntimeError(f"POLILEAN_ABSTAIN_THRESHOLD={raw!r} must be between 0 and 1")
    return value


DEFAULT_THRESHOLD = _env_threshold()

# Absolute path of the web GUI page served at GET /.
GUI_PAGE = Path(__file__).resolve().parent / "static" / "index.html"

# --- The FastAPI app ---------------------------------------------------------
# ELI5: this is the web server definition. Three routes:
# GET / (the GUI), GET /health (is the model loaded?), POST /predict (analyze).
app = FastAPI(
    title="PoliLean API",
    description="Political leaning analysis for text (spaCy + scikit-learn).",
    version=__version__,
)


# --- Request / response shapes -----------------------------------------------
# ELI5: Pydantic models = contract with the caller. They validate incoming
# JSON (threshold must be 0..1 or the client gets a 422) and document the
# response fields (lean, confidence, probabilities, evidence, axes).
class PredictRequest(BaseModel):
    text: str
    explain: bool = True
    threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Abstain with lean='uncertain' when max probability is below "
                    "this value. Falls back to POLILEAN_ABSTAIN_THRESHOLD when "
                    "omitted.",
    )


class PredictResponse(BaseModel):
    lean: str
    confidence: float
    probabilities: dict[str, float]
    evidence: dict | None = None
    axes: dict[str, dict] | None = None


# --- Loading the model once --------------------------------------------------
# ELI5: lru_cache = "remember the answer". The first request pays the cost of
# loading the trained model from disk; every later request reuses it. Raises a
# friendly error if no model file exists yet (run: polilean train).
@lru_cache(maxsize=1)
def get_classifier() -> PoliticalLeanClassifier:
    if CLASSIFIER not in AVAILABLE_CLASSIFIERS:
        raise RuntimeError(f"Unknown POLILEAN_CLASSIFIER '{CLASSIFIER}'")
    clf = PoliticalLeanClassifier(classifier=CLASSIFIER, model_path=MODEL_PATH)
    try:
        clf.load()
    except FileNotFoundError as exc:
        raise RuntimeError(
            "No trained model found. Train one first: polilean train"
        ) from exc
    return clf


# --- Routes ------------------------------------------------------------------
# ELI5: GET / serves the dark-themed GUI page (index.html) as-is.
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def gui() -> FileResponse:
    """Serve the PoliLean web GUI (percentile bars + axis radar chart)."""
    if not GUI_PAGE.exists():
        raise HTTPException(status_code=404, detail="GUI page not found")
    return FileResponse(GUI_PAGE, media_type="text/html; charset=utf-8")


# ELI5: /health reports "ok" when the model loaded, "degraded" with the
# reason when it did not - the GUI footer displays this text.
@app.get("/health")
def health() -> dict:
    try:
        get_classifier()
        return {"status": "ok", "version": __version__, "classifier": CLASSIFIER}
    except RuntimeError as exc:
        return {"status": "degraded", "detail": str(exc), "version": __version__}


# ELI5: /predict is the analyzer. Reject blank/too-long text, load the
# model, apply the abstain threshold (request value wins over env default),
# and return the full Prediction as JSON - including the five axis scores.
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=422, detail="'text' must be a non-empty string")
    if len(req.text) > 200_000:
        raise HTTPException(status_code=413, detail="text too long (max 200,000 chars)")
    try:
        clf = get_classifier()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    threshold = req.threshold if req.threshold is not None else DEFAULT_THRESHOLD
    pred = clf.predict(req.text, explain=req.explain, threshold=threshold)
    d = pred.to_dict()
    return PredictResponse(
        lean=d["lean"],
        confidence=d["confidence"],
        probabilities=d["probabilities"],
        evidence=d.get("evidence"),
        axes=d.get("axes"),
    )
