"""FastAPI service exposing the PoliLean analyzer over HTTP."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from polilean import __version__
from polilean.model import AVAILABLE_CLASSIFIERS, PoliticalLeanClassifier

_model_path_env = os.environ.get("POLILEAN_MODEL_PATH")
MODEL_PATH = Path(_model_path_env) if _model_path_env else None
CLASSIFIER = os.environ.get("POLILEAN_CLASSIFIER", "logreg")

app = FastAPI(
    title="PoliLean API",
    description="Political leaning analysis for text (spaCy + scikit-learn).",
    version=__version__,
)


class PredictRequest(BaseModel):
    text: str
    explain: bool = True


class PredictResponse(BaseModel):
    lean: str
    confidence: float
    probabilities: dict[str, float]
    evidence: dict | None = None


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


@app.get("/health")
def health() -> dict:
    try:
        get_classifier()
        return {"status": "ok", "version": __version__, "classifier": CLASSIFIER}
    except RuntimeError as exc:
        return {"status": "degraded", "detail": str(exc), "version": __version__}


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
    pred = clf.predict(req.text, explain=req.explain)
    d = pred.to_dict()
    return PredictResponse(
        lean=d["lean"],
        confidence=d["confidence"],
        probabilities=d["probabilities"],
        evidence=d.get("evidence"),
    )
