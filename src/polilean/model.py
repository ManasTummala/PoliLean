"""Political leaning classifier: TF-IDF + linear models via scikit-learn.

Classical-ML baseline chosen deliberately: fast to train on CPU, no GPU
required, and fully interpretable - you can inspect exactly which n-grams
pushed a prediction toward left/right/centrist, which matters when a system
makes claims about political leaning.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from polilean.data.dataset import load_dataset
from polilean.preprocess import preprocess_many

MODELS_DIR = Path(__file__).resolve().parent / "models" / "trained"

LEANS = ["left", "right", "centrist"]

AVAILABLE_CLASSIFIERS = ("logreg", "linear_svc", "naive_bayes")


@dataclass
class Prediction:
    """Single-text prediction result."""

    text: str
    lean: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    evidence: dict[str, list[tuple[str, float]]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "text": self.text,
            "lean": self.lean,
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
        }
        if self.evidence:
            d["evidence"] = {
                lean: [{"feature": f, "weight": round(w, 4)} for f, w in feats]
                for lean, feats in self.evidence.items()
            }
        return d


class PoliticalLeanClassifier:
    """TF-IDF (over spaCy-lemmatized text) -> linear classifier.

    Supports ``logreg`` (probabilities + coefficients), ``linear_svc``
    (LinearSVC - coefficients only, confidence from decision-function
    softmax) and ``naive_bayes``.
    """

    def __init__(
        self,
        classifier: str = "logreg",
        model_path: Path | None = None,
    ) -> None:
        if classifier not in AVAILABLE_CLASSIFIERS:
            raise ValueError(
                f"Unknown classifier '{classifier}'; use one of {AVAILABLE_CLASSIFIERS}"
            )
        self.classifier_name = classifier
        self.model_path = Path(model_path) if model_path else MODELS_DIR / f"{classifier}.pkl"
        self.pipeline: Pipeline | None = None

    # ------------------------------------------------------------- training
    def train(
        self,
        df: pd.DataFrame | None = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> dict:
        """Train on a labeled DataFrame (columns: text, lean)."""
        from sklearn.metrics import classification_report
        from sklearn.model_selection import train_test_split

        if df is None:
            df = load_dataset()

        X = df["text"].astype(str).tolist()  # noqa: N806
        y = df["lean"].astype(str).tolist()

        X_clean = preprocess_many(X)  # noqa: N806

        X_tr, X_te, y_tr, y_te = train_test_split(  # noqa: N806
            X_clean, y, test_size=test_size, random_state=random_state, stratify=y
        )

        if self.classifier_name == "logreg":
            clf = LogisticRegression(max_iter=1000, C=4.0)
        elif self.classifier_name == "linear_svc":
            clf = LinearSVC(C=1.0, dual="auto")
        else:
            clf = MultinomialNB()

        # min_df=2 prunes noise on large corpora but destroys tiny datasets
        min_df = 2 if len(X_tr) >= 200 else 1
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, sublinear_tf=True)),
            ("clf", clf),
        ])
        self.pipeline.fit(X_tr, y_tr)

        report = classification_report(
            y_te, self.pipeline.predict(X_te), output_dict=True, zero_division=0
        )
        return {
            "accuracy": report.get("accuracy", 0.0),
            "report": report,
            "train_size": len(X_tr),
            "test_size": len(X_te),
        }

    # ----------------------------------------------------------- inference
    def _featurize(self, texts: list[str]) -> list[str]:
        return preprocess_many(texts)

    def _evidence_for(
        self, clean_text: str, top_k: int = 8
    ) -> dict[str, list[tuple[str, float]]]:
        """Top weighted n-grams from this text, per class, from linear coefficients."""
        if self.pipeline is None:
            return {}
        tfidf = self.pipeline.named_steps["tfidf"]
        clf = self.pipeline.named_steps["clf"]
        if not hasattr(clf, "coef_"):
            return {}
        vec = tfidf.transform([clean_text])
        feature_names = np.asarray(tfidf.get_feature_names_out())
        active = vec.nonzero()[1]
        if active.size == 0:
            return {}
        coef = clf.coef_
        classes = [str(c) for c in clf.classes_]
        evidence: dict[str, list[tuple[str, float]]] = {}
        for idx, cls in enumerate(classes):
            row = coef[idx] if coef.shape[0] > 1 else coef[0]
            scored = sorted(
                ((str(feature_names[j]), float(row[j]) * float(vec[0, j])) for j in active),
                key=lambda t: -abs(t[1]),
            )[:top_k]
            evidence[cls] = scored
        return evidence

    def predict(self, text: str, explain: bool = False) -> Prediction:
        if self.pipeline is None:
            self.load()
        clean = self._featurize([text])[0]
        clf_step = self.pipeline.named_steps["clf"]

        if hasattr(clf_step, "predict_proba"):
            probs = self.pipeline.predict_proba([clean])[0]
            classes = [str(c) for c in clf_step.classes_]
            probabilities = {c: float(p) for c, p in zip(classes, probs, strict=True)}
        else:
            # e.g. LinearSVC: softmax over decision-function margins
            decision = self.pipeline.decision_function([clean])[0]
            classes = [str(c) for c in clf_step.classes_]
            exp = np.exp(decision - np.max(decision))
            softmax = exp / exp.sum()
            probabilities = {c: float(s) for c, s in zip(classes, softmax, strict=True)}

        lean = max(probabilities, key=probabilities.get)  # type: ignore[arg-type]
        evidence = self._evidence_for(clean) if explain else {}
        return Prediction(
            text=text,
            lean=lean,
            confidence=probabilities[lean],
            probabilities=probabilities,
            evidence=evidence,
        )

    def predict_many(self, texts: list[str]) -> list[Prediction]:
        if self.pipeline is None:
            self.load()
        X = self._featurize(texts)  # noqa: N806
        clf_step = self.pipeline.named_steps["clf"]
        if hasattr(clf_step, "predict_proba"):
            proba = self.pipeline.predict_proba(X)
        else:
            decision = self.pipeline.decision_function(X)
            exp = np.exp(decision - decision.max(axis=1, keepdims=True))
            proba = exp / exp.sum(axis=1, keepdims=True)
        classes = [str(c) for c in clf_step.classes_]
        results = []
        for text, probs in zip(texts, proba, strict=True):
            lean = max(zip(classes, probs, strict=True), key=lambda t: t[1])[0]
            results.append(Prediction(
                text=text,
                lean=lean,
                confidence=float(probs.max()),
                probabilities={c: float(p) for c, p in zip(classes, probs, strict=True)},
            ))
        return results

    # ------------------------------------------------------------ persistence
    def save(self, path: Path | None = None) -> Path:
        if self.pipeline is None:
            raise RuntimeError("Nothing to save - train or load first.")
        path = Path(path) if path else self.model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self.pipeline, fh)
        return path

    def load(self, path: Path | None = None) -> PoliticalLeanClassifier:
        path = Path(path) if path else self.model_path
        if not path.exists():
            raise FileNotFoundError(
                f"No trained model at '{path}'. Run: polilean train"
            )
        with path.open("rb") as fh:
            self.pipeline = pickle.load(fh)
        return self
