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
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from polilean.axes import AXES
from polilean.data.dataset import load_dataset
from polilean.preprocess import preprocess_many

MODELS_DIR = Path(__file__).resolve().parent / "models" / "trained"

LEANS = ["left", "right", "centrist"]

AVAILABLE_CLASSIFIERS = ("logreg", "linear_svc", "naive_bayes")

# Lean value returned when confidence falls below the abstain threshold.
UNCERTAIN = "uncertain"


def calibration_metrics(
    y_true: list[str] | np.ndarray,
    proba: np.ndarray,
    classes: list[str],
    n_bins: int = 10,
) -> dict:
    """Calibration quality of a probability matrix on labeled data.

    Returns the multiclass Brier score (0 = perfect, max 2.0) and the
    expected calibration error (ECE) over ``n_bins`` equal-width
    confidence bins (0 = perfect, max 1.0). Lower is better for both.
    """
    y_true = np.asarray(y_true).astype(str)
    classes = [str(c) for c in classes]
    index = {c: i for i, c in enumerate(classes)}
    onehot = np.zeros_like(proba, dtype=float)
    for i, y in enumerate(y_true):
        onehot[i, index[y]] = 1.0
    brier = float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))

    conf = proba.max(axis=1)
    pred = np.asarray(classes)[proba.argmax(axis=1)]
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(conf, edges[1:-1], right=False)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.any():
            ece += (mask.sum() / n) * abs(correct[mask].mean() - conf[mask].mean())
    return {"brier": brier, "ece": float(ece), "n_bins": n_bins}


@dataclass
class AxisScore:
    """Rating of one text on a single value axis.

    ``position`` is in [-1, +1]: positive toward the axis's positive
    pole, negative toward its negative pole (see ``AXES``). ``label`` is
    the dominant class and ``confidence`` its probability (0..1).
    """

    axis: str
    label: str
    position: float
    confidence: float
    description: str = ""
    probabilities: dict[str, float] = field(default_factory=dict)
    evidence: dict[str, list[tuple[str, float]]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "label": self.label,
            "position": round(self.position, 4),
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
        }
        if self.description:
            d["description"] = self.description
        spec = AXES.get(self.axis)
        if spec:
            d["negative"] = spec.negative
            d["positive"] = spec.positive
        if self.evidence:
            d["evidence"] = {
                cls: [{"feature": f, "weight": round(w, 4)} for f, w in feats]
                for cls, feats in self.evidence.items()
            }
        return d


@dataclass
class Prediction:
    """Single-text prediction result.

    ``lean`` is the predicted class, or ``"uncertain"`` when confidence
    fell below the abstain threshold (probabilities/evidence are kept so
    the caller can see the near-tie). ``axes`` holds value-axis scores
    when the model was trained with axis data.
    """

    text: str
    lean: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    evidence: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    axes: dict[str, AxisScore] = field(default_factory=dict)

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
        if self.axes:
            d["axes"] = {name: score.to_dict() for name, score in self.axes.items()}
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
        threshold: float | None = None,
    ) -> None:
        if classifier not in AVAILABLE_CLASSIFIERS:
            raise ValueError(
                f"Unknown classifier '{classifier}'; use one of {AVAILABLE_CLASSIFIERS}"
            )
        self.classifier_name = classifier
        self.model_path = Path(model_path) if model_path else MODELS_DIR / f"{classifier}.pkl"
        self.threshold = threshold  # default abstain threshold (None = never abstain)
        self.pipeline: Pipeline | None = None
        self.axis_pipelines: dict[str, Pipeline] = {}

    # ------------------------------------------------------------- training
    def _fit_pipeline(
        self,
        x_tr: list[str],
        y_tr: list[str],
        ngram_range: tuple[int, int] = (1, 2),
        c: float = 4.0,
    ) -> Pipeline:
        """Fit a TF-IDF + classifier pipeline (shared by lean/axis training)."""
        if self.classifier_name == "logreg":
            clf = LogisticRegression(max_iter=1000, C=c)
        elif self.classifier_name == "linear_svc":
            clf = LinearSVC(C=1.0, dual="auto")
        else:
            clf = MultinomialNB()

        # min_df=2 prunes noise on large corpora but destroys tiny datasets
        min_df = 2 if len(x_tr) >= 200 else 1
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, sublinear_tf=True)),
            ("clf", clf),
        ])
        pipeline.fit(x_tr, y_tr)
        return pipeline

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

        self.pipeline = self._fit_pipeline(X_tr, y_tr)  # noqa: N806

        report = classification_report(
            y_te, self.pipeline.predict(X_te), output_dict=True, zero_division=0
        )
        classes, proba = self._predict_proba(X_te)
        cal = calibration_metrics(y_te, proba, classes)
        return {
            "accuracy": report.get("accuracy", 0.0),
            "report": report,
            "train_size": len(X_tr),
            "test_size": len(X_te),
            # Calibration of test-set confidences: Brier (0 = perfect,
            # max 2.0) and expected calibration error (0 = perfect, max 1.0).
            "brier": cal["brier"],
            "ece": cal["ece"],
        }

    def train_axes(
        self,
        df: pd.DataFrame | None = None,
        n_folds: int = 5,
        random_state: int = 42,
    ) -> dict:
        """Train one classifier per value axis (columns: text, axis, label).

        Axis pipelines are fit on **raw** text (no spaCy cleanup): stop
        words and negations carry stance signal. Metrics come from
        stratified ``n_folds`` cross-validation, then each final
        pipeline is fit on all rows of its axis and kept in
        ``axis_pipelines`` (travels with ``save()``/``load()``).
        """
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import StratifiedKFold

        if df is None:
            from polilean.axes import load_axes_dataset

            df = load_axes_dataset()
        missing = {"text", "axis", "label"} - set(df.columns)
        if missing:
            raise ValueError(f"Axes dataset is missing columns: {sorted(missing)}")
        axes_seen = set(df["axis"].astype(str).str.strip().str.lower())
        unknown = axes_seen - set(AXES)
        if unknown:
            raise ValueError(
                f"Unknown axes {sorted(unknown)}; must be one of {sorted(AXES)}"
            )

        results: dict[str, dict] = {}
        for axis, spec in AXES.items():
            sub = df[df["axis"].astype(str).str.strip().str.lower() == axis]
            if sub.empty:
                continue  # custom datasets may cover a subset of axes
            y = sub["label"].astype(str).str.strip().str.lower().tolist()
            bad = set(y) - spec.labels
            if bad:
                raise ValueError(
                    f"Invalid labels {sorted(bad)} for axis '{axis}'; "
                    f"must be one of {sorted(spec.labels)}"
                )
            counts = pd.Series(y).value_counts()
            if counts.min() < n_folds:
                raise ValueError(
                    f"Axis '{axis}' needs at least {n_folds} rows per class, "
                    f"got {int(counts.min())}"
                )
            x = sub["text"].astype(str).tolist()  # raw text on purpose
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            accs: list[float] = []
            all_true: list[str] = []
            all_proba: list[np.ndarray] = []
            classes: list[str] = []
            for tr, te in skf.split(x, y):
                x_tr = [x[i] for i in tr]
                y_tr = [y[i] for i in tr]
                x_te = [x[i] for i in te]
                y_te = [y[i] for i in te]
                # unigrams + stronger regularization: tuned by CV, bigrams
                # overfit the small per-axis datasets
                fold = self._fit_pipeline(x_tr, y_tr, ngram_range=(1, 1))
                accs.append(accuracy_score(y_te, fold.predict(x_te)))
                classes, proba = self._predict_proba(x_te, fold)
                all_true.extend(y_te)
                all_proba.append(proba)
            cal = calibration_metrics(all_true, np.vstack(all_proba), classes)
            self.axis_pipelines[axis] = self._fit_pipeline(x, y, ngram_range=(1, 1))
            results[axis] = {
                "accuracy": float(np.mean(accs)),
                "brier": cal["brier"],
                "ece": cal["ece"],
                "n": len(x),
                "folds": n_folds,
            }
        return results

    # ----------------------------------------------------------- inference
    def _featurize(self, texts: list[str]) -> list[str]:
        return preprocess_many(texts)

    def _predict_proba(
        self, texts: list[str], pipeline: Pipeline | None = None
    ) -> tuple[list[str], np.ndarray]:
        """Class probabilities for already-preprocessed texts.

        Models without ``predict_proba`` (e.g. LinearSVC) fall back to a
        softmax over decision-function margins.
        """
        pipeline = pipeline if pipeline is not None else self.pipeline
        if pipeline is None:
            raise RuntimeError("No trained pipeline - train or load first.")
        clf_step = pipeline.named_steps["clf"]
        classes = [str(c) for c in clf_step.classes_]
        if hasattr(clf_step, "predict_proba"):
            proba = np.asarray(pipeline.predict_proba(texts))
        else:
            decision = np.asarray(pipeline.decision_function(texts), dtype=float)
            if decision.ndim == 1:  # binary edge case: one margin column
                decision = np.vstack([-decision, decision]).T
            exp = np.exp(decision - decision.max(axis=1, keepdims=True))
            proba = exp / exp.sum(axis=1, keepdims=True)
        return classes, proba

    def _evidence_for(
        self,
        clean_text: str,
        pipeline: Pipeline | None = None,
        top_k: int = 8,
        drop_stopwords: bool = False,
    ) -> dict[str, list[tuple[str, float]]]:
        """Top weighted n-grams from this text, per class, from linear coefficients."""
        pipeline = pipeline if pipeline is not None else self.pipeline
        if pipeline is None:
            return {}
        tfidf = pipeline.named_steps["tfidf"]
        clf = pipeline.named_steps["clf"]
        if not hasattr(clf, "coef_"):
            return {}
        vec = tfidf.transform([clean_text])
        feature_names = np.asarray(tfidf.get_feature_names_out())
        active = vec.nonzero()[1]
        if active.size == 0:
            return {}
        if drop_stopwords:
            # axis pipelines keep stop words (they carry stance signal),
            # but they make noisy explanations - filter them from evidence
            active = np.asarray(
                [j for j in active if str(feature_names[j]) not in ENGLISH_STOP_WORDS],
                dtype=int,
            )
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

    def _axis_scores(self, text: str, explain: bool) -> dict[str, AxisScore]:
        """Rate one text on every trained value axis.

        ``text`` is the *raw* input: axis pipelines are fit on raw text.
        """
        scores: dict[str, AxisScore] = {}
        for axis, spec in AXES.items():
            pipeline = self.axis_pipelines.get(axis)
            if pipeline is None:
                continue
            classes, proba = self._predict_proba([text], pipeline)
            probabilities = {c: float(p) for c, p in zip(classes, proba[0], strict=True)}
            label = max(probabilities, key=probabilities.get)  # type: ignore[arg-type]
            position = (
                probabilities.get(spec.positive, 0.0) - probabilities.get(spec.negative, 0.0)
            )
            scores[axis] = AxisScore(
                axis=axis,
                label=label,
                position=position,
                confidence=probabilities[label],
                description=spec.description,
                probabilities=probabilities,
                evidence=self._evidence_for(text, pipeline, drop_stopwords=True)
                if explain
                else {},
            )
        return scores

    def predict(
        self, text: str, explain: bool = False, threshold: float | None = None
    ) -> Prediction:
        """Predict the lean of one text.

        ``threshold`` (falling back to the instance default) is the minimum
        confidence required to emit a label: predictions below it abstain
        with ``lean == "uncertain"`` while keeping the raw probabilities
        and evidence. Pass ``0.0`` to always emit a label.

        Value-axis scores are included when the model was trained with
        axis data (see :meth:`train_axes`).
        """
        if self.pipeline is None:
            self.load()
        clean = self._featurize([text])[0]
        classes, proba = self._predict_proba([clean])
        probabilities = {c: float(p) for c, p in zip(classes, proba[0], strict=True)}
        lean = max(probabilities, key=probabilities.get)  # type: ignore[arg-type]
        confidence = probabilities[lean]
        eff_threshold = self.threshold if threshold is None else threshold
        if eff_threshold is not None and confidence < eff_threshold:
            lean = UNCERTAIN
        evidence = self._evidence_for(clean) if explain else {}
        return Prediction(
            text=text,
            lean=lean,
            confidence=confidence,
            probabilities=probabilities,
            evidence=evidence,
            axes=self._axis_scores(text, explain),
        )

    def predict_many(
        self, texts: list[str], threshold: float | None = None
    ) -> list[Prediction]:
        """Predict several texts; same abstain semantics as :meth:`predict`."""
        if self.pipeline is None:
            self.load()
        X = self._featurize(texts)  # noqa: N806
        classes, proba = self._predict_proba(X)
        eff_threshold = self.threshold if threshold is None else threshold
        results = []
        for text, probs in zip(texts, proba, strict=True):
            lean = max(zip(classes, probs, strict=True), key=lambda t: t[1])[0]
            confidence = float(probs.max())
            if eff_threshold is not None and confidence < eff_threshold:
                lean = UNCERTAIN
            results.append(Prediction(
                text=text,
                lean=lean,
                confidence=confidence,
                probabilities={c: float(p) for c, p in zip(classes, probs, strict=True)},
                axes=self._axis_scores(text, explain=False),
            ))
        return results

    # ------------------------------------------------------------ persistence
    def save(self, path: Path | None = None) -> Path:
        if self.pipeline is None:
            raise RuntimeError("Nothing to save - train or load first.")
        path = Path(path) if path else self.model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "classifier": self.classifier_name,
            "pipeline": self.pipeline,
            "axes": self.axis_pipelines,
        }
        with path.open("wb") as fh:
            pickle.dump(payload, fh)
        return path

    def load(self, path: Path | None = None) -> PoliticalLeanClassifier:
        path = Path(path) if path else self.model_path
        if not path.exists():
            raise FileNotFoundError(
                f"No trained model at '{path}'. Run: polilean train"
            )
        with path.open("rb") as fh:
            obj = pickle.load(fh)
        if isinstance(obj, dict) and "pipeline" in obj:
            self.pipeline = obj["pipeline"]
            self.axis_pipelines = obj.get("axes") or {}
        else:
            # legacy format: a bare sklearn Pipeline (no value axes)
            self.pipeline = obj
            self.axis_pipelines = {}
        return self
