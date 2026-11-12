"""Train a binary classifier on tile embeddings from a few labelled examples.

Typical use: the user tags a few tiles as positive (True) and negative (False) in a
boolean FiftyOne field, e.g. "water_tank". These functions read the labels, train a
scikit-learn model on the embeddings (logistic regression or random forest), estimate
its quality with stratified cross-validation, score every tile, and write the score
and the prediction back to the dataset. Re-run `run_classifier` after labelling more
tiles to retrain on the larger set.

    compute_*   : numpy/scikit-learn only, no FiftyOne access
    run_classifier : full workflow, reads labels from and writes results to FiftyOne
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_KINDS = ("logreg", "rf")


def make_model(kind: str = "logreg", C: float = 1.0, n_estimators: int = 300, random_state: int = 42):
    """Logistic regression (standardised inputs) or random forest, both class-balanced."""
    if kind == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=random_state),
        )
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=n_estimators, class_weight="balanced_subsample",
            random_state=random_state, n_jobs=-1,
        )
    raise ValueError(f"Unknown model kind '{kind}'. Choose from {MODEL_KINDS}.")


def labels_from_field(dataset, field: str) -> dict[str, int]:
    """{sample_id: 1 or 0} for every sample where the boolean field is set. None = unlabelled."""
    ids = dataset.values("id")
    values = dataset.values(field)
    return {sid: int(bool(v)) for sid, v in zip(ids, values) if v is not None}


def build_training_set(emb: np.ndarray, ids, labels: dict[str, int]):
    """Rows of `emb` for the labelled samples. Returns X (n, d), y (n,), sample_ids (n,)."""
    row_of = {str(s): i for i, s in enumerate(ids)}
    kept = [sid for sid in labels if sid in row_of]
    if not kept:
        raise ValueError("None of the labelled samples have an embedding.")
    X = np.asarray(emb)[[row_of[s] for s in kept]]
    y = np.array([labels[s] for s in kept], dtype=int)
    return X, y, kept


def class_counts(y: np.ndarray) -> dict[str, int]:
    return {"positive": int((y == 1).sum()), "negative": int((y == 0).sum())}


def fit_model(X: np.ndarray, y: np.ndarray, kind: str = "logreg", **kwargs):
    if len(np.unique(y)) < 2:
        raise ValueError("Both classes are needed: label at least one positive and one negative sample.")
    model = make_model(kind, **kwargs)
    model.fit(X, y)
    return model


def score_tiles(model, emb: np.ndarray) -> np.ndarray:
    """Probability of the positive class for every row of `emb`."""
    proba = model.predict_proba(np.asarray(emb))
    return proba[:, list(model.classes_).index(1)]


def cross_validate(X: np.ndarray, y: np.ndarray, kind: str = "logreg", n_splits: int = 5,
                   threshold: float = 0.5, seed: int = 42) -> dict:
    """Stratified k-fold cross-validation. Metrics are computed on the out-of-fold predictions.

    The number of folds is reduced when a class has fewer samples than `n_splits`.
    """
    counts = class_counts(y)
    folds = min(n_splits, counts["positive"], counts["negative"])
    if folds < 2:
        raise ValueError(f"Too few labels for cross-validation: {counts}. Label at least 2 per class.")

    oof = np.zeros(len(y), dtype=np.float64)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(X, y):
        model = fit_model(X[train_idx], y[train_idx], kind)
        oof[test_idx] = score_tiles(model, X[test_idx])

    pred = (oof >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary",
                                                               pos_label=1, zero_division=0)
    return {
        "model": kind,
        "folds": folds,
        "n_positive": counts["positive"],
        "n_negative": counts["negative"],
        "roc_auc": float(roc_auc_score(y, oof)),
        "average_precision": float(average_precision_score(y, oof)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "oof_scores": oof,
    }


def compare_models(X: np.ndarray, y: np.ndarray, kinds=MODEL_KINDS, **cv_kwargs) -> pd.DataFrame:
    """Cross-validated metrics of every model kind, one row each."""
    rows = []
    for kind in kinds:
        result = cross_validate(X, y, kind, **cv_kwargs)
        rows.append({k: v for k, v in result.items() if k != "oof_scores"})
    return pd.DataFrame(rows).set_index("model").round(3)


def write_predictions(dataset, sample_ids, scores: np.ndarray, name: str, threshold: float = 0.5) -> None:
    """Write `<name>_score` (float) and `<name>_pred` (bool) for every sample in `sample_ids`."""
    score_of = {str(s): float(v) for s, v in zip(sample_ids, scores)}
    for sample in dataset.iter_samples(autosave=True, progress=False):
        score = score_of.get(sample.id)
        if score is None:
            continue
        sample[f"{name}_score"] = score
        sample[f"{name}_pred"] = bool(score >= threshold)


def run_classifier(dataset, emb: np.ndarray, ids, label_field: str, name: str,
                   kind: str = "logreg", threshold: float = 0.5, n_splits: int = 5,
                   seed: int = 42, save: bool = True, **model_kwargs) -> dict:
    """Read the labels, cross-validate, train on all labels, score every tile and save the results.

    Returns a dict with the fitted model, the score of every tile (in `ids` order),
    the cross-validation metrics and the counts. Call it again after labelling more tiles.
    """
    labels = labels_from_field(dataset, label_field)
    X, y, labelled_ids = build_training_set(emb, ids, labels)
    cv = cross_validate(X, y, kind, n_splits=n_splits, threshold=threshold, seed=seed)
    model = fit_model(X, y, kind, random_state=seed, **model_kwargs)
    scores = score_tiles(model, emb)

    if save:
        write_predictions(dataset, [str(s) for s in ids], scores, name, threshold)

    return {
        "model": model,
        "scores": scores,
        "sample_ids": [str(s) for s in ids],
        "labelled_ids": labelled_ids,
        "cv": {k: v for k, v in cv.items() if k != "oof_scores"},
        "oof_scores": cv["oof_scores"],
        "counts": class_counts(y),
    }
