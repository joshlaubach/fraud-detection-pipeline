"""Evaluation and monitoring helpers for fraud detection models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def expected_loss(
    n_fn: int,
    n_fp: int,
    cost_fn: float = 500.0,
    cost_fp: float = 10.0,
) -> float:
    """Compute total expected loss under asymmetric FN/FP costs."""
    return float(n_fn) * float(cost_fn) + float(n_fp) * float(cost_fp)


def score_classifier(y_true: pd.Series | np.ndarray, scores: pd.Series | np.ndarray) -> dict[str, float]:
    """Return standard ranking/calibration metrics for probabilistic classifiers."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    return {
        "ROC_AUC": roc_auc_score(y, s),
        "PR_AUC": average_precision_score(y, s),
        "Brier": brier_score_loss(y, s),
    }


def predict_model_scores(
    model_name: str,
    x_frame: pd.DataFrame,
    *,
    logreg_model: Any | None = None,
    xgb_model: Any | None = None,
    logreg_scaler: Any | None = None,
) -> np.ndarray:
    """Predict fraud probabilities from the selected model name."""
    if model_name == "LogisticRegression":
        if logreg_model is None or logreg_scaler is None:
            raise ValueError("logreg_model and logreg_scaler are required for LogisticRegression scoring.")
        x_scaled = logreg_scaler.transform(x_frame)
        return logreg_model.predict_proba(x_scaled)[:, 1]

    if model_name == "XGBoost":
        if xgb_model is None:
            raise ValueError("xgb_model is required for XGBoost scoring.")
        return xgb_model.predict_proba(x_frame)[:, 1]

    raise ValueError(f"Unknown model: {model_name}")


def threshold_stats(
    y_true: pd.Series | np.ndarray,
    scores: pd.Series | np.ndarray,
    threshold: float,
    *,
    fn_cost: float = 500.0,
    fp_cost: float = 10.0,
) -> dict[str, float | int]:
    """Compute confusion, precision/recall, flagged-rate, and expected loss at threshold."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    y_pred = (s >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y == 1)).sum())
    fp = int(((y_pred == 1) & (y == 0)).sum())
    fn = int(((y_pred == 0) & (y == 1)).sum())
    tn = int(((y_pred == 0) & (y == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    exp_loss = expected_loss(n_fn=fn, n_fp=fp, cost_fn=fn_cost, cost_fp=fp_cost)

    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "flagged_rate": float(y_pred.mean()),
        "expected_loss": float(exp_loss),
    }


def psi_numeric(
    train_series: pd.Series,
    other_series: pd.Series,
    bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Compute PSI-like numeric shift score using training quantile bins."""
    train_clean = train_series.replace([np.inf, -np.inf], np.nan).dropna()
    other_clean = other_series.replace([np.inf, -np.inf], np.nan).dropna()
    if train_clean.empty or other_clean.empty:
        return np.nan

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(train_clean.quantile(quantiles).values)
    if len(edges) < 3:
        return np.nan

    edges[0] = -np.inf
    edges[-1] = np.inf
    train_bins = pd.cut(train_clean, bins=edges, include_lowest=True)
    other_bins = pd.cut(other_clean, bins=edges, include_lowest=True)

    train_dist = train_bins.value_counts(normalize=True).sort_index()
    other_dist = other_bins.value_counts(normalize=True).reindex(train_dist.index, fill_value=0)

    return float(((other_dist - train_dist) * np.log((other_dist + eps) / (train_dist + eps))).sum())
