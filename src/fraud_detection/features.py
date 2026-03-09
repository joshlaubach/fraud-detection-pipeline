"""Feature engineering helpers for fraud detection models."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def add_amount_features(df: pd.DataFrame, amount_col: str = "TransactionAmt") -> pd.DataFrame:
    """Add amount-derived features in-place and return ``df`` for chaining."""
    if amount_col not in df.columns:
        return df

    amount = pd.to_numeric(df[amount_col], errors="coerce").clip(lower=0)
    df["TransactionAmt_nonneg"] = amount
    df["log_TransactionAmt"] = np.log1p(amount)
    df["is_micro_amount"] = (amount <= 5).astype("int8")
    df["is_tiny_amount"] = (amount <= 1).astype("int8")

    bins = [-0.01, 1, 5, 25, 100, 500, np.inf]
    df["amount_bin_code"] = pd.cut(amount, bins=bins, labels=False).astype("float32")
    return df


def add_temporal_proxy_features(df: pd.DataFrame, time_col: str = "TransactionDT") -> pd.DataFrame:
    """Add hour/day cyclical proxy features derived from elapsed-seconds time."""
    if time_col not in df.columns:
        return df

    dt = pd.to_numeric(df[time_col], errors="coerce").fillna(0)
    df["hour_of_day_proxy"] = ((dt // 3600) % 24).astype("int8")
    df["day_of_week_proxy"] = ((dt // 86400) % 7).astype("int8")
    return df


def build_feature_matrix(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    median_values: pd.Series | Mapping[str, float] | None,
) -> pd.DataFrame:
    """Return a float32 feature matrix aligned to ``feature_cols`` with median fill."""
    ordered_cols = list(feature_cols)
    out = df.reindex(columns=ordered_cols).copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.apply(pd.to_numeric, errors="coerce")

    if median_values is not None:
        medians = median_values if isinstance(median_values, pd.Series) else pd.Series(median_values)
        medians = pd.to_numeric(medians, errors="coerce")
        out = out.fillna(medians.reindex(out.columns))

    return out.astype("float32")


def prepare_inference_matrix(
    transactions: pd.DataFrame,
    feature_columns: Iterable[str],
    median_values: pd.Series | Mapping[str, float] | None = None,
    *,
    time_col: str = "TransactionDT",
    amount_col: str = "TransactionAmt",
) -> pd.DataFrame:
    """Build a model-ready matrix from transaction rows for inference."""
    frame = transactions.copy()
    add_amount_features(frame, amount_col=amount_col)
    add_temporal_proxy_features(frame, time_col=time_col)
    matrix = build_feature_matrix(frame, feature_columns, median_values)
    return matrix.fillna(0.0).astype("float32")


# ---------------------------------------------------------------------------
# Unsupervised feature preparation
# ---------------------------------------------------------------------------

def prepare_unsupervised_features(
    X: pd.DataFrame,
    feature_subset: list[str] | None = None,
    scaler: StandardScaler | None = None,
    fit: bool = False,
) -> tuple[np.ndarray, StandardScaler, list[str]]:
    """Select and scale features for unsupervised anomaly detection.

    Parameters
    ----------
    X : pd.DataFrame
        Full feature matrix.
    feature_subset : list of str or None
        Columns to select.  If None, all columns are used.
    scaler : StandardScaler or None
        Pre-fitted scaler for transform-only mode.  Ignored when *fit=True*.
    fit : bool
        If True, fit a new scaler on *X*.  If False, use the provided *scaler*.

    Returns
    -------
    tuple of (np.ndarray, StandardScaler, list[str])
        (X_scaled, scaler, used_columns)
    """
    if feature_subset is not None:
        available = [c for c in feature_subset if c in X.columns]
        if not available:
            raise ValueError(
                f"None of the requested features {feature_subset} are in X.columns"
            )
        X_sub = X[available].copy()
    else:
        available = X.columns.tolist()
        X_sub = X.copy()

    # Impute missing values with 0 before scaling
    X_sub = X_sub.fillna(0.0).astype("float32")

    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sub)
    else:
        if scaler is None:
            raise ValueError("scaler must be provided when fit=False")
        X_scaled = scaler.transform(X_sub)

    return X_scaled, scaler, available


def add_anomaly_features(
    df: pd.DataFrame,
    anomaly_scores: np.ndarray,
) -> pd.DataFrame:
    """Append GMM anomaly score column to *df* in-place.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction DataFrame (must have same length as *anomaly_scores*).
    anomaly_scores : np.ndarray
        Normalized anomaly scores in [0, 1].

    Returns
    -------
    pd.DataFrame
        *df* with ``gmm_anomaly_score`` column added.
    """
    df["gmm_anomaly_score"] = anomaly_scores
    return df
