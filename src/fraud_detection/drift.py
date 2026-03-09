"""Data drift monitoring: feature PSI, anomaly score drift, retraining triggers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .evaluate import psi_numeric


# ---------------------------------------------------------------------------
# Feature-level drift pre-check
# ---------------------------------------------------------------------------

def feature_drift_precheck(
    X_train: pd.DataFrame,
    X_other: pd.DataFrame,
    feature_cols: list[str] | None = None,
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Compute PSI per feature BEFORE anomaly computation.

    If upstream features have drifted, the k-NN anomaly model is stale.
    This is an early warning that catches drift at the source before it
    contaminates anomaly scores.

    Parameters
    ----------
    X_train, X_other : pd.DataFrame
        Training and comparison feature matrices.
    feature_cols : list of str or None
        Features to check.  If None, checks all numeric columns.
    cfg : dict or None
        Pipeline config; reads ``drift.psi_warning`` and ``drift.psi_critical``.

    Returns
    -------
    pd.DataFrame
        Per-feature PSI with status ("ok", "warning", "critical").
    """
    drift_cfg = (cfg or {}).get("drift", {})
    psi_warn = float(drift_cfg.get("psi_warning", 0.1))
    psi_crit = float(drift_cfg.get("psi_critical", 0.2))

    if feature_cols is None:
        feature_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    rows = []
    for col in feature_cols:
        if col in X_train.columns and col in X_other.columns:
            psi_val = psi_numeric(X_train[col], X_other[col])
            if np.isnan(psi_val):
                status = "skip"
            elif psi_val >= psi_crit:
                status = "critical"
            elif psi_val >= psi_warn:
                status = "warning"
            else:
                status = "ok"
            rows.append({
                "metric": "feature_psi",
                "scope": col,
                "value": float(psi_val),
                "status": status,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Anomaly score drift
# ---------------------------------------------------------------------------

def anomaly_score_drift(
    train_scores: np.ndarray,
    other_scores: np.ndarray,
    bins: int = 10,
) -> dict[str, float]:
    """Compute PSI + KS test on anomaly score distributions.

    Parameters
    ----------
    train_scores, other_scores : np.ndarray
        Anomaly scores from training and comparison data.
    bins : int
        Number of bins for PSI computation.

    Returns
    -------
    dict
        {"psi": float, "ks_stat": float, "ks_pval": float}
    """
    train_s = pd.Series(train_scores)
    other_s = pd.Series(other_scores)
    psi_val = psi_numeric(train_s, other_s, bins=bins)

    ks_result = stats.ks_2samp(train_scores, other_scores)

    return {
        "psi": float(psi_val),
        "ks_stat": float(ks_result.statistic),
        "ks_pval": float(ks_result.pvalue),
    }


# ---------------------------------------------------------------------------
# Combined drift report
# ---------------------------------------------------------------------------

def drift_report(
    X_train: pd.DataFrame,
    X_other: pd.DataFrame,
    train_anomaly_scores: np.ndarray,
    other_anomaly_scores: np.ndarray,
    feature_cols: list[str] | None = None,
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Full drift report: feature PSI pre-check + anomaly score drift.

    Runs feature_drift_precheck first, then anomaly score drift.

    Returns
    -------
    pd.DataFrame
        Combined report with columns [metric, scope, value, status].
    """
    drift_cfg = (cfg or {}).get("drift", {})
    psi_warn = float(drift_cfg.get("psi_warning", 0.1))
    psi_crit = float(drift_cfg.get("psi_critical", 0.2))

    # Feature-level pre-check
    feature_df = feature_drift_precheck(X_train, X_other, feature_cols, cfg)

    # Anomaly score drift
    anom_drift = anomaly_score_drift(train_anomaly_scores, other_anomaly_scores)

    anom_psi = anom_drift["psi"]
    if np.isnan(anom_psi):
        anom_status = "skip"
    elif anom_psi >= psi_crit:
        anom_status = "critical"
    elif anom_psi >= psi_warn:
        anom_status = "warning"
    else:
        anom_status = "ok"

    anomaly_rows = pd.DataFrame([
        {
            "metric": "anomaly_psi",
            "scope": "knn_anomaly_score",
            "value": float(anom_psi),
            "status": anom_status,
        },
        {
            "metric": "anomaly_ks",
            "scope": "knn_anomaly_score",
            "value": float(anom_drift["ks_stat"]),
            "status": "info",
        },
    ])

    return pd.concat([feature_df, anomaly_rows], ignore_index=True)


# ---------------------------------------------------------------------------
# Retraining triggers
# ---------------------------------------------------------------------------

def check_retraining_triggers(
    drift_df: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate drift report against thresholds.

    Parameters
    ----------
    drift_df : pd.DataFrame
        Output of ``drift_report()``.
    cfg : dict or None
        Pipeline config.

    Returns
    -------
    dict
        {"retrain_recommended": bool, "reasons": list of str}
    """
    reasons = []

    critical_rows = drift_df[drift_df["status"] == "critical"]
    if len(critical_rows) > 0:
        for _, row in critical_rows.iterrows():
            reasons.append(
                f"CRITICAL drift on {row['scope']}: "
                f"{row['metric']} = {row['value']:.4f}"
            )

    warning_rows = drift_df[drift_df["status"] == "warning"]
    n_warnings = len(warning_rows)
    if n_warnings >= 3:
        reasons.append(
            f"{n_warnings} features with WARNING-level drift (>= 3 triggers retraining)"
        )

    return {
        "retrain_recommended": len(reasons) > 0,
        "reasons": reasons,
    }
