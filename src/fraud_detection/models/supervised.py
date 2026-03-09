"""Supervised model training: Logistic Regression, XGBoost, calibration, and baseline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import xgboost as xgb


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------

def train_logreg(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: dict[str, Any] | None = None,
) -> tuple[LogisticRegression, StandardScaler, np.ndarray]:
    """Train a logistic regression model with standard scaling.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training labels (0/1).
    cfg : dict or None
        Pipeline config; reads ``supervised.logreg`` and ``random_seed``.

    Returns
    -------
    tuple of (LogisticRegression, StandardScaler, np.ndarray)
        Fitted model, fitted scaler, and training probability scores.
    """
    lr_cfg = (cfg or {}).get("supervised", {}).get("logreg", {})
    seed = int((cfg or {}).get("random_seed", 42))

    scaler = StandardScaler(with_mean=False)
    X_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        solver=lr_cfg.get("solver", "lbfgs"),
        max_iter=int(lr_cfg.get("max_iter", 300)),
        class_weight=lr_cfg.get("class_weight", "balanced"),
        random_state=seed,
    )
    model.fit(X_scaled, y_train)
    scores = model.predict_proba(X_scaled)[:, 1]
    return model, scaler, scores


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    cfg: dict[str, Any] | None = None,
) -> tuple[xgb.XGBClassifier, np.ndarray]:
    """Train an XGBoost classifier with early stopping on validation.

    Parameters
    ----------
    X_train, y_train : training data
    X_valid, y_valid : validation data (for early stopping + returned scores)
    cfg : dict or None
        Pipeline config; reads ``supervised.xgboost`` and ``random_seed``.

    Returns
    -------
    tuple of (XGBClassifier, np.ndarray)
        Fitted model and validation probability scores.
    """
    xgb_cfg = (cfg or {}).get("supervised", {}).get("xgboost", {})
    seed = int((cfg or {}).get("random_seed", 42))

    # Compute class imbalance ratio for scale_pos_weight
    n_neg = int((y_train == 0).sum())
    n_pos = max(int((y_train == 1).sum()), 1)
    class_weight_ratio = n_neg / n_pos

    model = xgb.XGBClassifier(
        n_estimators=int(xgb_cfg.get("n_estimators", 600)),
        learning_rate=float(xgb_cfg.get("learning_rate", 0.05)),
        max_depth=int(xgb_cfg.get("max_depth", 6)),
        min_child_weight=int(xgb_cfg.get("min_child_weight", 5)),
        subsample=float(xgb_cfg.get("subsample", 0.8)),
        colsample_bytree=float(xgb_cfg.get("colsample_bytree", 0.8)),
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=seed,
        n_jobs=-1,
        scale_pos_weight=class_weight_ratio,
        early_stopping_rounds=int(xgb_cfg.get("early_stopping_rounds", 50)),
    )
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    valid_scores = model.predict_proba(X_valid)[:, 1]
    return model, valid_scores


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate_model(
    model: Any,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    method: str = "isotonic",
) -> CalibratedClassifierCV:
    """Wrap a pre-fitted model in a calibration layer.

    Parameters
    ----------
    model : fitted classifier
        Must implement ``predict_proba``.
    X_cal : pd.DataFrame
        Calibration feature matrix (held-out from validation).
    y_cal : pd.Series
        Calibration labels.
    method : str
        ``"isotonic"`` or ``"sigmoid"``.

    Returns
    -------
    CalibratedClassifierCV
        Calibrated model wrapper.
    """
    calibrated = CalibratedClassifierCV(model, method=method, cv="prefit")
    calibrated.fit(X_cal, y_cal)
    return calibrated


# ---------------------------------------------------------------------------
# Rule-based baseline (non-ML benchmark)
# ---------------------------------------------------------------------------

def rule_based_baseline(
    X: pd.DataFrame,
    amount_col: str = "TransactionAmt_nonneg",
    hour_col: str = "hour_of_day_proxy",
    amount_pctile: float = 0.95,
    risky_hours: tuple[int, ...] = (0, 1, 2, 3, 4, 5),
    train_amount_threshold: float | None = None,
) -> np.ndarray:
    """Simple rule-based fraud flags: high amount OR unusual hour.

    This baseline establishes a non-ML benchmark to justify model complexity.
    Flag a transaction if its amount exceeds the *amount_pctile* percentile
    OR it occurs during risky hours.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix containing amount and hour columns.
    amount_col : str
        Column with non-negative transaction amounts.
    hour_col : str
        Column with hour-of-day proxy (0-23).
    amount_pctile : float
        Percentile threshold (0-1) for flagging high amounts.
    risky_hours : tuple of int
        Hours considered risky (e.g., late night / early morning).
    train_amount_threshold : float or None
        Pre-computed amount threshold from training data.  If None,
        the threshold is computed from *X* directly (use this only
        when *X* is the training set).

    Returns
    -------
    np.ndarray
        Binary predictions (1 = flag, 0 = allow).
    """
    # Amount rule
    if amount_col in X.columns:
        if train_amount_threshold is None:
            train_amount_threshold = float(X[amount_col].quantile(amount_pctile))
        high_amount = (X[amount_col] >= train_amount_threshold).astype(int)
    else:
        high_amount = pd.Series(0, index=X.index)

    # Time rule
    if hour_col in X.columns:
        risky_time = X[hour_col].isin(risky_hours).astype(int)
    else:
        risky_time = pd.Series(0, index=X.index)

    # Flag if either condition is met
    flags = ((high_amount == 1) | (risky_time == 1)).astype(int)
    return flags.values
