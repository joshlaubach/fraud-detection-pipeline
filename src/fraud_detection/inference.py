"""Unified inference: score transactions with supervised, ensemble, or both."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .evaluate import predict_model_scores
from .features import prepare_inference_matrix, prepare_unsupervised_features
from .models.ensemble import DecisionRecord, EnsembleScorer
from .models.unsupervised import compute_gmm_anomaly_scores


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

def load_ensemble_artifact(path: str | Path) -> dict[str, Any]:
    """Load and validate a combined ensemble artifact.

    The artifact is expected to contain:
    - ``champion_model_name``: str
    - ``model``: fitted supervised model
    - ``feature_columns``: list of str
    - ``optimal_threshold``: float
    - ``imputation_medians``: dict
    - ``gmm_model``: fitted GaussianMixture
    - ``gmm_ll_min``: float (training log-likelihood min for normalization)
    - ``gmm_ll_max``: float (training log-likelihood max for normalization)
    - ``unsupervised_scaler``: fitted StandardScaler
    - ``unsupervised_feature_subset``: list of str
    - ``ensemble_threshold``: float
    - ``ensemble_weights``: dict with supervised_weight, anomaly_weight
    """
    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise TypeError("Artifact must be a dictionary.")

    required = {
        "champion_model_name", "model", "feature_columns",
        "gmm_model", "gmm_ll_min", "gmm_ll_max",
        "unsupervised_scaler", "unsupervised_feature_subset",
        "ensemble_threshold", "ensemble_weights",
    }
    missing = sorted(required - set(artifact))
    if missing:
        raise KeyError(f"Ensemble artifact is missing required keys: {missing}")

    return artifact


def build_ensemble_scorer(artifact: dict[str, Any]) -> EnsembleScorer:
    """Construct an EnsembleScorer from a loaded artifact."""
    weights = artifact["ensemble_weights"]
    return EnsembleScorer(
        supervised_model=artifact["model"],
        gmm_model=artifact["gmm_model"],
        unsupervised_scaler=artifact["unsupervised_scaler"],
        feature_columns=artifact["feature_columns"],
        unsupervised_feature_subset=artifact["unsupervised_feature_subset"],
        threshold=artifact["ensemble_threshold"],
        gmm_ll_min=artifact["gmm_ll_min"],
        gmm_ll_max=artifact["gmm_ll_max"],
        supervised_weight=weights.get("supervised_weight", 0.7),
        anomaly_weight=weights.get("anomaly_weight", 0.3),
        model_name=artifact["champion_model_name"],
        logreg_scaler=artifact.get("logreg_scaler"),
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_transactions(
    transactions: pd.DataFrame,
    artifact: dict[str, Any],
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Full inference pipeline: features -> ensemble score -> decisions.

    Parameters
    ----------
    transactions : pd.DataFrame
        Raw transaction rows.
    artifact : dict
        Loaded ensemble artifact.
    cfg : dict or None
        Pipeline config.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with appended columns: fraud_score, knn_anomaly_score,
        ensemble_score, decision, decision_label, decision_threshold, model_name.
    """
    data_cfg = (cfg or {}).get("data", {})
    time_col = data_cfg.get("time_col", "TransactionDT")
    amount_col = data_cfg.get("amount_col", "TransactionAmt")

    feature_cols = artifact["feature_columns"]
    medians = artifact.get("imputation_medians")
    if isinstance(medians, dict):
        medians = pd.Series(medians)

    # Build supervised feature matrix
    X = prepare_inference_matrix(
        transactions, feature_cols, medians,
        time_col=time_col, amount_col=amount_col,
    )

    # Supervised scores
    model_name = artifact["champion_model_name"]
    sup_scores = predict_model_scores(
        model_name, X,
        logreg_model=artifact["model"] if model_name == "LogisticRegression" else None,
        xgb_model=artifact["model"] if model_name == "XGBoost" else None,
        logreg_scaler=artifact.get("logreg_scaler"),
    )

    # GMM anomaly scores
    X_unsup, _, _ = prepare_unsupervised_features(
        X,
        feature_subset=artifact["unsupervised_feature_subset"],
        scaler=artifact["unsupervised_scaler"],
        fit=False,
    )
    anom_scores = compute_gmm_anomaly_scores(
        X_unsup, artifact["gmm_model"],
        ll_min=artifact["gmm_ll_min"],
        ll_max=artifact["gmm_ll_max"],
    )

    # Ensemble
    weights = artifact["ensemble_weights"]
    w_s = weights.get("supervised_weight", 0.7)
    w_a = weights.get("anomaly_weight", 0.3)
    ensemble = w_s * sup_scores + w_a * anom_scores

    threshold = artifact["ensemble_threshold"]
    decisions = (ensemble >= threshold).astype(int)

    # Build output
    output = transactions.copy()
    output["fraud_score"] = sup_scores.astype(float)
    output["gmm_anomaly_score"] = anom_scores.astype(float)
    output["ensemble_score"] = ensemble.astype(float)
    output["decision"] = decisions
    output["decision_label"] = np.where(decisions == 1, "flag_fraud", "allow")
    output["decision_threshold"] = threshold
    output["model_name"] = f"Ensemble({model_name}+GMM)"

    return output


def explain_transactions(
    transactions: pd.DataFrame,
    artifact: dict[str, Any],
    cfg: dict[str, Any] | None = None,
    n_shap_features: int = 5,
) -> list[DecisionRecord]:
    """Score transactions with full decision logging and SHAP explanations.

    Parameters
    ----------
    transactions : pd.DataFrame
        Raw transaction rows.
    artifact : dict
        Loaded ensemble artifact.
    cfg : dict or None
        Pipeline config.
    n_shap_features : int
        Number of top SHAP features per transaction.

    Returns
    -------
    list of DecisionRecord
    """
    data_cfg = (cfg or {}).get("data", {})
    time_col = data_cfg.get("time_col", "TransactionDT")
    amount_col = data_cfg.get("amount_col", "TransactionAmt")

    feature_cols = artifact["feature_columns"]
    medians = artifact.get("imputation_medians")
    if isinstance(medians, dict):
        medians = pd.Series(medians)

    X = prepare_inference_matrix(
        transactions, feature_cols, medians,
        time_col=time_col, amount_col=amount_col,
    )

    scorer = build_ensemble_scorer(artifact)

    # Try SHAP if available
    shap_explainer = None
    try:
        import shap
        if artifact["champion_model_name"] == "XGBoost":
            shap_explainer = shap.TreeExplainer(artifact["model"])
    except ImportError:
        pass

    records = []
    for i in range(len(X)):
        row = X.iloc[[i]]
        record = scorer.explain_decision(row, shap_explainer, n_top=n_shap_features)
        records.append(record)

    return records
