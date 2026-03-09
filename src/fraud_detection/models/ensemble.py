"""Interpretable ensemble: supervised (XGBoost) + unsupervised (GMM anomaly)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from ..evaluate import predict_model_scores, threshold_stats
from ..features import prepare_unsupervised_features
from .unsupervised import compute_gmm_anomaly_scores


@dataclass
class DecisionRecord:
    """Per-transaction decision log for explainability."""

    transaction_id: Any = None
    supervised_score: float = 0.0
    gmm_anomaly_score: float = 0.0
    ensemble_score: float = 0.0
    decision: str = "allow"         # "flag_fraud" | "allow"
    threshold: float = 0.0
    top_shap_features: list[tuple[str, float]] = field(default_factory=list)


class EnsembleScorer:
    """Combine supervised + GMM anomaly scores with decision logging.

    Ensemble formula:
        score = w_supervised * P(fraud|XGBoost) + w_anomaly * gmm_anomaly_score

    Both terms are in [0, 1] so the blend is clean.  A weighted blend is
    smoother than OR-rules and produces a single calibrated score for
    cost-sensitive threshold optimization.
    """

    def __init__(
        self,
        supervised_model: Any,
        gmm_model: GaussianMixture,
        unsupervised_scaler: StandardScaler,
        feature_columns: list[str],
        unsupervised_feature_subset: list[str],
        threshold: float,
        gmm_ll_min: float,
        gmm_ll_max: float,
        supervised_weight: float = 0.7,
        anomaly_weight: float = 0.3,
        model_name: str = "XGBoost",
        logreg_scaler: StandardScaler | None = None,
    ):
        self.supervised_model = supervised_model
        self.gmm_model = gmm_model
        self.unsupervised_scaler = unsupervised_scaler
        self.feature_columns = feature_columns
        self.unsupervised_feature_subset = unsupervised_feature_subset
        self.threshold = threshold
        self.gmm_ll_min = gmm_ll_min
        self.gmm_ll_max = gmm_ll_max
        self.supervised_weight = supervised_weight
        self.anomaly_weight = anomaly_weight
        self.model_name = model_name
        self.logreg_scaler = logreg_scaler

    def _supervised_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Get supervised fraud probabilities."""
        return predict_model_scores(
            self.model_name,
            X,
            logreg_model=self.supervised_model if self.model_name == "LogisticRegression" else None,
            xgb_model=self.supervised_model if self.model_name == "XGBoost" else None,
            logreg_scaler=self.logreg_scaler,
        )

    def _anomaly_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Get GMM anomaly scores."""
        X_scaled, _, _ = prepare_unsupervised_features(
            X,
            feature_subset=self.unsupervised_feature_subset,
            scaler=self.unsupervised_scaler,
            fit=False,
        )
        return compute_gmm_anomaly_scores(
            X_scaled, self.gmm_model,
            ll_min=self.gmm_ll_min,
            ll_max=self.gmm_ll_max,
        )

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Compute ensemble scores for a batch of transactions.

        Returns
        -------
        np.ndarray
            Ensemble scores in [0, 1].
        """
        sup = self._supervised_scores(X)
        anom = self._anomaly_scores(X)
        return self.supervised_weight * sup + self.anomaly_weight * anom

    def predict(
        self, X: pd.DataFrame, id_col: str | None = None,
    ) -> tuple[np.ndarray, list[DecisionRecord]]:
        """Score + threshold -> binary decisions with full logging.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        id_col : str or None
            Column name for transaction IDs (for logging).

        Returns
        -------
        tuple of (np.ndarray, list[DecisionRecord])
            Binary decisions (1=flag, 0=allow) and per-transaction logs.
        """
        sup = self._supervised_scores(X)
        anom = self._anomaly_scores(X)
        ensemble = self.supervised_weight * sup + self.anomaly_weight * anom
        decisions = (ensemble >= self.threshold).astype(int)

        records = []
        ids = X[id_col].values if id_col and id_col in X.columns else range(len(X))
        for i, tx_id in enumerate(ids):
            records.append(DecisionRecord(
                transaction_id=tx_id,
                supervised_score=float(sup[i]),
                gmm_anomaly_score=float(anom[i]),
                ensemble_score=float(ensemble[i]),
                decision="flag_fraud" if decisions[i] == 1 else "allow",
                threshold=self.threshold,
            ))

        return decisions, records

    def explain_decision(
        self,
        X_row: pd.DataFrame,
        shap_explainer: Any = None,
        n_top: int = 5,
    ) -> DecisionRecord:
        """Explain a single transaction decision with optional SHAP.

        Parameters
        ----------
        X_row : pd.DataFrame
            Single-row feature matrix.
        shap_explainer : shap.TreeExplainer or None
            If provided, SHAP values are included in the record.
        n_top : int
            Number of top SHAP features to include.

        Returns
        -------
        DecisionRecord
        """
        sup = self._supervised_scores(X_row)
        anom = self._anomaly_scores(X_row)
        ensemble = self.supervised_weight * sup + self.anomaly_weight * anom
        decision = "flag_fraud" if ensemble[0] >= self.threshold else "allow"

        top_shap = []
        if shap_explainer is not None:
            shap_values = shap_explainer(X_row)
            sv = shap_values.values[0]
            feature_names = X_row.columns.tolist()
            indices = np.argsort(np.abs(sv))[::-1][:n_top]
            top_shap = [(feature_names[j], float(sv[j])) for j in indices]

        return DecisionRecord(
            supervised_score=float(sup[0]),
            gmm_anomaly_score=float(anom[0]),
            ensemble_score=float(ensemble[0]),
            decision=decision,
            threshold=self.threshold,
            top_shap_features=top_shap,
        )


def optimize_ensemble_threshold(
    y_true: np.ndarray,
    ensemble_scores: np.ndarray,
    cost_fn: float = 500.0,
    cost_fp: float = 10.0,
    grid_start: float = 0.01,
    grid_end: float = 0.99,
    grid_steps: int = 199,
) -> tuple[float, pd.DataFrame]:
    """Find the optimal threshold for ensemble scores under asymmetric costs.

    Returns
    -------
    tuple of (float, pd.DataFrame)
        (optimal_threshold, threshold_stats_dataframe)
    """
    thresholds = np.linspace(grid_start, grid_end, grid_steps)
    rows = [
        threshold_stats(y_true, ensemble_scores, t, fn_cost=cost_fn, fp_cost=cost_fp)
        for t in thresholds
    ]
    df = pd.DataFrame(rows)
    best_idx = df["expected_loss"].idxmin()
    return float(df.loc[best_idx, "threshold"]), df
