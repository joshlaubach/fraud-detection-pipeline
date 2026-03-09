"""Unsupervised anomaly detection: GMM (production) + DBSCAN (validation)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import roc_auc_score, silhouette_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture


# ---------------------------------------------------------------------------
# GMM anomaly detection (production)
# ---------------------------------------------------------------------------

def fit_gmm_anomaly(
    X_scaled_normal: np.ndarray,
    n_components_range: list[int] | None = None,
    covariance_type: str = "full",
    max_iter: int = 200,
    n_init: int = 10,
    random_state: int = 42,
) -> tuple[GaussianMixture, float, float]:
    """Fit a Gaussian Mixture Model on NORMAL (non-fraud) training transactions.

    The GMM learns the probability distribution of normal spending patterns.
    At inference time, transactions with low log-likelihood under this model
    are anomalous -- they do not fit the learned normal patterns.

    Model selection: we try several values of n_components and pick the one
    with the lowest BIC (Bayesian Information Criterion).  Lower BIC means
    a better balance of fit quality and model complexity.

    Parameters
    ----------
    X_scaled_normal : np.ndarray
        Scaled feature matrix of normal (non-fraud) transactions only.
    n_components_range : list of int or None
        Numbers of mixture components to evaluate.
        Defaults to [2, 3, 4, 5, 6, 7, 8].
    covariance_type : str
        GMM covariance structure ('full' recommended for fraud data).
    max_iter : int
        Max EM iterations per fit.
    n_init : int
        Number of random initializations (higher = more stable).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple of (GaussianMixture, float, float)
        (fitted_gmm, ll_min, ll_max) where ll_min and ll_max are the
        training log-likelihood range used for score normalization.
    """
    if n_components_range is None:
        n_components_range = [2, 3, 4, 5, 6, 7, 8]

    # BIC-based component selection
    best_bic = np.inf
    best_n = n_components_range[0]
    for n in n_components_range:
        gmm_tmp = GaussianMixture(
            n_components=n,
            covariance_type=covariance_type,
            max_iter=max_iter,
            n_init=max(1, n_init // 2),  # faster for selection
            random_state=random_state,
        )
        gmm_tmp.fit(X_scaled_normal)
        bic = gmm_tmp.bic(X_scaled_normal)
        if bic < best_bic:
            best_bic = bic
            best_n = n

    # Refit with full n_init for stability
    gmm = GaussianMixture(
        n_components=best_n,
        covariance_type=covariance_type,
        max_iter=max_iter,
        n_init=n_init,
        random_state=random_state,
    )
    gmm.fit(X_scaled_normal)

    # Store training range for normalization
    train_ll = gmm.score_samples(X_scaled_normal)
    ll_min = float(train_ll.min())
    ll_max = float(train_ll.max())

    return gmm, ll_min, ll_max


def compute_gmm_anomaly_scores(
    X_scaled: np.ndarray,
    gmm_model: GaussianMixture,
    ll_min: float,
    ll_max: float,
) -> np.ndarray:
    """Compute GMM anomaly scores normalized to [0, 1].

    Anomaly score = 1 - normalized(log-likelihood under the GMM).
    High score (near 1) = does not fit normal patterns = anomalous.
    Low score (near 0) = fits normal patterns well = likely legitimate.

    Normalization uses the training log-likelihood range so scores are
    comparable across train / validation / test splits.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix to score.
    gmm_model : GaussianMixture
        Fitted GMM (trained on normal transactions only).
    ll_min : float
        Minimum log-likelihood from training set (for normalization).
    ll_max : float
        Maximum log-likelihood from training set (for normalization).

    Returns
    -------
    np.ndarray
        Anomaly scores in [0, 1].  Higher = more anomalous.
    """
    ll = gmm_model.score_samples(X_scaled)
    ll_range = ll_max - ll_min
    if ll_range == 0:
        return np.zeros(len(X_scaled), dtype=float)
    scores = 1.0 - (ll - ll_min) / ll_range
    return np.clip(scores, 0.0, 1.0)


# ---------------------------------------------------------------------------
# DBSCAN validation (sanity check only, NOT used in ensemble)
# ---------------------------------------------------------------------------

def dbscan_validate_clusters(
    X_scaled: np.ndarray,
    eps_range: list[float] | None = None,
    min_samples_range: list[int] | None = None,
) -> dict[str, Any]:
    """Grid search DBSCAN parameters as a cluster structure sanity check.

    Selection metric: silhouette score (primary), with noise ratio < 30%
    as a constraint.  Results are for analysis only -- DBSCAN is NOT used
    in the production ensemble.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix.
    eps_range : list of float or None
        Epsilon values to test.
    min_samples_range : list of int or None
        min_samples values to test.

    Returns
    -------
    dict
        {"best_eps", "best_min_samples", "results": list of dicts,
         "best_silhouette": float}
    """
    if eps_range is None:
        eps_range = [0.3, 0.5, 0.7, 1.0, 1.5]
    if min_samples_range is None:
        min_samples_range = [5, 10, 20]

    results = []
    for eps in eps_range:
        for min_samples in min_samples_range:
            db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
            labels = db.fit_predict(X_scaled)

            n_clusters = len(set(labels) - {-1})
            n_noise = int((labels == -1).sum())
            noise_ratio = n_noise / len(labels)

            sil = np.nan
            db_index = np.nan
            if n_clusters >= 2 and noise_ratio < 1.0:
                non_noise_mask = labels != -1
                if non_noise_mask.sum() > n_clusters:
                    sil = float(silhouette_score(
                        X_scaled[non_noise_mask], labels[non_noise_mask]
                    ))
                    db_index = float(davies_bouldin_score(
                        X_scaled[non_noise_mask], labels[non_noise_mask]
                    ))

            results.append({
                "eps": eps,
                "min_samples": min_samples,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "noise_ratio": noise_ratio,
                "silhouette": sil,
                "davies_bouldin": db_index,
            })

    valid_results = [
        r for r in results
        if not np.isnan(r["silhouette"]) and r["noise_ratio"] < 0.30
    ]
    if valid_results:
        best = max(valid_results, key=lambda r: r["silhouette"])
    else:
        scored = [r for r in results if not np.isnan(r["silhouette"])]
        best = max(scored, key=lambda r: r["silhouette"]) if scored else results[0]

    return {
        "best_eps": best["eps"],
        "best_min_samples": best["min_samples"],
        "best_silhouette": best.get("silhouette", np.nan),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Anomaly analysis and profiling (works for any [0,1] anomaly score)
# ---------------------------------------------------------------------------

def analyze_anomaly_distribution(
    scores: np.ndarray,
    y: np.ndarray,
    percentiles: list[int] | None = None,
) -> dict[str, Any]:
    """Analyze fraud rates at different anomaly score percentiles.

    Parameters
    ----------
    scores : np.ndarray
        Anomaly scores in [0, 1].
    y : np.ndarray
        Labels (0/1).
    percentiles : list of int or None
        Percentile thresholds to analyze.

    Returns
    -------
    dict
        {"roc_auc": float, "percentile_analysis": list of dicts}
    """
    if percentiles is None:
        percentiles = [50, 75, 90, 95, 99]

    y_arr = np.asarray(y).astype(int)
    s_arr = np.asarray(scores, dtype=float)

    auc = float(roc_auc_score(y_arr, s_arr))

    analysis = []
    for pct in percentiles:
        threshold = np.percentile(s_arr, pct)
        above_mask = s_arr >= threshold
        n_above = int(above_mask.sum())
        fraud_above = int(y_arr[above_mask].sum()) if n_above > 0 else 0
        fraud_rate = fraud_above / n_above if n_above > 0 else 0.0
        analysis.append({
            "percentile": pct,
            "threshold": float(threshold),
            "n_above": n_above,
            "fraud_count": fraud_above,
            "fraud_rate": float(fraud_rate),
        })

    return {"roc_auc": auc, "percentile_analysis": analysis}


def profile_anomaly_groups(
    X: pd.DataFrame,
    scores: np.ndarray,
    y: pd.Series | np.ndarray,
    n_bins: int = 5,
) -> pd.DataFrame:
    """Profile transactions binned by anomaly score quintile.

    Returns a DataFrame with per-bin size, fraud rate, and feature means.
    """
    df = X.copy()
    df["_anomaly_score"] = scores
    df["_label"] = np.asarray(y).astype(int)

    df["_anomaly_bin"] = pd.qcut(
        df["_anomaly_score"], q=n_bins, labels=False, duplicates="drop"
    )

    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]

    rows = []
    for bin_id in sorted(df["_anomaly_bin"].dropna().unique()):
        mask = df["_anomaly_bin"] == bin_id
        subset = df.loc[mask]
        row = {
            "anomaly_bin": int(bin_id),
            "count": int(mask.sum()),
            "fraud_rate": float(subset["_label"].mean()),
            "mean_anomaly_score": float(subset["_anomaly_score"].mean()),
        }
        for col in numeric_cols[:20]:
            row[f"mean_{col}"] = float(subset[col].mean())
        rows.append(row)

    return pd.DataFrame(rows)
