"""Model training and scoring sub-package."""

from .supervised import (
    calibrate_model,
    rule_based_baseline,
    train_logreg,
    train_xgboost,
)
from .unsupervised import (
    analyze_anomaly_distribution,
    compute_gmm_anomaly_scores,
    dbscan_validate_clusters,
    fit_gmm_anomaly,
    profile_anomaly_groups,
)
from .ensemble import (
    DecisionRecord,
    EnsembleScorer,
    optimize_ensemble_threshold,
)

__all__ = [
    # Supervised
    "calibrate_model",
    "rule_based_baseline",
    "train_logreg",
    "train_xgboost",
    # Unsupervised
    "analyze_anomaly_distribution",
    "compute_gmm_anomaly_scores",
    "dbscan_validate_clusters",
    "fit_gmm_anomaly",
    "profile_anomaly_groups",
    # Ensemble
    "DecisionRecord",
    "EnsembleScorer",
    "optimize_ensemble_threshold",
]
