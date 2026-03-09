"""Reusable fraud detection feature, evaluation, and modeling utilities."""

# --- Feature engineering ---
from .features import (
    add_amount_features,
    add_anomaly_features,
    add_temporal_proxy_features,
    build_feature_matrix,
    prepare_inference_matrix,
    prepare_unsupervised_features,
)

# --- Evaluation ---
from .evaluate import (
    expected_loss,
    predict_model_scores,
    psi_numeric,
    score_classifier,
    threshold_stats,
)

# --- Configuration ---
from .config import (
    get_cost_params,
    get_random_seed,
    load_config,
    log_config,
)

# --- Cost analysis ---
from .cost_analysis import (
    compute_breakeven_threshold,
    cost_sensitivity_table,
    derive_false_negative_cost,
    derive_false_positive_cost,
    print_cost_analysis_report,
)

# --- Data loading ---
from .data import (
    apply_cleaning_rules,
    load_processed_splits,
    load_raw_data,
    merge_transaction_identity,
    normalize_types,
    temporal_split,
)

__all__ = [
    # Features
    "add_amount_features",
    "add_anomaly_features",
    "add_temporal_proxy_features",
    "build_feature_matrix",
    "prepare_inference_matrix",
    "prepare_unsupervised_features",
    # Evaluation
    "expected_loss",
    "predict_model_scores",
    "psi_numeric",
    "score_classifier",
    "threshold_stats",
    # Config
    "get_cost_params",
    "get_random_seed",
    "load_config",
    "log_config",
    # Cost analysis
    "compute_breakeven_threshold",
    "cost_sensitivity_table",
    "derive_false_negative_cost",
    "derive_false_positive_cost",
    "print_cost_analysis_report",
    # Data
    "apply_cleaning_rules",
    "load_processed_splits",
    "load_raw_data",
    "merge_transaction_identity",
    "normalize_types",
    "temporal_split",
]
