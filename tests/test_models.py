"""Unit tests for supervised, unsupervised, and ensemble models."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs, make_classification

from fraud_detection.models.supervised import (
    rule_based_baseline,
    train_logreg,
    train_xgboost,
)
from fraud_detection.models.unsupervised import (
    analyze_anomaly_distribution,
    compute_gmm_anomaly_scores,
    fit_gmm_anomaly,
)
from fraud_detection.models.ensemble import (
    EnsembleScorer,
    optimize_ensemble_threshold,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_classification():
    """Small binary classification dataset."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5,
        random_state=42, weights=[0.9, 0.1],
    )
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    y_s = pd.Series(y, name="target")
    # Split 80/20
    split = 160
    return {
        "X_train": X_df.iloc[:split],
        "y_train": y_s.iloc[:split],
        "X_valid": X_df.iloc[split:],
        "y_valid": y_s.iloc[split:],
    }


@pytest.fixture
def synthetic_normal_data():
    """Synthetic data with clear normal cluster and outliers."""
    np.random.seed(42)
    # Normal cluster
    normal = np.random.randn(100, 4) * 0.5
    # Outliers (far from normal)
    outliers = np.random.randn(10, 4) * 0.5 + 5
    X_all = np.vstack([normal, outliers])
    y = np.array([0] * 100 + [1] * 10)
    return normal, X_all, y


# ---------------------------------------------------------------------------
# Supervised: LogReg
# ---------------------------------------------------------------------------

class TestTrainLogreg:
    def test_returns_model_and_scaler(self, synthetic_classification):
        d = synthetic_classification
        model, scaler, scores = train_logreg(d["X_train"], d["y_train"])
        assert hasattr(model, "predict_proba")
        assert hasattr(scaler, "transform")
        assert len(scores) == len(d["X_train"])

    def test_scores_in_range(self, synthetic_classification):
        d = synthetic_classification
        _, _, scores = train_logreg(d["X_train"], d["y_train"])
        assert (scores >= 0).all() and (scores <= 1).all()


# ---------------------------------------------------------------------------
# Supervised: XGBoost
# ---------------------------------------------------------------------------

class TestTrainXgboost:
    def test_returns_model(self, synthetic_classification):
        d = synthetic_classification
        model, scores = train_xgboost(
            d["X_train"], d["y_train"], d["X_valid"], d["y_valid"],
        )
        assert hasattr(model, "predict_proba")
        assert len(scores) == len(d["X_valid"])


# ---------------------------------------------------------------------------
# Supervised: Rule-based baseline
# ---------------------------------------------------------------------------

class TestRuleBasedBaseline:
    def test_flags_high_amounts(self):
        df = pd.DataFrame({
            "TransactionAmt_nonneg": [1, 2, 3, 100, 200, 300, 400, 500, 600, 1000],
            "hour_of_day_proxy": [12] * 10,  # safe hour
        })
        flags = rule_based_baseline(df, amount_pctile=0.90)
        # Top 10% by amount should be flagged
        assert flags[-1] == 1  # $1000 should be flagged

    def test_flags_risky_hours(self):
        df = pd.DataFrame({
            "TransactionAmt_nonneg": [5.0] * 10,
            "hour_of_day_proxy": [2, 12, 3, 15, 0, 12, 12, 12, 12, 12],
        })
        # Set a high amount threshold so only the hour rule fires
        flags = rule_based_baseline(
            df, risky_hours=(0, 1, 2, 3), train_amount_threshold=999.0
        )
        assert flags[0] == 1  # hour 2 -> risky
        assert flags[1] == 0  # hour 12 -> safe
        assert flags[2] == 1  # hour 3 -> risky
        assert flags[4] == 1  # hour 0 -> risky


# ---------------------------------------------------------------------------
# Unsupervised: GMM anomaly
# ---------------------------------------------------------------------------

class TestFitGmmAnomaly:
    def test_returns_fitted_model(self, synthetic_normal_data):
        normal, _, _ = synthetic_normal_data
        gmm, ll_min, ll_max = fit_gmm_anomaly(normal, n_components_range=[2, 3])
        assert hasattr(gmm, "score_samples")
        assert ll_max >= ll_min

    def test_scores_in_range(self, synthetic_normal_data):
        normal, X_all, _ = synthetic_normal_data
        gmm, ll_min, ll_max = fit_gmm_anomaly(normal, n_components_range=[2, 3])
        scores = compute_gmm_anomaly_scores(X_all, gmm, ll_min, ll_max)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_outliers_score_higher(self, synthetic_normal_data):
        normal, X_all, y = synthetic_normal_data
        gmm, ll_min, ll_max = fit_gmm_anomaly(normal, n_components_range=[2, 3])
        scores = compute_gmm_anomaly_scores(X_all, gmm, ll_min, ll_max)

        normal_scores = scores[y == 0]
        outlier_scores = scores[y == 1]
        # Outliers should have higher mean anomaly score on average
        assert outlier_scores.mean() >= normal_scores.mean()


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

def _build_gmm_scorer(d, lr_scaler, model):
    """Helper to build an EnsembleScorer backed by a GMM."""
    from sklearn.preprocessing import StandardScaler

    subset = ["f0", "f1", "f2", "f3"]
    unsup_scaler = StandardScaler()
    X_train_sub = d["X_train"][subset].fillna(0).values
    unsup_scaler.fit(X_train_sub)

    normal_mask = d["y_train"] == 0
    X_normal = unsup_scaler.transform(
        d["X_train"].loc[normal_mask, subset].fillna(0).values
    )
    gmm, ll_min, ll_max = fit_gmm_anomaly(X_normal, n_components_range=[2, 3])

    return EnsembleScorer(
        supervised_model=model,
        gmm_model=gmm,
        unsupervised_scaler=unsup_scaler,
        feature_columns=d["X_train"].columns.tolist(),
        unsupervised_feature_subset=subset,
        threshold=0.5,
        gmm_ll_min=ll_min,
        gmm_ll_max=ll_max,
        model_name="LogisticRegression",
        logreg_scaler=lr_scaler,
    )


class TestEnsembleScorerOutput:
    def test_output_shape(self, synthetic_classification, synthetic_normal_data):
        d = synthetic_classification
        from fraud_detection.models.supervised import train_logreg
        model, lr_scaler, _ = train_logreg(d["X_train"], d["y_train"])
        scorer = _build_gmm_scorer(d, lr_scaler, model)

        scores = scorer.score(d["X_valid"])
        assert len(scores) == len(d["X_valid"])
        assert (scores >= 0).all()

    def test_decision_records(self, synthetic_classification, synthetic_normal_data):
        d = synthetic_classification
        from fraud_detection.models.supervised import train_logreg
        model, lr_scaler, _ = train_logreg(d["X_train"], d["y_train"])
        scorer = _build_gmm_scorer(d, lr_scaler, model)

        decisions, records = scorer.predict(d["X_valid"])
        assert len(decisions) == len(d["X_valid"])
        assert len(records) == len(d["X_valid"])
        assert all(r.decision in ("flag_fraud", "allow") for r in records)


class TestOptimizeEnsembleThreshold:
    def test_returns_threshold_and_df(self):
        y = np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 1])
        scores = np.array([0.1, 0.2, 0.05, 0.15, 0.8, 0.9, 0.3, 0.1, 0.2, 0.7])
        threshold, df = optimize_ensemble_threshold(y, scores)
        assert 0.0 < threshold < 1.0
        assert "expected_loss" in df.columns
