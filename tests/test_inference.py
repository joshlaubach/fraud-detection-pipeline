"""Unit tests for inference and drift monitoring."""

import numpy as np
import pandas as pd
import pytest

from fraud_detection.drift import (
    anomaly_score_drift,
    check_retraining_triggers,
    drift_report,
    feature_drift_precheck,
)


# ---------------------------------------------------------------------------
# Drift monitoring
# ---------------------------------------------------------------------------

class TestFeatureDriftPrecheck:
    def test_output_structure(self):
        np.random.seed(42)
        X_train = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})
        X_other = pd.DataFrame({"a": np.random.randn(50), "b": np.random.randn(50)})
        result = feature_drift_precheck(X_train, X_other)
        assert "metric" in result.columns
        assert "scope" in result.columns
        assert "value" in result.columns
        assert "status" in result.columns

    def test_identical_distributions_ok(self):
        np.random.seed(42)
        data = np.random.randn(200)
        X = pd.DataFrame({"feat": data})
        result = feature_drift_precheck(X, X)
        # Same data should have PSI ~ 0
        assert result.iloc[0]["status"] == "ok"


class TestAnomalyScoreDrift:
    def test_zero_for_identical(self):
        scores = np.random.RandomState(42).rand(100)
        result = anomaly_score_drift(scores, scores)
        assert result["psi"] == pytest.approx(0.0, abs=0.01)
        assert result["ks_stat"] == pytest.approx(0.0, abs=0.01)

    def test_detects_shifted_distribution(self):
        train = np.random.RandomState(42).rand(200)
        other = train + 0.5  # Shifted
        result = anomaly_score_drift(train, other)
        assert result["psi"] > 0.1
        assert result["ks_pval"] < 0.05


class TestDriftReport:
    def test_combined_output(self):
        np.random.seed(42)
        X_train = pd.DataFrame({"a": np.random.randn(100)})
        X_other = pd.DataFrame({"a": np.random.randn(50)})
        train_anom = np.random.rand(100)
        other_anom = np.random.rand(50)
        result = drift_report(X_train, X_other, train_anom, other_anom)
        # Should have feature PSI rows + anomaly rows
        assert len(result) >= 2
        assert "anomaly_psi" in result["metric"].values


class TestCheckRetrainingTriggers:
    def test_no_drift_no_retrain(self):
        df = pd.DataFrame({
            "metric": ["feature_psi", "anomaly_psi"],
            "scope": ["feat_a", "gmm_anomaly_score"],
            "value": [0.01, 0.02],
            "status": ["ok", "ok"],
        })
        result = check_retraining_triggers(df)
        assert result["retrain_recommended"] is False
        assert len(result["reasons"]) == 0

    def test_critical_drift_triggers_retrain(self):
        df = pd.DataFrame({
            "metric": ["feature_psi"],
            "scope": ["feat_a"],
            "value": [0.3],
            "status": ["critical"],
        })
        result = check_retraining_triggers(df)
        assert result["retrain_recommended"] is True
        assert len(result["reasons"]) > 0
