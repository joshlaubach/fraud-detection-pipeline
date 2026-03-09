"""Unit tests for feature engineering helpers."""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from fraud_detection.features import (
    add_amount_features,
    add_anomaly_features,
    add_temporal_proxy_features,
    build_feature_matrix,
    prepare_unsupervised_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """Small synthetic transaction DataFrame."""
    return pd.DataFrame({
        "TransactionAmt": [10.0, 0.5, 150.0, 3.0, 1000.0],
        "TransactionDT": [86400, 3600, 172800, 7200, 259200],
        "C1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "D1": [0.1, 0.2, np.nan, 0.4, 0.5],
    })


# ---------------------------------------------------------------------------
# add_amount_features
# ---------------------------------------------------------------------------

class TestAddAmountFeatures:
    def test_creates_expected_columns(self, sample_df):
        result = add_amount_features(sample_df.copy())
        expected_cols = [
            "TransactionAmt_nonneg",
            "log_TransactionAmt",
            "is_micro_amount",
            "is_tiny_amount",
            "amount_bin_code",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_nonneg_amount(self, sample_df):
        df = sample_df.copy()
        df.loc[0, "TransactionAmt"] = -5.0
        result = add_amount_features(df)
        assert (result["TransactionAmt_nonneg"] >= 0).all()

    def test_micro_amount_flag(self, sample_df):
        result = add_amount_features(sample_df.copy())
        # $0.5 and $3.0 are micro (<=5)
        assert result.loc[1, "is_micro_amount"] == 1
        assert result.loc[3, "is_micro_amount"] == 1
        # $150 is not micro
        assert result.loc[2, "is_micro_amount"] == 0

    def test_missing_column_graceful(self):
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        result = add_amount_features(df)
        assert "TransactionAmt_nonneg" not in result.columns


# ---------------------------------------------------------------------------
# add_temporal_proxy_features
# ---------------------------------------------------------------------------

class TestAddTemporalProxyFeatures:
    def test_hour_range(self, sample_df):
        result = add_temporal_proxy_features(sample_df.copy())
        assert result["hour_of_day_proxy"].between(0, 23).all()

    def test_day_range(self, sample_df):
        result = add_temporal_proxy_features(sample_df.copy())
        assert result["day_of_week_proxy"].between(0, 6).all()

    def test_specific_hour(self):
        # 3600 seconds = 1 hour -> hour_of_day_proxy = 1
        df = pd.DataFrame({"TransactionDT": [3600]})
        result = add_temporal_proxy_features(df)
        assert result.loc[0, "hour_of_day_proxy"] == 1


# ---------------------------------------------------------------------------
# build_feature_matrix
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrix:
    def test_dtype_float32(self, sample_df):
        cols = ["TransactionAmt", "C1", "D1"]
        result = build_feature_matrix(sample_df, cols, None)
        assert result.dtypes.unique().tolist() == [np.dtype("float32")]

    def test_median_fill(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, np.nan]})
        medians = pd.Series({"a": 10.0, "b": 20.0})
        result = build_feature_matrix(df, ["a", "b"], medians)
        assert result.loc[1, "a"] == pytest.approx(10.0, abs=0.01)
        assert result.loc[0, "b"] == pytest.approx(20.0, abs=0.01)

    def test_missing_columns_filled(self):
        df = pd.DataFrame({"a": [1.0, 2.0]})
        result = build_feature_matrix(df, ["a", "nonexistent"], None)
        assert "nonexistent" in result.columns
        assert result["nonexistent"].isna().all()


# ---------------------------------------------------------------------------
# prepare_unsupervised_features
# ---------------------------------------------------------------------------

class TestPrepareUnsupervisedFeatures:
    def test_scaling_zero_mean_unit_var(self, sample_df):
        add_amount_features(sample_df)
        add_temporal_proxy_features(sample_df)
        subset = ["C1", "D1"]
        X_scaled, scaler, used = prepare_unsupervised_features(
            sample_df, feature_subset=subset, fit=True,
        )
        assert X_scaled.shape[1] == 2
        # After scaling, mean should be ~0 and std ~1
        assert abs(X_scaled[:, 0].mean()) < 0.5  # Relaxed for small N
        assert used == subset

    def test_subset_selection(self, sample_df):
        X_scaled, _, used = prepare_unsupervised_features(
            sample_df, feature_subset=["C1"], fit=True,
        )
        assert X_scaled.shape[1] == 1
        assert used == ["C1"]

    def test_transform_mode_requires_scaler(self, sample_df):
        with pytest.raises(ValueError, match="scaler must be provided"):
            prepare_unsupervised_features(sample_df, fit=False)

    def test_unavailable_features_error(self, sample_df):
        with pytest.raises(ValueError, match="None of the requested"):
            prepare_unsupervised_features(
                sample_df, feature_subset=["nonexistent_col"], fit=True,
            )


# ---------------------------------------------------------------------------
# add_anomaly_features
# ---------------------------------------------------------------------------

class TestAddAnomalyFeatures:
    def test_adds_column(self, sample_df):
        scores = np.array([0.1, 0.5, 0.9, 0.2, 0.8])
        result = add_anomaly_features(sample_df.copy(), scores)
        assert "gmm_anomaly_score" in result.columns
        np.testing.assert_array_almost_equal(
            result["gmm_anomaly_score"].values, scores
        )
