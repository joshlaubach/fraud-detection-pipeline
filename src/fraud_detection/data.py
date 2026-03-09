"""Data loading, merging, preprocessing, and temporal splitting."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def infer_optimized_dtypes(csv_path: Path, sample_rows: int = 50_000) -> dict[str, str]:
    """Infer a lower-memory dtype map from a CSV sample."""
    sample = pd.read_csv(csv_path, nrows=sample_rows, low_memory=True)
    dtype_map: dict[str, str] = {}
    for col, dt in sample.dtypes.items():
        if pd.api.types.is_float_dtype(dt):
            dtype_map[col] = "float32"
        elif pd.api.types.is_integer_dtype(dt):
            dtype_map[col] = "Int32"
    return dtype_map


def read_csv_optimized(csv_path: Path) -> pd.DataFrame:
    """Read a CSV with memory-optimized dtypes."""
    dtype_map = infer_optimized_dtypes(csv_path)
    return pd.read_csv(csv_path, dtype=dtype_map, low_memory=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw_data(
    raw_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train_transaction and train_identity CSVs from *raw_dir*.

    Tries optimized CSV paths first (``train_transaction_raw.csv``), then
    falls back to the standard names (``train_transaction.csv``).

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        (train_transaction, train_identity)
    """
    raw_dir = Path(raw_dir)

    def _load(preferred: str, fallback: str) -> pd.DataFrame:
        for name in (preferred, fallback):
            p = raw_dir / name
            if p.exists():
                return read_csv_optimized(p)
        raise FileNotFoundError(
            f"Neither {preferred!r} nor {fallback!r} found in {raw_dir}"
        )

    train_tx = _load("train_transaction_raw.csv", "train_transaction.csv")
    train_id = _load("train_identity_raw.csv", "train_identity.csv")
    return train_tx, train_id


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

def merge_transaction_identity(
    transactions: pd.DataFrame,
    identity: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join transactions with identity on TransactionID.

    Deduplicates identity rows before merging and validates a one-to-one
    relationship.
    """
    if "TransactionID" not in transactions.columns:
        raise KeyError("TransactionID not found in transactions DataFrame")
    if "TransactionID" not in identity.columns:
        raise KeyError("TransactionID not found in identity DataFrame")

    identity_work = identity.copy()
    if not identity_work["TransactionID"].is_unique:
        identity_work = identity_work.drop_duplicates(
            subset=["TransactionID"], keep="first"
        )

    merged = transactions.merge(
        identity_work, on="TransactionID", how="left", validate="one_to_one"
    )
    return merged


# ---------------------------------------------------------------------------
# Type normalization
# ---------------------------------------------------------------------------

def normalize_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce nearly-numeric object columns and standardize string values.

    * Object columns that parse as numeric >= 98 % of the time are coerced.
    * Remaining string columns are stripped and blank/null sentinels replaced.
    * Boolean-like string columns (T/F, Y/N, 1/0) are mapped to float32 flags.

    Operates **in-place** on *df* and returns it for chaining.
    """
    # Coerce object columns that are almost fully numeric
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in object_cols:
        parsed = pd.to_numeric(df[col], errors="coerce")
        coverage = parsed.notna().mean()
        if coverage > 0.98 and df[col].notna().any():
            df[col] = parsed

    # Standardize remaining object/string values
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in object_cols:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    # Map boolean-like strings to numeric flags
    bool_like_map = {
        "T": 1, "F": 0, "TRUE": 1, "FALSE": 0,
        "Y": 1, "N": 0, "1": 1, "0": 0,
    }
    for col in df.select_dtypes(include=["string", "object"]).columns:
        vals = df[col].dropna().astype(str).str.upper().unique()
        if len(vals) > 0 and set(vals).issubset(set(bool_like_map.keys())):
            df[col] = (
                df[col].astype(str).str.upper().map(bool_like_map).astype("float32")
            )

    return df


# ---------------------------------------------------------------------------
# Cleaning rules
# ---------------------------------------------------------------------------

def apply_cleaning_rules(
    df: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Apply data cleaning rules: clip amounts/timestamps, drop bad rows.

    Operates **in-place** on *df* and returns it for chaining.

    Parameters
    ----------
    df : pd.DataFrame
        Raw merged transaction table.
    cfg : dict or None
        Pipeline configuration (used for column name lookups).
    """
    data_cfg = (cfg or {}).get("data", {})
    amount_col = data_cfg.get("amount_col", "TransactionAmt")
    time_col = data_cfg.get("time_col", "TransactionDT")
    target_col = data_cfg.get("target_col", "isFraud")
    id_col = data_cfg.get("id_col", "TransactionID")

    # Non-negative transaction amount
    if amount_col in df.columns:
        df[amount_col] = df[amount_col].clip(lower=0)

    # Drop duplicate transaction keys
    if id_col in df.columns:
        df = df.drop_duplicates(subset=[id_col], keep="first")

    # Replace inf with NaN across numerics
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    # Non-negative timestamps
    if time_col in df.columns:
        df[time_col] = df[time_col].clip(lower=0)

    # Validate binary target
    if target_col in df.columns:
        valid_mask = df[target_col].isin([0, 1])
        if not valid_mask.all():
            df = df.loc[valid_mask].copy()
        df[target_col] = df[target_col].astype(int)

    return df


# ---------------------------------------------------------------------------
# Temporal splitting
# ---------------------------------------------------------------------------

def temporal_split(
    df: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create chronological train / gap / validation / test splits.

    A 30-day gap window between train and validation respects label delay
    (fraud confirmation lag).  The gap rows are returned for analysis but
    should **never** be used for training or evaluation.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction table with target and time columns.
    cfg : dict or None
        Pipeline configuration.

    Returns
    -------
    tuple of (split_train, split_gap, split_valid, split_test)
    """
    data_cfg = (cfg or {}).get("data", {})
    splits_cfg = (cfg or {}).get("splits", {})
    time_col = data_cfg.get("time_col", "TransactionDT")
    target_col = data_cfg.get("target_col", "isFraud")
    id_col = data_cfg.get("id_col", "TransactionID")
    train_q = float(splits_cfg.get("train_quantile", 0.75))
    valid_q = float(splits_cfg.get("valid_quantile", 0.90))
    gap_days = int(splits_cfg.get("gap_days", 30))

    required = [target_col, id_col, time_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Required columns missing for temporal split: {missing}")

    tx = df.sort_values(time_col).reset_index(drop=True)

    train_cutoff = tx[time_col].quantile(train_q)
    valid_cutoff = tx[time_col].quantile(valid_q)

    gap_seconds = gap_days * 24 * 3600
    gap_start = train_cutoff
    gap_end = train_cutoff + gap_seconds

    # Adjust valid_cutoff if gap window overlaps
    if gap_end >= valid_cutoff:
        max_dt = tx[time_col].max()
        valid_cutoff = max(valid_cutoff, gap_end + max(1.0, (max_dt - gap_end) * 0.5))

    split_train = tx.loc[tx[time_col] <= train_cutoff].copy()
    split_gap = tx.loc[
        (tx[time_col] > gap_start) & (tx[time_col] <= gap_end)
    ].copy()
    split_valid = tx.loc[
        (tx[time_col] > gap_end) & (tx[time_col] <= valid_cutoff)
    ].copy()
    split_test = tx.loc[tx[time_col] > valid_cutoff].copy()

    if min(len(split_train), len(split_valid), len(split_test)) == 0:
        raise ValueError(
            "At least one split is empty. "
            "Check time column distribution and cutoff logic."
        )

    return split_train, split_gap, split_valid, split_test


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def load_processed_splits(
    processed_dir: str | Path,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load pickled X/y splits from a checkpoint directory.

    Returns
    -------
    tuple of (X_train, y_train, X_valid, y_valid, X_test, y_test)
    """
    d = Path(processed_dir)
    results = []
    for name in ("X_train", "y_train", "X_valid", "y_valid", "X_test", "y_test"):
        pkl_path = d / f"{name}.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {pkl_path}")
        with open(pkl_path, "rb") as fh:
            obj = pickle.load(fh)
        # Ensure y arrays are 1-D Series, not single-column DataFrames
        if name.startswith("y") and isinstance(obj, pd.DataFrame):
            obj = obj.iloc[:, 0]
        results.append(obj)
    return tuple(results)  # type: ignore[return-value]
