"""CLI for scoring transactions with the serialized fraud model artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fraud_detection.evaluate import predict_model_scores
from fraud_detection.features import prepare_inference_matrix
from fraud_detection.inference import load_ensemble_artifact, score_transactions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score transaction CSV rows and emit cost-based fraud decisions."
    )
    parser.add_argument("input_csv", type=Path, help="Input CSV path containing transaction rows.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <input_stem>_scored.csv.",
    )
    parser.add_argument(
        "--model-artifact",
        type=Path,
        default=Path("data/artifacts/fraud_model.joblib"),
        help="Path to serialized artifact created by notebook Section 10.3.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override decision threshold. Defaults to artifact optimal_threshold.",
    )
    parser.add_argument(
        "--time-col",
        default="TransactionDT",
        help="Name of elapsed-seconds time column for temporal proxy features.",
    )
    parser.add_argument(
        "--amount-col",
        default="TransactionAmt",
        help="Name of amount column for amount-derived features.",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        default=False,
        help="Use ensemble artifact (supervised + k-NN anomaly). "
             "Requires an artifact saved with unsupervised components.",
    )
    return parser.parse_args()


def validate_artifact(artifact: dict[str, Any]) -> None:
    required = {"champion_model_name", "model", "feature_columns", "optimal_threshold"}
    missing = sorted(required - set(artifact))
    if missing:
        raise KeyError(f"Artifact is missing required keys: {missing}")


def resolve_output_path(input_csv: Path, output: Path | None) -> Path:
    if output is not None:
        return output
    return input_csv.with_name(f"{input_csv.stem}_scored.csv")


def load_medians(raw_medians: Any) -> pd.Series | None:
    if raw_medians is None:
        return None
    if isinstance(raw_medians, pd.Series):
        return pd.to_numeric(raw_medians, errors="coerce")
    if isinstance(raw_medians, dict):
        return pd.to_numeric(pd.Series(raw_medians), errors="coerce")
    return None


def main() -> int:
    args = parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")
    if not args.model_artifact.exists():
        raise FileNotFoundError(f"Model artifact not found: {args.model_artifact}")

    tx = pd.read_csv(args.input_csv)

    if args.ensemble:
        # --- Ensemble mode: supervised + k-NN anomaly ---
        ensemble_artifact = load_ensemble_artifact(args.model_artifact)
        cfg = {"data": {"time_col": args.time_col, "amount_col": args.amount_col}}
        output_df = score_transactions(tx, ensemble_artifact, cfg)
    else:
        # --- Supervised-only mode (backward compatible) ---
        artifact = joblib.load(args.model_artifact)
        if not isinstance(artifact, dict):
            raise TypeError("Model artifact must be a dictionary.")
        validate_artifact(artifact)

        feature_cols = artifact["feature_columns"]
        medians = load_medians(artifact.get("imputation_medians"))

        x_matrix = prepare_inference_matrix(
            tx,
            feature_cols,
            medians,
            time_col=args.time_col,
            amount_col=args.amount_col,
        )

        model_name = artifact["champion_model_name"]
        model = artifact["model"]
        scaler = artifact.get("logreg_scaler")

        scores = predict_model_scores(
            model_name,
            x_matrix,
            logreg_model=model if model_name == "LogisticRegression" else None,
            xgb_model=model if model_name == "XGBoost" else None,
            logreg_scaler=scaler,
        )

        threshold = float(args.threshold) if args.threshold is not None else float(artifact["optimal_threshold"])
        decisions = (scores >= threshold).astype("int8")

        output_df = tx.copy()
        output_df["fraud_score"] = scores.astype(float)
        output_df["decision"] = decisions
        output_df["decision_label"] = np.where(decisions == 1, "flag_fraud", "allow")
        output_df["decision_threshold"] = threshold
        output_df["model_name"] = model_name

    output_path = resolve_output_path(args.input_csv, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    n_flagged = int((output_df["decision"] == 1).sum()) if "decision" in output_df.columns else 0
    used_threshold = output_df["decision_threshold"].iloc[0] if "decision_threshold" in output_df.columns else "N/A"
    print(f"Rows scored: {len(output_df):,}")
    print(f"Threshold used: {used_threshold}")
    print(f"Flagged rows: {n_flagged:,}")
    print(f"Output written: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
