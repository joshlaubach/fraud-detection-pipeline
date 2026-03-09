"""Configuration management for the fraud detection pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """Load YAML configuration file with basic validation.

    Parameters
    ----------
    path : str or Path
        Path to the YAML config file.  Relative paths are resolved from
        the current working directory.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(cfg).__name__}")
    return cfg


def get_cost_params(cfg: dict[str, Any]) -> tuple[float, float]:
    """Extract (cost_fn, cost_fp) from config.

    Returns
    -------
    tuple of (float, float)
        (false-negative cost, false-positive cost).
    """
    costs = cfg.get("costs", {})
    return float(costs.get("false_negative", 500.0)), float(costs.get("false_positive", 10.0))


def get_random_seed(cfg: dict[str, Any]) -> int:
    """Extract random seed from config."""
    return int(cfg.get("random_seed", 42))


def get_optimal_threshold(c_fn: float, c_fp: float) -> float:
    """Compute the cost-optimal Bayes decision threshold.

    Formula
    -------
    tau* = C_fp / (C_fn + C_fp)

    At this threshold the expected cost of flagging equals the expected cost
    of allowing a transaction through, so the classifier minimises total
    expected loss.

    Parameters
    ----------
    c_fn : float
        False negative cost (cost of missing a fraud).
    c_fp : float
        False positive cost (cost of a false decline).

    Returns
    -------
    float
        Optimal probability threshold in (0, 1).
    """
    return c_fp / (c_fn + c_fp)


def log_config(cfg: dict[str, Any], output_path: Path | None = None) -> str:
    """Serialize full config to JSON for reproducibility tracking.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary.
    output_path : Path or None
        If provided, write JSON to this file.

    Returns
    -------
    str
        JSON string of the configuration.
    """
    # Convert any non-serializable values (like .inf) to strings
    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, float) and (obj == float("inf") or obj == float("-inf")):
            return str(obj)
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(item) for item in obj]
        return obj

    sanitized = _sanitize(cfg)
    json_str = json.dumps(sanitized, indent=2, default=str)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json_str, encoding="utf-8")

    return json_str
