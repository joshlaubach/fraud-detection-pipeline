"""Visualization helpers: anomaly plots, cluster heatmaps, ensemble boundaries."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Anomaly score visualizations
# ---------------------------------------------------------------------------

def plot_anomaly_pca(
    X_scaled: np.ndarray,
    anomaly_scores: np.ndarray,
    y: np.ndarray | None = None,
    title: str = "Anomaly Scores (PCA Projection)",
) -> tuple[plt.Figure, np.ndarray]:
    """2D PCA scatter colored by anomaly score, with optional fraud overlay.

    Returns (fig, axs).
    """
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)

    n_plots = 2 if y is not None else 1
    fig, axs = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))
    if n_plots == 1:
        axs = np.array([axs])

    sc = axs[0].scatter(
        X_2d[:, 0], X_2d[:, 1], c=anomaly_scores,
        cmap="YlOrRd", s=3, alpha=0.5,
    )
    axs[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axs[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    axs[0].set_title(title)
    plt.colorbar(sc, ax=axs[0], label="Anomaly Score")

    if y is not None:
        y_arr = np.asarray(y).astype(int)
        colors = np.where(y_arr == 1, "red", "steelblue")
        axs[1].scatter(X_2d[:, 0], X_2d[:, 1], c=colors, s=3, alpha=0.4)
        axs[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        axs[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        axs[1].set_title("Fraud Overlay (red = fraud)")

    plt.tight_layout()
    return fig, axs


def plot_anomaly_tsne(
    X_scaled: np.ndarray,
    anomaly_scores: np.ndarray,
    y: np.ndarray | None = None,
    perplexity: int = 30,
    title: str = "Anomaly Scores (t-SNE Projection)",
) -> tuple[plt.Figure, np.ndarray]:
    """2D t-SNE scatter colored by anomaly score."""
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_2d = tsne.fit_transform(X_scaled)

    n_plots = 2 if y is not None else 1
    fig, axs = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))
    if n_plots == 1:
        axs = np.array([axs])

    sc = axs[0].scatter(
        X_2d[:, 0], X_2d[:, 1], c=anomaly_scores,
        cmap="YlOrRd", s=3, alpha=0.5,
    )
    axs[0].set_xlabel("t-SNE 1")
    axs[0].set_ylabel("t-SNE 2")
    axs[0].set_title(title)
    plt.colorbar(sc, ax=axs[0], label="Anomaly Score")

    if y is not None:
        y_arr = np.asarray(y).astype(int)
        colors = np.where(y_arr == 1, "red", "steelblue")
        axs[1].scatter(X_2d[:, 0], X_2d[:, 1], c=colors, s=3, alpha=0.4)
        axs[1].set_xlabel("t-SNE 1")
        axs[1].set_ylabel("t-SNE 2")
        axs[1].set_title("Fraud Overlay (red = fraud)")

    plt.tight_layout()
    return fig, axs


def plot_anomaly_distribution(
    scores: np.ndarray,
    y: np.ndarray,
    title: str = "Anomaly Score Distribution by Class",
) -> tuple[plt.Figure, plt.Axes]:
    """Histogram of anomaly scores, split by fraud/non-fraud."""
    y_arr = np.asarray(y).astype(int)
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))

    axs.hist(
        scores[y_arr == 0], bins=50, alpha=0.6, label="Non-fraud",
        color="steelblue", density=True,
    )
    axs.hist(
        scores[y_arr == 1], bins=50, alpha=0.6, label="Fraud",
        color="red", density=True,
    )
    axs.set_xlabel("k-NN Anomaly Score")
    axs.set_ylabel("Density")
    axs.set_title(title)
    axs.legend()
    axs.grid(alpha=0.3)

    plt.tight_layout()
    return fig, axs


def plot_anomaly_fraud_rate(
    scores: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
    title: str = "Fraud Rate by Anomaly Score Bin",
) -> tuple[plt.Figure, plt.Axes]:
    """Bar chart of fraud rate per anomaly score bin."""
    y_arr = np.asarray(y).astype(int)
    df = pd.DataFrame({"score": scores, "label": y_arr})
    df["bin"] = pd.qcut(df["score"], q=n_bins, labels=False, duplicates="drop")

    grouped = df.groupby("bin").agg(
        count=("label", "count"),
        fraud_rate=("label", "mean"),
    ).reset_index()

    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    bars = axs.bar(grouped["bin"], grouped["fraud_rate"], color="coral", edgecolor="black")
    overall_rate = y_arr.mean()
    axs.axhline(overall_rate, color="navy", linestyle="--", label=f"Overall: {overall_rate:.4f}")
    axs.set_xlabel("Anomaly Score Bin (0=lowest, higher=more anomalous)")
    axs.set_ylabel("Fraud Rate")
    axs.set_title(title)
    axs.legend()
    axs.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    return fig, axs


# ---------------------------------------------------------------------------
# k selection
# ---------------------------------------------------------------------------

def plot_knn_k_selection(
    results: list[dict[str, Any]],
    title: str = "k-NN Anomaly ROC-AUC vs k",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot anomaly ROC-AUC across different k values.

    Parameters
    ----------
    results : list of dict
        Output of ``unsupervised.select_k()["results"]``.
    """
    k_vals = [r["k"] for r in results]
    aucs = [r["roc_auc"] for r in results]

    fig, axs = plt.subplots(1, 1, figsize=(8, 5))
    axs.plot(k_vals, aucs, "o-", linewidth=2, markersize=8, color="teal")
    best_idx = int(np.argmax(aucs))
    axs.plot(k_vals[best_idx], aucs[best_idx], "r*", markersize=15, label=f"Best k={k_vals[best_idx]}")
    axs.set_xlabel("k (number of neighbors)")
    axs.set_ylabel("Anomaly ROC-AUC")
    axs.set_title(title)
    axs.legend()
    axs.grid(alpha=0.3)

    plt.tight_layout()
    return fig, axs


# ---------------------------------------------------------------------------
# DBSCAN validation
# ---------------------------------------------------------------------------

def plot_dbscan_sensitivity(
    results: list[dict[str, Any]],
    title: str = "DBSCAN Parameter Sensitivity",
) -> tuple[plt.Figure, np.ndarray]:
    """Heatmaps of cluster count and noise ratio over eps x min_samples."""
    df = pd.DataFrame(results)
    eps_vals = sorted(df["eps"].unique())
    ms_vals = sorted(df["min_samples"].unique())

    cluster_grid = np.full((len(eps_vals), len(ms_vals)), np.nan)
    noise_grid = np.full((len(eps_vals), len(ms_vals)), np.nan)

    for _, row in df.iterrows():
        i = eps_vals.index(row["eps"])
        j = ms_vals.index(row["min_samples"])
        cluster_grid[i, j] = row["n_clusters"]
        noise_grid[i, j] = row["noise_ratio"]

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    im1 = axs[0].imshow(cluster_grid, cmap="viridis", aspect="auto")
    axs[0].set_xticks(range(len(ms_vals)))
    axs[0].set_yticks(range(len(eps_vals)))
    axs[0].set_xticklabels(ms_vals)
    axs[0].set_yticklabels([f"{e:.1f}" for e in eps_vals])
    axs[0].set_xlabel("min_samples")
    axs[0].set_ylabel("eps")
    axs[0].set_title("Number of Clusters")
    plt.colorbar(im1, ax=axs[0])

    im2 = axs[1].imshow(noise_grid, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
    axs[1].set_xticks(range(len(ms_vals)))
    axs[1].set_yticks(range(len(eps_vals)))
    axs[1].set_xticklabels(ms_vals)
    axs[1].set_yticklabels([f"{e:.1f}" for e in eps_vals])
    axs[1].set_xlabel("min_samples")
    axs[1].set_ylabel("eps")
    axs[1].set_title("Noise Ratio")
    plt.colorbar(im2, ax=axs[1])

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig, axs


# ---------------------------------------------------------------------------
# Ensemble decision boundary
# ---------------------------------------------------------------------------

def plot_ensemble_decision_boundary(
    supervised_scores: np.ndarray,
    anomaly_scores: np.ndarray,
    y: np.ndarray,
    threshold: float,
    weights: tuple[float, float] = (0.7, 0.3),
    title: str = "Ensemble Decision Boundary",
) -> tuple[plt.Figure, plt.Axes]:
    """Scatter of supervised vs anomaly scores with threshold contour.

    The decision boundary is a line where:
        w_s * supervised + w_a * anomaly = threshold
    =>  anomaly = (threshold - w_s * supervised) / w_a
    """
    w_s, w_a = weights
    y_arr = np.asarray(y).astype(int)

    fig, axs = plt.subplots(1, 1, figsize=(10, 8))

    # Scatter fraud vs non-fraud
    axs.scatter(
        supervised_scores[y_arr == 0], anomaly_scores[y_arr == 0],
        c="steelblue", s=5, alpha=0.3, label="Non-fraud",
    )
    axs.scatter(
        supervised_scores[y_arr == 1], anomaly_scores[y_arr == 1],
        c="red", s=20, alpha=0.7, label="Fraud",
    )

    # Decision boundary line
    if w_a > 0:
        x_line = np.linspace(0, 1, 100)
        y_line = (threshold - w_s * x_line) / w_a
        valid = (y_line >= 0) & (y_line <= 1)
        axs.plot(
            x_line[valid], y_line[valid], "k--", linewidth=2,
            label=f"Threshold = {threshold:.4f}",
        )

    axs.set_xlabel("Supervised Score (XGBoost P(fraud))")
    axs.set_ylabel("k-NN Anomaly Score")
    axs.set_title(title)
    axs.set_xlim([-0.02, 1.02])
    axs.set_ylim([-0.02, 1.02])
    axs.legend(loc="upper left")
    axs.grid(alpha=0.3)

    plt.tight_layout()
    return fig, axs


# ---------------------------------------------------------------------------
# Anomaly feature heatmap
# ---------------------------------------------------------------------------

def plot_anomaly_feature_heatmap(
    anomaly_profiles: pd.DataFrame,
    title: str = "Feature Means by Anomaly Score Bin",
) -> tuple[plt.Figure, plt.Axes]:
    """Heatmap of normalized feature means per anomaly score bin."""
    # Select only mean_* columns
    mean_cols = [c for c in anomaly_profiles.columns if c.startswith("mean_")]
    if not mean_cols:
        fig, axs = plt.subplots(1, 1, figsize=(10, 6))
        axs.text(0.5, 0.5, "No feature means found", ha="center", va="center")
        return fig, axs

    heat_data = anomaly_profiles[mean_cols].copy()
    # Normalize each column to [0, 1] for visualization
    for col in heat_data.columns:
        col_range = heat_data[col].max() - heat_data[col].min()
        if col_range > 0:
            heat_data[col] = (heat_data[col] - heat_data[col].min()) / col_range

    # Clean column labels
    heat_data.columns = [c.replace("mean_", "") for c in heat_data.columns]
    heat_data.index = anomaly_profiles["anomaly_bin"].values

    fig, axs = plt.subplots(1, 1, figsize=(max(10, len(mean_cols) * 0.8), 6))
    im = axs.imshow(heat_data.values, cmap="YlOrRd", aspect="auto")
    axs.set_xticks(range(len(heat_data.columns)))
    axs.set_xticklabels(heat_data.columns, rotation=45, ha="right")
    axs.set_yticks(range(len(heat_data.index)))
    axs.set_yticklabels([f"Bin {b}" for b in heat_data.index])
    axs.set_xlabel("Feature")
    axs.set_ylabel("Anomaly Score Bin")
    axs.set_title(title)
    plt.colorbar(im, ax=axs, label="Normalized Mean")

    plt.tight_layout()
    return fig, axs
