"""
Visualization utilities for representation space analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from pathlib import Path
from typing import Optional


def reduce_dims(
    X: np.ndarray,
    method: str = "pca",
    n_components: int = 2,
    labels: np.ndarray = None,
    **kwargs,
) -> np.ndarray:
    """Reduce dimensionality using PCA, t-SNE, UMAP, or LDA."""
    if method == "pca":
        reducer = PCA(n_components=n_components)
        return reducer.fit_transform(X)
    elif method == "lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        lda = LinearDiscriminantAnalysis(n_components=min(n_components, len(np.unique(labels)) - 1))
        X_lda = lda.fit_transform(X, labels)
        if X_lda.shape[1] == 1:
            # LDA with 2 classes gives 1D; add PCA second dim for visualization
            pca = PCA(n_components=1)
            X_pca_1d = pca.fit_transform(X)
            X_lda = np.column_stack([X_lda, X_pca_1d])
        return X_lda
    elif method == "tsne":
        perplexity = kwargs.get("perplexity", 30)
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        return reducer.fit_transform(X)
    elif method == "umap":
        import umap
        n_neighbors = kwargs.get("n_neighbors", 15)
        reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=42)
        return reducer.fit_transform(X)
    else:
        raise ValueError(f"Unknown method: {method}")


def plot_2d_scatter(
    X_2d: np.ndarray,
    labels: np.ndarray,
    label_names: Optional[dict] = None,
    title: str = "Representation Space",
    figsize: tuple = (10, 8),
    alpha: float = 0.5,
    save_path: Optional[str] = None,
):
    """
    Plot 2D scatter of representations colored by label.

    Args:
        X_2d: (n_samples, 2) reduced representations
        labels: integer labels for each sample
        label_names: dict mapping label int -> name string
        title: plot title
        save_path: if provided, save figure to this path
    """
    if label_names is None:
        label_names = {0: "Normal", 1: "Attack"}

    fig, ax = plt.subplots(figsize=figsize)

    for label_val in sorted(np.unique(labels)):
        mask = labels == label_val
        name = label_names.get(label_val, str(label_val))
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], label=name, alpha=alpha, s=10)

    ax.set_title(title)
    ax.legend(markerscale=3)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_by_attack_type(
    X_2d: np.ndarray,
    attack_types: np.ndarray,
    title: str = "Representation Space by Attack Type",
    figsize: tuple = (12, 9),
    alpha: float = 0.5,
    save_path: Optional[str] = None,
):
    """Plot 2D scatter colored by specific attack type."""
    fig, ax = plt.subplots(figsize=figsize)

    unique_types = sorted(np.unique(attack_types))
    palette = sns.color_palette("husl", n_colors=len(unique_types))

    for i, atype in enumerate(unique_types):
        mask = attack_types == atype
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], label=atype,
                   color=palette[i], alpha=alpha, s=10)

    ax.set_title(title)
    ax.legend(markerscale=3, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_layer_accuracy_curve(
    layer_results: dict[int, dict],
    metric: str = "cv_accuracy",
    title: str = "Probe Accuracy Across Layers",
    save_path: Optional[str] = None,
):
    """Plot probe accuracy as a function of layer index."""
    layers = sorted(layer_results.keys())
    values = [layer_results[l][metric] for l in layers]
    stds = [layer_results[l].get(f"{metric}_std", 0) for l in layers]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, values, marker="o", linewidth=2)
    ax.fill_between(layers,
                    [v - s for v, s in zip(values, stds)],
                    [v + s for v, s in zip(values, stds)],
                    alpha=0.2)
    ax.set_xlabel("Layer")
    ax.set_ylabel(metric.replace("cv_", "").replace("_", " ").title())
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_direction_histogram(
    projections_normal: np.ndarray,
    projections_attack: np.ndarray,
    title: str = "Projection onto Attack Direction",
    save_path: Optional[str] = None,
):
    """Plot histogram of projections onto the attack direction vector."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(projections_normal, bins=50, alpha=0.6, label="Normal", density=True)
    ax.hist(projections_attack, bins=50, alpha=0.6, label="Attack", density=True)
    ax.set_xlabel("Projection Value")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def compute_separation_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    """
    Compute clustering/separation metrics for representations.

    Returns:
        dict with silhouette_score, davies_bouldin_score
    """
    return {
        "silhouette": silhouette_score(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels),
    }
