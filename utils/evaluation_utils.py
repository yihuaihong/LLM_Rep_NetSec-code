"""
Evaluation utilities for measuring representation separation quality.
"""

import numpy as np
import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> dict:
    """Compute standard binary classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(y_true, y_pred, output_dict=True),
    }
    if y_prob is not None:
        metrics["auroc"] = roc_auc_score(y_true, y_prob)
    return metrics


def compute_generalization_gap(known_metrics: dict, holdout_metrics: dict) -> dict:
    """Compute the gap between known-attack and holdout-attack performance."""
    return {
        "accuracy_gap": known_metrics["accuracy"] - holdout_metrics["accuracy"],
        "auroc_gap": known_metrics.get("auroc", 0) - holdout_metrics.get("auroc", 0),
        "known_accuracy": known_metrics["accuracy"],
        "holdout_accuracy": holdout_metrics["accuracy"],
        "known_auroc": known_metrics.get("auroc"),
        "holdout_auroc": holdout_metrics.get("auroc"),
    }


def save_metrics(metrics: dict, save_path: str):
    """Save metrics dict to JSON file."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    cleaned = json.loads(json.dumps(metrics, default=convert))
    with open(path, "w") as f:
        json.dump(cleaned, f, indent=2)


def load_metrics(path: str) -> dict:
    """Load metrics from JSON file."""
    with open(path) as f:
        return json.load(f)
