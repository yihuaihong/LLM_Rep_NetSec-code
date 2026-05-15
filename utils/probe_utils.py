"""
Probing classifiers to quantify separation in representation space.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from typing import Optional


def train_logistic_probe(
    X: np.ndarray,
    y: np.ndarray,
    C: float = 1.0,
    cv_folds: int = 5,
    seed: int = 42,
) -> dict:
    """
    Train a logistic regression probe with cross-validation.

    Pipeline = StandardScaler + LogisticRegression(LBFGS), since unscaled
    LLM hidden states (esp. bf16-cast) make LBFGS very slow to converge.

    Returns dict with 'model' (the fitted Pipeline), 'cv_accuracy', 'cv_auroc',
    'cv_scores'. Pipeline.predict / .predict_proba work as expected.
    """
    pipe = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(C=C, max_iter=2000, random_state=seed, solver="lbfgs", n_jobs=-1),
    )
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    acc_scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=cv_folds)
    auc_scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=cv_folds)

    # Fit on full data for the returned model
    pipe.fit(X, y)

    return {
        "model": pipe,
        "cv_accuracy": acc_scores.mean(),
        "cv_accuracy_std": acc_scores.std(),
        "cv_auroc": auc_scores.mean(),
        "cv_auroc_std": auc_scores.std(),
        "cv_scores": {"accuracy": acc_scores, "auroc": auc_scores},
    }


def evaluate_probe(
    clf,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Evaluate a trained probe on test data."""
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True),
    }
    if y_prob is not None:
        results["auroc"] = roc_auc_score(y_test, y_prob)

    return results


def layer_sweep(
    hidden_states: dict[int, torch.Tensor],
    labels: np.ndarray,
    C: float = 1.0,
    cv_folds: int = 5,
    seed: int = 42,
) -> dict[int, dict]:
    """
    Train probes at every layer and return layer-wise results.

    Args:
        hidden_states: dict mapping layer_idx -> tensor (n_samples, hidden_dim)
        labels: binary labels
        C: regularization
        cv_folds: CV folds
        seed: random seed

    Returns:
        dict mapping layer_idx -> probe results
    """
    results = {}
    for layer_idx in sorted(hidden_states.keys()):
        X = hidden_states[layer_idx].numpy()
        result = train_logistic_probe(X, labels, C=C, cv_folds=cv_folds, seed=seed)
        results[layer_idx] = result
        print(f"Layer {layer_idx:3d}: accuracy={result['cv_accuracy']:.4f} "
              f"(±{result['cv_accuracy_std']:.4f}), "
              f"auroc={result['cv_auroc']:.4f} (±{result['cv_auroc_std']:.4f})")
    return results


def extract_separation_direction(
    hidden_states: torch.Tensor,
    labels: np.ndarray,
) -> torch.Tensor:
    """
    Compute the "attack direction" vector as the mean difference
    between attack and normal representations.

    Args:
        hidden_states: tensor (n_samples, hidden_dim)
        labels: binary labels (0=normal, 1=attack)

    Returns:
        direction vector (hidden_dim,), normalized
    """
    normal_mask = labels == 0
    attack_mask = labels == 1

    normal_mean = hidden_states[normal_mask].mean(dim=0)
    attack_mean = hidden_states[attack_mask].mean(dim=0)

    direction = attack_mean - normal_mean
    direction = direction / direction.norm()

    return direction


def project_onto_direction(
    hidden_states: torch.Tensor,
    direction: torch.Tensor,
) -> np.ndarray:
    """Project representations onto a direction vector. Returns scalar projections."""
    return (hidden_states @ direction).numpy()
