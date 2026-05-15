"""
Full generalization experiment: train probes on known attacks, test on held-out attack types.
Also tests cross-dataset transfer and concept vector analysis.

Usage: python scripts/run_generalization.py --reps_dir results/representations/cicids2017/Meta-Llama-3-8B-Instruct
"""

import argparse
import os
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.probe_utils import (
    train_logistic_probe,
    evaluate_probe,
    extract_separation_direction,
    project_onto_direction,
)
from utils.visualization_utils import (
    reduce_dims,
    plot_2d_scatter,
    plot_by_attack_type,
    plot_direction_histogram,
    compute_separation_metrics,
)
from utils.evaluation_utils import save_metrics


def main(args):
    reps_dir = Path(args.reps_dir)
    fig_dir = Path(args.fig_dir) if args.fig_dir else reps_dir.parent.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_data = torch.load(reps_dir / "train.pt", weights_only=False)
    test_holdout = torch.load(reps_dir / "test_holdout.pt", weights_only=False)

    layer = args.layer

    X_train = train_data["hidden_states"][layer]
    y_train = train_data["labels"].numpy()
    types_train = train_data["attack_types"]

    X_holdout = test_holdout["hidden_states"][layer]
    y_holdout = test_holdout["labels"].numpy()
    types_holdout = test_holdout["attack_types"]

    print(f"Layer {layer}: train={len(y_train)}, holdout={len(y_holdout)}")

    # 1. Train probe on known attacks
    probe_result = train_logistic_probe(X_train.numpy(), y_train)
    print(f"Train CV accuracy: {probe_result['cv_accuracy']:.4f}")

    # Evaluate on holdout
    holdout_eval = evaluate_probe(probe_result["model"], X_holdout.numpy(), y_holdout)
    print(f"Holdout accuracy: {holdout_eval['accuracy']:.4f}")

    # 2. Extract attack direction from known attacks
    direction = extract_separation_direction(X_train, y_train)

    # Project holdout onto this direction
    proj_holdout_normal = project_onto_direction(X_holdout[y_holdout == 0], direction)
    proj_holdout_attack = project_onto_direction(X_holdout[y_holdout == 1], direction)

    plot_direction_histogram(
        proj_holdout_normal, proj_holdout_attack,
        title=f"Holdout Projection onto Attack Direction (Layer {layer})",
        save_path=str(fig_dir / f"holdout_direction_hist_layer{layer}.png"),
    )

    # 3. Visualize holdout in 2D
    X_all = torch.cat([X_train, X_holdout], dim=0).numpy()
    y_all = np.concatenate([y_train, y_holdout])
    types_all = np.concatenate([types_train, types_holdout])

    X_2d = reduce_dims(X_all, method="pca")
    plot_2d_scatter(
        X_2d, y_all,
        title=f"PCA: Known + Holdout Attacks (Layer {layer})",
        save_path=str(fig_dir / f"pca_all_layer{layer}.png"),
    )
    plot_by_attack_type(
        X_2d, types_all,
        title=f"PCA by Attack Type (Layer {layer})",
        save_path=str(fig_dir / f"pca_attack_types_layer{layer}.png"),
    )

    # 4. Separation metrics
    sep = compute_separation_metrics(X_all, y_all)
    print(f"Silhouette: {sep['silhouette']:.4f}, Davies-Bouldin: {sep['davies_bouldin']:.4f}")

    # Save results
    results = {
        "layer": layer,
        "train_cv_accuracy": probe_result["cv_accuracy"],
        "holdout_accuracy": holdout_eval["accuracy"],
        "holdout_auroc": holdout_eval.get("auroc"),
        "separation_metrics": sep,
    }
    save_metrics(results, str(fig_dir.parent / "metrics" / f"generalization_layer{layer}.json"))
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps_dir", type=str, required=True)
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--fig_dir", type=str, default=None)
    args = parser.parse_args()
    main(args)
