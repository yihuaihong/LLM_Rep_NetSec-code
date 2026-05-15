"""
Train probing classifiers across all layers to measure representation separation.
Usage: python scripts/train_probes.py --reps_dir results/representations/cicids2017/Meta-Llama-3-8B-Instruct
"""

import argparse
import os
import sys
import torch
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.probe_utils import layer_sweep, evaluate_probe, train_logistic_probe
from utils.evaluation_utils import compute_binary_metrics, compute_generalization_gap, save_metrics


def main(args):
    reps_dir = Path(args.reps_dir)

    # Load train split
    train_data = torch.load(reps_dir / "train.pt", weights_only=False)
    train_hs = train_data["hidden_states"]
    train_labels = train_data["labels"].numpy()

    print(f"Train: {len(train_labels)} samples, {train_labels.sum()} attacks")
    print(f"Layers available: {sorted(train_hs.keys())}")

    # Layer sweep on training data
    print("\n=== Layer Sweep (CV on train) ===")
    sweep_results = layer_sweep(train_hs, train_labels)

    # Evaluate on known test split
    test_known_path = reps_dir / "test_known.pt"
    if test_known_path.exists():
        test_known = torch.load(test_known_path, weights_only=False)
        print("\n=== Evaluation on Known Attacks ===")
        best_layer = max(sweep_results, key=lambda l: sweep_results[l]["cv_accuracy"])
        print(f"Best layer: {best_layer}")

        clf = sweep_results[best_layer]["model"]
        X_test = test_known["hidden_states"][best_layer].numpy()
        y_test = test_known["labels"].numpy()
        known_results = evaluate_probe(clf, X_test, y_test)
        print(f"  Known test accuracy: {known_results['accuracy']:.4f}, AUROC: {known_results.get('auroc', 'N/A')}")

    # Evaluate on holdout (zero-day) split
    test_holdout_path = reps_dir / "test_holdout.pt"
    if test_holdout_path.exists():
        test_holdout = torch.load(test_holdout_path, weights_only=False)
        print("\n=== Evaluation on Holdout (Zero-Day) Attacks ===")

        X_holdout = test_holdout["hidden_states"][best_layer].numpy()
        y_holdout = test_holdout["labels"].numpy()
        holdout_results = evaluate_probe(clf, X_holdout, y_holdout)
        print(f"  Holdout accuracy: {holdout_results['accuracy']:.4f}, AUROC: {holdout_results.get('auroc', 'N/A')}")

        gap = compute_generalization_gap(known_results, holdout_results)
        print(f"  Generalization gap (accuracy): {gap['accuracy_gap']:.4f}")

    # Save all results
    output_dir = Path(args.output_dir) if args.output_dir else reps_dir.parent.parent / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        "layer_sweep": {
            int(k): {"cv_accuracy": v["cv_accuracy"], "cv_auroc": v["cv_auroc"]}
            for k, v in sweep_results.items()
        },
        "best_layer": int(best_layer) if "best_layer" in dir() else None,
    }
    if test_known_path.exists():
        all_results["known_test"] = known_results
    if test_holdout_path.exists():
        all_results["holdout_test"] = holdout_results
        all_results["generalization_gap"] = gap

    model_name = reps_dir.name
    save_metrics(all_results, str(output_dir / f"probe_results_{model_name}.json"))
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps_dir", type=str, required=True,
                        help="Path to directory with train.pt, test_known.pt, test_holdout.pt")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    main(args)
