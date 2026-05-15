"""
End-to-end pipeline for UNSW-NB15 dataset:
1. Load data & create generalization splits
2. Extract representations from Llama-3
3. Run probing classifiers (layer sweep)
4. Generate direction histograms + LDA plots (train + holdout)
5. Save metrics and compile into PDF
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import yaml
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from pathlib import Path

from utils.data_utils import load_dataset, format_dataset, create_generalization_splits
from utils.model_utils import load_model_and_tokenizer, extract_hidden_states, flush, set_random_seed
from utils.probe_utils import (
    train_logistic_probe, evaluate_probe, layer_sweep,
    extract_separation_direction, project_onto_direction,
)
from utils.evaluation_utils import compute_generalization_gap, compute_binary_metrics

# ── Config ──
MODEL_NAME = "Meta-Llama-3-8B-Instruct"
CACHE_DIR = "/scratch/yh6210/transformers"
DATASET_NAME = "unsw_nb15"
DATASET_PATH = "/scratch/yh6210/datasets/LLM_Rep_NetSec_datasets/unsw_nb15/"
HOLDOUT_TYPES = ["Shellcode", "Worms", "Backdoor"]
MAX_SAMPLES = 5000
LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31]
VIS_LAYERS = [0, 8, 16, 31]
BATCH_SIZE = 32
MAX_SEQ_LEN = 512

BASE = Path(__file__).resolve().parent.parent
REPS_DIR = BASE / "results" / "representations" / DATASET_NAME / MODEL_NAME
FIGS_DIR = BASE / "results" / "figures"
METRICS_DIR = BASE / "results" / "metrics"
OUT_PDF = BASE / "results" / f"llama3_{DATASET_NAME}_results.pdf"


def step1_extract():
    """Extract representations if not already cached."""
    if (REPS_DIR / "train.pt").exists():
        print("Representations already cached, loading...")
        train = torch.load(REPS_DIR / "train.pt", weights_only=False)
        known = torch.load(REPS_DIR / "test_known.pt", weights_only=False)
        holdout = torch.load(REPS_DIR / "test_holdout.pt", weights_only=False)
        return train, known, holdout

    print(f"Loading dataset: {DATASET_NAME}")
    df = load_dataset(DATASET_NAME, DATASET_PATH)
    df = format_dataset(df, DATASET_NAME, fmt="natural_language")

    print(f"Attack types: {df['attack_type'].value_counts().to_dict()}")
    print(f"Holdout types: {HOLDOUT_TYPES}")

    splits = create_generalization_splits(
        df, holdout_attack_types=HOLDOUT_TYPES, max_samples_per_class=MAX_SAMPLES
    )
    for k, v in splits.items():
        print(f"  {k}: {len(v)} samples, attacks={v['is_attack'].sum()}")
        print(f"    types: {v['attack_type'].value_counts().to_dict()}")

    print(f"\nLoading model: {MODEL_NAME}")
    model_path = os.path.join(CACHE_DIR, MODEL_NAME)
    model, tokenizer = load_model_and_tokenizer(model_path, dtype="float16", device="cuda")

    REPS_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for split_name, split_df in splits.items():
        print(f"\nExtracting {split_name}: {len(split_df)} samples")
        hidden_states = extract_hidden_states(
            model, tokenizer, split_df["text"].tolist(),
            layers=LAYERS, token_position="last",
            batch_size=BATCH_SIZE, max_seq_length=MAX_SEQ_LEN,
        )
        save_data = {
            "hidden_states": hidden_states,
            "labels": torch.tensor(split_df["is_attack"].values),
            "attack_types": split_df["attack_type"].values,
        }
        torch.save(save_data, REPS_DIR / f"{split_name}.pt")
        results[split_name] = save_data
        flush()

    del model, tokenizer
    flush()
    return results["train"], results["test_known"], results["test_holdout"]


def step2_probe_and_metrics(train, known, holdout):
    """Run probing classifiers and compute metrics."""
    y_train = train["labels"].numpy()
    y_known = known["labels"].numpy()
    y_holdout = holdout["labels"].numpy()

    print("\n=== Layer Sweep (train CV) ===")
    sweep = {}
    for layer in sorted(train["hidden_states"].keys()):
        X = train["hidden_states"][layer].numpy()
        result = train_logistic_probe(X, y_train)
        sweep[layer] = result
        print(f"  Layer {layer:3d}: acc={result['cv_accuracy']:.4f} auroc={result['cv_auroc']:.4f}")

    # Find best layer
    best_layer = max(sweep, key=lambda l: sweep[l]["cv_auroc"])
    print(f"\nBest layer: {best_layer}")

    # Evaluate on known and holdout
    X_train_best = train["hidden_states"][best_layer].numpy()
    probe = train_logistic_probe(X_train_best, y_train)

    X_known = known["hidden_states"][best_layer].numpy()
    known_eval = evaluate_probe(probe["model"], X_known, y_known)
    print(f"Known: acc={known_eval['accuracy']:.4f} auroc={known_eval.get('auroc', 'N/A')}")

    X_holdout = holdout["hidden_states"][best_layer].numpy()
    holdout_eval = evaluate_probe(probe["model"], X_holdout, y_holdout)
    print(f"Holdout: acc={holdout_eval['accuracy']:.4f} auroc={holdout_eval.get('auroc', 'N/A')}")

    gap = compute_generalization_gap(known_eval, holdout_eval)
    print(f"Gap: acc={gap['accuracy_gap']:.4f} auroc={gap['auroc_gap']:.4f}")

    # Direction-based classification
    X_train_best_t = train["hidden_states"][best_layer]
    direction = extract_separation_direction(X_train_best_t, y_train)
    proj_all = project_onto_direction(holdout["hidden_states"][best_layer], direction)
    proj_n = project_onto_direction(X_train_best_t[y_train == 0], direction)
    proj_a = project_onto_direction(X_train_best_t[y_train == 1], direction)
    threshold = (proj_n.mean() + proj_a.mean()) / 2
    y_pred_dir = (proj_all > threshold).astype(int)
    dir_acc = (y_pred_dir == y_holdout).mean()
    print(f"Direction-based holdout acc: {dir_acc:.4f}")

    # Save metrics
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {
        "model": MODEL_NAME,
        "dataset": DATASET_NAME,
        "best_layer": int(best_layer),
        "known_accuracy": known_eval["accuracy"],
        "known_auroc": known_eval.get("auroc"),
        "holdout_accuracy": holdout_eval["accuracy"],
        "holdout_auroc": holdout_eval.get("auroc"),
        "generalization_gap": gap,
        "direction_accuracy": dir_acc,
        "layer_sweep": {
            int(l): {"cv_accuracy": r["cv_accuracy"], "cv_auroc": r["cv_auroc"]}
            for l, r in sweep.items()
        },
    }
    with open(METRICS_DIR / f"{MODEL_NAME}_{DATASET_NAME}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return sweep, best_layer, metrics


def make_direction_hist(ax, proj0, proj1, title, label0="Normal", label1="Attack", show_acc=False):
    ax.hist(proj0, bins=50, alpha=0.6, label=label0, density=True, color="steelblue")
    ax.hist(proj1, bins=50, alpha=0.6, label=label1, density=True, color="darkorange")
    if show_acc:
        threshold = (proj0.mean() + proj1.mean()) / 2
        all_proj = np.concatenate([proj0, proj1])
        all_labels = np.concatenate([np.zeros(len(proj0)), np.ones(len(proj1))])
        pred = (all_proj > threshold).astype(int)
        acc = (pred == all_labels).mean()
        ax.text(0.02, 0.95, f"Acc: {acc:.3f}", transform=ax.transAxes,
                fontsize=9, va="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    ax.set_xlabel("Projection Value", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)


def make_lda_scatter(ax, X, y, title, fit_X=None, fit_y=None):
    label_names = {0: "Normal", 1: "Attack"}
    lda = LinearDiscriminantAnalysis(n_components=1)
    pca = PCA(n_components=1)
    if fit_X is not None:
        lda.fit(fit_X, fit_y)
        pca.fit(fit_X)
    else:
        lda.fit(X, y)
        pca.fit(X)
    X_2d = np.column_stack([lda.transform(X), pca.transform(X)])
    for lv in sorted(np.unique(y)):
        mask = y == lv
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], label=label_names[lv], alpha=0.4, s=8,
                   color="steelblue" if lv == 0 else "darkorange")
    ax.set_xlabel("LDA Dim 1", fontsize=9)
    ax.set_ylabel("PCA Dim 1", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, markerscale=3)


def step3_visualize_and_pdf(train, known, holdout, sweep, best_layer, metrics):
    """Generate all figures and compile into PDF."""
    y_train = train["labels"].numpy()
    y_holdout = holdout["labels"].numpy()

    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    with PdfPages(str(OUT_PDF)) as pdf:
        # Title page
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis("off")
        ax.text(0.5, 0.65, "LLM Representation Analysis — UNSW-NB15", fontsize=24,
                ha="center", va="center", fontweight="bold")
        ax.text(0.5, 0.50, MODEL_NAME, fontsize=18, ha="center", va="center", color="gray")
        ax.text(0.5, 0.38, f"Holdout attack types: {', '.join(HOLDOUT_TYPES)}", fontsize=14,
                ha="center", va="center")
        ax.text(0.5, 0.28, f"Best layer: {best_layer} | "
                f"Known acc: {metrics['known_accuracy']:.3f} | "
                f"Holdout acc: {metrics['holdout_accuracy']:.3f}", fontsize=13,
                ha="center", va="center", color="steelblue")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Layer accuracy curve
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        layers_sorted = sorted(sweep.keys())
        accs = [sweep[l]["cv_accuracy"] for l in layers_sorted]
        aurocs = [sweep[l]["cv_auroc"] for l in layers_sorted]
        axes[0].plot(layers_sorted, accs, "o-", linewidth=2)
        axes[0].set_xlabel("Layer"); axes[0].set_ylabel("CV Accuracy")
        axes[0].set_title("Probe Accuracy Across Layers"); axes[0].grid(True, alpha=0.3)
        axes[1].plot(layers_sorted, aurocs, "o-", linewidth=2, color="darkorange")
        axes[1].set_xlabel("Layer"); axes[1].set_ylabel("CV AUROC")
        axes[1].set_title("Probe AUROC Across Layers"); axes[1].grid(True, alpha=0.3)
        fig.suptitle(f"UNSW-NB15 — {MODEL_NAME}", fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Mean Diff Direction - Train (4 layers)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("UNSW-NB15 — Mean Diff Direction — Train (Known Attacks)",
                     fontsize=14, fontweight="bold")
        for idx, layer in enumerate(VIS_LAYERS):
            ax = axes[idx // 2, idx % 2]
            X = train["hidden_states"][layer]
            direction = extract_separation_direction(X, y_train)
            p0 = project_onto_direction(X[y_train == 0], direction)
            p1 = project_onto_direction(X[y_train == 1], direction)
            make_direction_hist(ax, p0, p1, f"Layer {layer}", show_acc=True)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # LDA Scatter - Train (4 layers)
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle("UNSW-NB15 — LDA Scatter — Train (Known Attacks)",
                     fontsize=14, fontweight="bold")
        for idx, layer in enumerate(VIS_LAYERS):
            ax = axes[idx // 2, idx % 2]
            X = train["hidden_states"][layer].numpy()
            make_lda_scatter(ax, X, y_train, f"Layer {layer}")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Mean Diff Direction - Holdout (4 layers)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("UNSW-NB15 — Mean Diff Direction — Holdout (Zero-Day Generalization)",
                     fontsize=14, fontweight="bold")
        for idx, layer in enumerate(VIS_LAYERS):
            ax = axes[idx // 2, idx % 2]
            X_tr = train["hidden_states"][layer]
            X_ho = holdout["hidden_states"][layer]
            direction = extract_separation_direction(X_tr, y_train)
            p0 = project_onto_direction(X_ho[y_holdout == 0], direction)
            p1 = project_onto_direction(X_ho[y_holdout == 1], direction)
            make_direction_hist(ax, p0, p1, f"Layer {layer} (direction from train)", show_acc=True)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # LDA Scatter - Holdout (4 layers)
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle("UNSW-NB15 — LDA Scatter — Holdout (Zero-Day Generalization)",
                     fontsize=14, fontweight="bold")
        for idx, layer in enumerate(VIS_LAYERS):
            ax = axes[idx // 2, idx % 2]
            X_tr = train["hidden_states"][layer].numpy()
            X_ho = holdout["hidden_states"][layer].numpy()
            make_lda_scatter(ax, X_ho, y_holdout, f"Layer {layer} (LDA fit on train)",
                           fit_X=X_tr, fit_y=y_train)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Per holdout attack type breakdown
        fig, ax = plt.subplots(figsize=(12, 6))
        holdout_types = holdout["attack_types"]
        unique_atk = np.unique(holdout_types[y_holdout == 1])
        best_X_ho = holdout["hidden_states"][best_layer]
        best_X_tr = train["hidden_states"][best_layer]
        direction = extract_separation_direction(best_X_tr, y_train)

        type_data = {}
        for atype in unique_atk:
            mask = holdout_types == atype
            proj = project_onto_direction(best_X_ho[mask], direction)
            type_data[atype] = proj

        proj_normal = project_onto_direction(best_X_ho[y_holdout == 0], direction)

        all_types = ["Normal"] + list(unique_atk)
        all_means = [proj_normal.mean()] + [type_data[t].mean() for t in unique_atk]
        all_stds = [proj_normal.std()] + [type_data[t].std() for t in unique_atk]
        colors = ["steelblue"] + ["darkorange"] * len(unique_atk)

        ax.barh(all_types, all_means, xerr=all_stds, color=colors, alpha=0.7, capsize=5)
        ax.set_xlabel(f"Mean Projection onto Attack Direction (Layer {best_layer})")
        ax.set_title(f"UNSW-NB15 — Per Attack Type Projection (Holdout)")
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"\nPDF saved to: {OUT_PDF}")


def main():
    set_random_seed(42)
    train, known, holdout = step1_extract()
    sweep, best_layer, metrics = step2_probe_and_metrics(train, known, holdout)
    step3_visualize_and_pdf(train, known, holdout, sweep, best_layer, metrics)
    print("\nDone!")


if __name__ == "__main__":
    main()
