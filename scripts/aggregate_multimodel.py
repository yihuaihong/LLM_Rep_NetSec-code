"""Aggregate metrics from multiple models × datasets into comparison tables and figures."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
METRICS = BASE / "results" / "metrics"
FIGS = BASE / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

MODELS = [
    "Meta-Llama-3-8B-Instruct",
    "Llama-3.1-8B-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "Qwen3-8B",
    "gemma-2-9b-it",
]
DATASETS = ["unsw_nb15", "cicids2017"]


def load_metrics(model, dataset):
    p = METRICS / f"{model}_{dataset}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def main():
    table = []
    for ds in DATASETS:
        for m in MODELS:
            data = load_metrics(m, ds)
            if data is None:
                print(f"missing: {m} / {ds}")
                continue
            sweep = data.get("layer_sweep", {})
            best = data.get("best_layer")
            row = {
                "model": m,
                "dataset": ds,
                "best_layer": best,
                "known_acc": data.get("known_accuracy"),
                "known_auroc": data.get("known_auroc"),
                "holdout_acc": data.get("holdout_accuracy"),
                "holdout_auroc": data.get("holdout_auroc"),
                "gap_auroc": data.get("generalization_gap", {}).get("auroc_gap"),
                "n_layers_sampled": len(sweep),
            }
            table.append(row)

    print(f"\n{'='*120}")
    print(f"{'model':30s} {'dataset':12s} {'best L':>7s} {'known acc':>10s} {'holdout acc':>12s} {'holdout AUROC':>14s} {'gap AUROC':>10s}")
    print("="*120)
    for r in table:
        print(f"{r['model']:30s} {r['dataset']:12s} {r['best_layer']!s:>7s} "
              f"{r['known_acc']:>10.4f} {r['holdout_acc']:>12.4f} "
              f"{r['holdout_auroc']:>14.4f} {r['gap_auroc']:>+10.4f}")

    # Save aggregated json
    with open(METRICS / "multimodel_comparison.json", "w") as f:
        json.dump(table, f, indent=2)

    # Bar plot: holdout AUROC, grouped by dataset
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, ds in zip(axes, DATASETS):
        rows = [r for r in table if r["dataset"] == ds]
        models = [r["model"] for r in rows]
        known_aurocs = [r["known_auroc"] for r in rows]
        holdout_aurocs = [r["holdout_auroc"] for r in rows]
        x = np.arange(len(models))
        w = 0.4
        ax.bar(x - w/2, known_aurocs, w, label="Known AUROC", color="steelblue", alpha=0.85)
        ax.bar(x + w/2, holdout_aurocs, w, label="Holdout AUROC", color="darkorange", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("-Instruct", "").replace("Meta-", "") for m in models],
                           rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("AUROC")
        ax.set_ylim(0.4, 1.02)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4)
        ax.set_title(f"{ds.upper()}")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Multi-model probe AUROC: known (in-distribution) vs holdout (zero-day)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = FIGS / "multimodel_holdout_auroc_comparison.png"
    plt.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\nSaved {out}")

    # Layer-sweep panel
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, ds in zip(axes, DATASETS):
        for r in table:
            if r["dataset"] != ds:
                continue
            data = load_metrics(r["model"], ds)
            sweep = data["layer_sweep"]
            xs = sorted(int(k) for k in sweep.keys())
            ys = [sweep[str(x)]["cv_auroc"] for x in xs]
            num_layers = max(xs) + 1
            xs_norm = [x / max(1, num_layers - 1) for x in xs]
            ax.plot(xs_norm, ys, "o-", label=r["model"].replace("-Instruct", "").replace("Meta-", ""), alpha=0.85)
        ax.set_xlabel("Relative layer depth (0 = embedding, 1 = final)")
        ax.set_ylabel("Train CV AUROC")
        ax.set_title(f"{ds.upper()} — probe quality across layers")
        ax.set_ylim(0.45, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    out2 = FIGS / "multimodel_layer_sweep.png"
    plt.savefig(out2, dpi=130)
    plt.close(fig)
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
