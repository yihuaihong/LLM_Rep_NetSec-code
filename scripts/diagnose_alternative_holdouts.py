"""Test alternative CICIDS holdout configurations.

Pool train + test_known reps (which contain DoS Hulk / DDoS / PortScan / DoS variants /
brute-force / web attacks), then re-split with different holdout types and re-evaluate.
This tells us if poor CICIDS holdout generalization is intrinsic to the dataset or
specific to the {Bot, Heartbleed, Infiltration} choice.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from utils.probe_utils import (
    train_logistic_probe, evaluate_probe,
    extract_separation_direction, project_onto_direction,
)

BASE = Path(__file__).resolve().parent.parent
MODEL = "Meta-Llama-3-8B-Instruct"
LAYER = 31
RNG = np.random.RandomState(42)


def pool_known_reps(dataset_name: str):
    reps_dir = BASE / "results" / "representations" / dataset_name / MODEL
    train = torch.load(reps_dir / "train.pt", weights_only=False)
    known = torch.load(reps_dir / "test_known.pt", weights_only=False)
    X = torch.cat([train["hidden_states"][LAYER], known["hidden_states"][LAYER]], dim=0)
    y = np.concatenate([train["labels"].numpy(), known["labels"].numpy()])
    types = np.concatenate([np.array(train["attack_types"]), np.array(known["attack_types"])])
    return X, y, types


def evaluate_holdout_split(X, y, types, holdout_types, label, dataset_name):
    holdout_mask = np.isin(types, holdout_types)
    normal_mask = (y == 0)
    known_attack_mask = (y == 1) & (~holdout_mask)

    if known_attack_mask.sum() < 50 or holdout_mask.sum() < 5:
        return None

    # Split normal 80/20
    normal_idx = np.where(normal_mask)[0]
    RNG.shuffle(normal_idx)
    n_normal_train = int(0.8 * len(normal_idx))
    normal_train_idx = normal_idx[:n_normal_train]
    normal_test_idx = normal_idx[n_normal_train:]

    # Use all known attacks 80/20
    known_idx = np.where(known_attack_mask)[0]
    RNG.shuffle(known_idx)
    n_known_train = int(0.8 * len(known_idx))
    known_train_idx = known_idx[:n_known_train]
    known_test_idx = known_idx[n_known_train:]

    holdout_idx = np.where(holdout_mask)[0]

    X_train = X[np.concatenate([normal_train_idx, known_train_idx])].numpy()
    y_train = np.concatenate([np.zeros(len(normal_train_idx)), np.ones(len(known_train_idx))]).astype(int)

    X_known_test = X[np.concatenate([normal_test_idx, known_test_idx])].numpy()
    y_known_test = np.concatenate([np.zeros(len(normal_test_idx)), np.ones(len(known_test_idx))]).astype(int)

    X_holdout = X[np.concatenate([normal_test_idx, holdout_idx])].numpy()
    y_holdout = np.concatenate([np.zeros(len(normal_test_idx)), np.ones(len(holdout_idx))]).astype(int)

    probe = train_logistic_probe(X_train, y_train)
    clf = probe["model"]
    known_eval = evaluate_probe(clf, X_known_test, y_known_test)
    holdout_eval = evaluate_probe(clf, X_holdout, y_holdout)

    # Direction-only acc
    direction = extract_separation_direction(torch.tensor(X_train), y_train)
    proj_n = project_onto_direction(torch.tensor(X[normal_train_idx].numpy()), direction)
    proj_a = project_onto_direction(torch.tensor(X[known_train_idx].numpy()), direction)
    threshold = (proj_n.mean() + proj_a.mean()) / 2
    proj_holdout_attack = project_onto_direction(torch.tensor(X[holdout_idx].numpy()), direction)
    proj_holdout_normal = project_onto_direction(torch.tensor(X[normal_test_idx].numpy()), direction)

    dir_acc_attacks = (proj_holdout_attack > threshold).mean()
    holdout_proj_mean = float(np.nanmean(proj_holdout_attack))

    return {
        "dataset": dataset_name,
        "holdout_types": holdout_types,
        "label": label,
        "n_train": int(len(y_train)),
        "n_known_test": int(len(y_known_test)),
        "n_holdout_attack": int(len(holdout_idx)),
        "known_acc": float(known_eval["accuracy"]),
        "known_auroc": float(known_eval.get("auroc", float("nan"))),
        "holdout_acc": float(holdout_eval["accuracy"]),
        "holdout_auroc": float(holdout_eval.get("auroc", float("nan"))),
        "holdout_dir_attack_recall": float(dir_acc_attacks),
        "holdout_proj_mean_attack": holdout_proj_mean,
    }


def main():
    print(f"=== Alternative holdout configurations @ layer {LAYER} ===\n")

    # CICIDS configurations
    X_c, y_c, t_c = pool_known_reps("cicids2017")
    print(f"CICIDS pooled: {len(y_c)} samples ({(y_c==1).sum()} attacks, {(y_c==0).sum()} normal)")
    print(f"CICIDS types: {dict(zip(*np.unique(t_c, return_counts=True)))}\n")

    cicids_configs = [
        ("hold out low-rate DoS variants",
         ["DoS GoldenEye", "DoS Slowhttptest", "DoS slowloris"]),
        ("hold out Patator (brute-force)",
         ["FTP-Patator", "SSH-Patator"]),
        ("hold out Web Attacks",
         ["Web Attack – Brute Force", "Web Attack – XSS"]),
        ("hold out PortScan",
         ["PortScan"]),
        ("hold out DDoS",
         ["DDoS"]),
        ("hold out DoS Hulk",
         ["DoS Hulk"]),
    ]

    results = []
    for label, ho_types in cicids_configs:
        res = evaluate_holdout_split(X_c, y_c, t_c, ho_types, label, "cicids2017")
        if res is not None:
            results.append(res)
            print(f"[CICIDS] {label}: known_acc={res['known_acc']:.3f} "
                  f"holdout_acc={res['holdout_acc']:.3f} "
                  f"holdout_auroc={res['holdout_auroc']:.3f} "
                  f"dir_attack_recall={res['holdout_dir_attack_recall']:.3f} "
                  f"proj_mean={res['holdout_proj_mean_attack']:+.2f} "
                  f"(n_holdout={res['n_holdout_attack']})")

    # UNSW configurations for comparison
    X_u, y_u, t_u = pool_known_reps("unsw_nb15")
    print(f"\nUNSW pooled: {len(y_u)} samples ({(y_u==1).sum()} attacks)")
    print(f"UNSW types: {dict(zip(*np.unique(t_u, return_counts=True)))}\n")

    unsw_configs = [
        ("hold out Reconnaissance", ["Reconnaissance"]),
        ("hold out Fuzzers", ["Fuzzers"]),
        ("hold out Generic", ["Generic"]),
        ("hold out Exploits", ["Exploits"]),
        ("hold out Dos", ["Dos"]),
        ("hold out Analysis", ["Analysis"]),
    ]
    for label, ho_types in unsw_configs:
        res = evaluate_holdout_split(X_u, y_u, t_u, ho_types, label, "unsw_nb15")
        if res is not None:
            results.append(res)
            print(f"[UNSW]   {label}: known_acc={res['known_acc']:.3f} "
                  f"holdout_acc={res['holdout_acc']:.3f} "
                  f"holdout_auroc={res['holdout_auroc']:.3f} "
                  f"dir_attack_recall={res['holdout_dir_attack_recall']:.3f} "
                  f"proj_mean={res['holdout_proj_mean_attack']:+.2f} "
                  f"(n_holdout={res['n_holdout_attack']})")

    # Save and plot
    out_json = BASE / "results" / "metrics" / "alternative_holdouts.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_json}")

    # Summary bar plot: holdout AUROC for each configuration
    fig, ax = plt.subplots(figsize=(11, 6))
    cic = [r for r in results if r["dataset"] == "cicids2017"]
    uns = [r for r in results if r["dataset"] == "unsw_nb15"]

    labels_c = [r["label"].replace("hold out ", "") + f" (n={r['n_holdout_attack']})" for r in cic]
    aurocs_c = [r["holdout_auroc"] for r in cic]
    labels_u = [r["label"].replace("hold out ", "") + f" (n={r['n_holdout_attack']})" for r in uns]
    aurocs_u = [r["holdout_auroc"] for r in uns]

    y1 = np.arange(len(labels_c))
    y2 = np.arange(len(labels_u)) + len(labels_c) + 1
    ax.barh(y1, aurocs_c, color="#4a90e2", alpha=0.85, label="CICIDS2017")
    ax.barh(y2, aurocs_u, color="#e2884a", alpha=0.85, label="UNSW-NB15")
    ax.set_yticks(np.concatenate([y1, y2]))
    ax.set_yticklabels(labels_c + labels_u, fontsize=9)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="chance")
    ax.set_xlabel("Holdout AUROC (zero-day generalization)")
    ax.set_xlim(0, 1.02)
    ax.set_title(f"Alternative holdout choices @ layer {LAYER} — Llama-3-8B-Instruct")
    ax.legend(loc="lower right")
    plt.tight_layout()
    out_fig = BASE / "results" / "figures" / "alternative_holdouts_auroc.png"
    plt.savefig(out_fig, dpi=130)
    plt.close(fig)
    print(f"Saved {out_fig}")


if __name__ == "__main__":
    main()
