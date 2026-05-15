"""Diagnose CICIDS holdout failure: per-type breakdown of probe and direction."""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict

from utils.probe_utils import (
    train_logistic_probe, evaluate_probe,
    extract_separation_direction, project_onto_direction,
)

BASE = Path(__file__).resolve().parent.parent
FIGS = BASE / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


def per_type_breakdown(model: str, dataset_name: str, layers: list[int], holdout_label: str):
    reps_dir = BASE / "results" / "representations" / dataset_name / model
    train = torch.load(reps_dir / "train.pt", weights_only=False)
    holdout = torch.load(reps_dir / "test_holdout.pt", weights_only=False)

    y_train = train["labels"].numpy()
    y_holdout = holdout["labels"].numpy()
    types_holdout = np.array(holdout["attack_types"])

    rows = []
    layer_to_per_type = {}
    for layer in layers:
        X_tr = train["hidden_states"][layer]
        X_ho = holdout["hidden_states"][layer]

        probe = train_logistic_probe(X_tr.numpy(), y_train)
        clf = probe["model"]
        prob_ho = clf.predict_proba(X_ho.numpy())[:, 1]
        pred_ho = (prob_ho > 0.5).astype(int)

        direction = extract_separation_direction(X_tr, y_train)
        proj_ho = project_onto_direction(X_ho, direction)
        proj_normal_train = project_onto_direction(X_tr[y_train == 0], direction)
        proj_attack_train = project_onto_direction(X_tr[y_train == 1], direction)
        threshold = (proj_normal_train.mean() + proj_attack_train.mean()) / 2

        per_type = OrderedDict()
        for atype in np.unique(types_holdout):
            mask = types_holdout == atype
            if mask.sum() == 0:
                continue
            true_label = int((y_holdout[mask][0] == 1))
            n = int(mask.sum())
            probe_acc = float((pred_ho[mask] == y_holdout[mask]).mean())
            proj_mean = float(proj_ho[mask].mean())
            proj_std = float(proj_ho[mask].std())
            dir_pred = (proj_ho[mask] > threshold).astype(int)
            dir_acc = float((dir_pred == y_holdout[mask]).mean())
            per_type[atype] = {
                "n": n,
                "true_label": true_label,
                "probe_acc": probe_acc,
                "direction_acc": dir_acc,
                "projection_mean": proj_mean,
                "projection_std": proj_std,
            }
        layer_to_per_type[layer] = per_type
        for atype, d in per_type.items():
            rows.append({
                "dataset": dataset_name, "layer": layer, "attack_type": atype,
                **d,
            })

    return layer_to_per_type, rows


def make_per_type_proj_figure(model: str, dataset_name: str, layer: int, layer_to_per_type, save_path):
    per_type = layer_to_per_type[layer]
    types = list(per_type.keys())
    means = [per_type[t]["projection_mean"] for t in types]
    stds = [per_type[t]["projection_std"] for t in types]
    accs = [per_type[t]["direction_acc"] for t in types]
    ns = [per_type[t]["n"] for t in types]
    labels = [per_type[t]["true_label"] for t in types]

    colors = ["steelblue" if l == 0 else "darkorange" for l in labels]
    order = np.argsort(means)
    types_o = [types[i] for i in order]
    means_o = [means[i] for i in order]
    stds_o = [stds[i] for i in order]
    accs_o = [accs[i] for i in order]
    ns_o = [ns[i] for i in order]
    colors_o = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(11, max(4, 0.45 * len(types))))
    ypos = np.arange(len(types_o))
    ax.barh(ypos, means_o, xerr=stds_o, color=colors_o, alpha=0.75, capsize=4)
    for i, (m, a, n) in enumerate(zip(means_o, accs_o, ns_o)):
        ax.text(m, i, f"  acc={a:.2f}, n={n}", va="center", fontsize=8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(types_o, fontsize=9)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel(f"Mean projection on attack direction (layer {layer}, train fit)")
    ax.set_title(f"{dataset_name.upper()} — per attack-type projection on holdout ({model})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Meta-Llama-3-8B-Instruct")
    ap.add_argument("--datasets", nargs="+", default=["cicids2017", "unsw_nb15"])
    args = ap.parse_args()

    summary = {}
    for ds in args.datasets:
        reps_dir = BASE / "results" / "representations" / ds / args.model
        layers_json = reps_dir / "layers.json"
        if layers_json.exists():
            layers = json.loads(layers_json.read_text())["layers"]
        else:
            # fall back: peek at a saved .pt
            sample = torch.load(reps_dir / "train.pt", weights_only=False)
            layers = sorted(sample["hidden_states"].keys())
        best = layers[-1]

        l2pt, rows = per_type_breakdown(args.model, ds, layers, holdout_label="holdout")
        summary[ds] = l2pt
        out = FIGS / f"per_type_projection_{args.model}_{ds}_layer{best}.png"
        make_per_type_proj_figure(args.model, ds, best, l2pt, out)
        print(f"saved {out}")

        print(f"\n=== [{args.model}] {ds} per-type @ layer {best} ===")
        per_type = l2pt[best]
        for t, d in per_type.items():
            print(f"  {t:35s} label={d['true_label']} n={d['n']:4d} "
                  f"probe_acc={d['probe_acc']:.3f}  dir_acc={d['direction_acc']:.3f}  "
                  f"proj_mean={d['projection_mean']:+.3f}±{d['projection_std']:.3f}")

    out_json = BASE / "results" / "metrics" / f"per_type_holdout_{args.model}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {
        ds: {str(l): pt for l, pt in d.items()} for ds, d in summary.items()
    }
    with open(out_json, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"\nSaved {out_json}")


if __name__ == "__main__":
    main()
