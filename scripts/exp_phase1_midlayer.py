"""Re-run Exp 2 (per-type) + Exp D (cross-dataset) at MIDDLE layer per model.

Hypothesis: middle layers carry more abstract / generalizable concepts than
last layer. Best-layer selection is greedy on CV AUROC (in-distribution),
which biases toward last layers.

For each model use:
  middle_layer = layers[len(layers)//2]   (if 9 layers cached, that's layers[4])

This gives a fair "mid-layer" comparison where every model uses the same depth%.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

from utils.probe_utils import (
    extract_separation_direction,
    project_onto_direction,
)

BASE = Path(__file__).resolve().parent.parent
REPS = BASE / "results" / "representations"
OUT = BASE / "results" / "metrics"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["Llama-3.1-8B-Instruct", "Mistral-7B-Instruct-v0.3", "Qwen3-8B", "gemma-2-9b-it"]
DATASETS = ["cicids2017", "unsw_nb15"]


def to_f32(t):
    return t.float() if t.dtype == torch.bfloat16 else t


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(
        a.float().unsqueeze(0), b.float().unsqueeze(0)
    ).item())


def load(model, ds, split):
    p = REPS / ds / model / f"{split}.pt"
    if not p.exists():
        return None
    return torch.load(p, weights_only=False)


def best_layer(model, ds):
    f = OUT / f"{model}_{ds}.json"
    if not f.exists():
        return None
    return int(json.load(open(f))["best_layer"])


def mid_layer(model, ds):
    """Middle of cached layers (50% depth)."""
    train = load(model, ds, "train")
    if train is None:
        return None
    layers = sorted(train["hidden_states"].keys())
    return layers[len(layers) // 2]


def run_per_type_at_layer(model, ds, layer):
    train = load(model, ds, "train")
    holdout = load(model, ds, "test_holdout")
    known = load(model, ds, "test_known")
    if train is None or holdout is None:
        return None

    y_tr = train["labels"].numpy()
    y_ho = holdout["labels"].numpy()
    types_ho = np.array(holdout["attack_types"])

    H_tr = to_f32(train["hidden_states"][layer])
    H_ho = to_f32(holdout["hidden_states"][layer])

    d = extract_separation_direction(H_tr, y_tr)
    proj_n = project_onto_direction(H_tr[y_tr == 0], d)
    proj_a = project_onto_direction(H_tr[y_tr == 1], d)
    threshold = (proj_n.mean() + proj_a.mean()) / 2
    proj_ho = project_onto_direction(H_ho, d)

    # probe
    X_tr = H_tr.numpy()
    X_ho = H_ho.numpy()
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1))
    pipe.fit(X_tr, y_tr)
    probe_acc_ho = float((pipe.predict(X_ho) == y_ho).mean())
    probe_auroc_ho = float(roc_auc_score(y_ho, pipe.predict_proba(X_ho)[:, 1]))

    if known is not None:
        y_kn = known["labels"].numpy()
        X_kn = to_f32(known["hidden_states"][layer]).numpy()
        probe_acc_kn = float((pipe.predict(X_kn) == y_kn).mean())
        probe_auroc_kn = float(roc_auc_score(y_kn, pipe.predict_proba(X_kn)[:, 1]))
    else:
        probe_acc_kn = probe_auroc_kn = None

    per_type = {}
    for t in np.unique(types_ho):
        mask = types_ho == t
        if mask.sum() == 0:
            continue
        per_type[t] = {
            "n": int(mask.sum()),
            "true_label": int(y_ho[mask][0] == 1),
            "probe_acc": float((pipe.predict(X_ho[mask]) == y_ho[mask]).mean()),
            "direction_acc": float(((proj_ho[mask] > threshold).astype(int) == y_ho[mask]).mean()),
            "projection_mean": float(np.nanmean(proj_ho[mask])),
        }

    return {
        "layer": layer,
        "n_layers_cached": len(train["hidden_states"]),
        "probe_known_acc": probe_acc_kn,
        "probe_known_auroc": probe_auroc_kn,
        "probe_holdout_acc": probe_acc_ho,
        "probe_holdout_auroc": probe_auroc_ho,
        "direction": d.float().tolist(),
        "threshold": float(threshold),
        "per_type": per_type,
    }


def main():
    out = {}
    for m in MODELS:
        out[m] = {}
        for ds in DATASETS:
            bl = best_layer(m, ds)
            ml = mid_layer(m, ds)
            if bl is None or ml is None:
                print(f"  [{m}/{ds}] no metrics, skip"); continue

            train = load(m, ds, "train")
            layers = sorted(train["hidden_states"].keys())

            print(f"\n=== {m} / {ds} ===")
            print(f"  cached layers: {layers}")
            print(f"  best_layer (from CV) = {bl}, mid_layer (50%) = {ml}")

            results_at = {}
            for tag, L in [("best", bl), ("mid", ml)]:
                r = run_per_type_at_layer(m, ds, L)
                if r is None:
                    continue
                results_at[tag] = r
                pt = r["per_type"]
                print(f"\n  [{tag} layer = {L}]  probe known={r['probe_known_acc']:.3f}  "
                      f"holdout={r['probe_holdout_acc']:.3f}  gap={r['probe_known_acc'] - r['probe_holdout_acc']:+.3f}")
                for t, v in pt.items():
                    marker = "❌" if v["true_label"] == 1 and v["projection_mean"] < r["threshold"] else "✓"
                    print(f"    {marker} {t:30s} n={v['n']:>4}  probe={v['probe_acc']:.3f}  "
                          f"dir={v['direction_acc']:.3f}  proj={v['projection_mean']:+.2f}")

            # Cosine between best-layer direction and mid-layer direction
            if "best" in results_at and "mid" in results_at:
                d_best = torch.tensor(results_at["best"]["direction"])
                d_mid = torch.tensor(results_at["mid"]["direction"])
                # Only meaningful if same hidden_dim (same layer dim — yes, same model)
                if d_best.shape == d_mid.shape:
                    c = cos(d_best, d_mid)
                    print(f"\n  cos(best_dir, mid_dir) = {c:+.4f}")
                    results_at["cos_best_vs_mid"] = c
            out[m][ds] = results_at

    json.dump(out, open(OUT / "phase1_midlayer_per_type.json", "w"), indent=2)
    print("\nsaved -> phase1_midlayer_per_type.json")

    # Cross-dataset transfer at mid layer
    print("\n========== Cross-dataset transfer at MID layer ==========")
    transfer = {}
    for m in MODELS:
        bl_c = best_layer(m, "cicids2017")
        bl_u = best_layer(m, "unsw_nb15")
        ml_c = mid_layer(m, "cicids2017")
        ml_u = mid_layer(m, "unsw_nb15")
        if any(x is None for x in [bl_c, bl_u, ml_c, ml_u]):
            print(f"  [{m}] missing data, skip"); continue
        # align layer indices
        cic_train = load(m, "cicids2017", "train")
        uns_train = load(m, "unsw_nb15", "train")

        rec = {}
        for tag, (L_c, L_u) in [("best", (bl_c, bl_u)), ("mid", (ml_c, ml_u))]:
            shared = min(L_c, L_u)
            d_c = extract_separation_direction(to_f32(cic_train["hidden_states"][shared]), cic_train["labels"].numpy())
            d_u = extract_separation_direction(to_f32(uns_train["hidden_states"][shared]), uns_train["labels"].numpy())
            cs = cos(d_c, d_u)
            # Project the OTHER dataset's training reps through this direction to compute AUROC
            proj_uns_with_cic = project_onto_direction(to_f32(uns_train["hidden_states"][shared]), d_c)
            proj_cic_with_uns = project_onto_direction(to_f32(cic_train["hidden_states"][shared]), d_u)
            auroc_c2u = float(roc_auc_score(uns_train["labels"].numpy(), proj_uns_with_cic))
            auroc_u2c = float(roc_auc_score(cic_train["labels"].numpy(), proj_cic_with_uns))
            rec[tag] = {
                "layer_used": shared,
                "cos_cic_uns": cs,
                "cic_to_uns_auroc": auroc_c2u,
                "uns_to_cic_auroc": auroc_u2c,
            }
        transfer[m] = rec
        print(f"\n  [{m}]")
        for tag, r in rec.items():
            print(f"    {tag:5s} @ L{r['layer_used']:>2}:  cos={r['cos_cic_uns']:+.4f}  "
                  f"cic→uns AUROC={r['cic_to_uns_auroc']:.3f}  uns→cic AUROC={r['uns_to_cic_auroc']:.3f}")

    json.dump(transfer, open(OUT / "phase1_midlayer_cross_dataset.json", "w"), indent=2)
    print("\nsaved -> phase1_midlayer_cross_dataset.json")


if __name__ == "__main__":
    main()
