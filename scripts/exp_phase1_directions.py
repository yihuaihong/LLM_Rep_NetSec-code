"""Phase-1 experiments using cached reps only:
  Exp 1: cross-model direction cosine within each dataset
  Exp 2: per-type holdout breakdown for all 4 models
  Exp 7: layer dynamics from layer_sweep (no compute, just plot)
  Exp A: minority-only training (drop DoS/PortScan/DDoS, retrain probe + direction)
  Exp D: cross-dataset direction transfer

All output goes to results/metrics/phase1_*.json
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.linear_model import LogisticRegression

from utils.probe_utils import (
    extract_separation_direction,
    project_onto_direction,
    train_logistic_probe,
    evaluate_probe,
)

BASE = Path(__file__).resolve().parent.parent
REPS = BASE / "results" / "representations"
OUT = BASE / "results" / "metrics"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["Llama-3.1-8B-Instruct", "Mistral-7B-Instruct-v0.3", "Qwen3-8B", "gemma-2-9b-it"]
DATASETS = ["cicids2017", "unsw_nb15"]


def to_f32(t):
    if t.dtype == torch.bfloat16:
        return t.float()
    return t


def cos_sim(a, b):
    a = a.float() if hasattr(a, "float") else torch.tensor(a, dtype=torch.float32)
    b = b.float() if hasattr(b, "float") else torch.tensor(b, dtype=torch.float32)
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def load_split(model, ds, split):
    p = REPS / ds / model / f"{split}.pt"
    if not p.exists():
        return None
    return torch.load(p, weights_only=False)


def best_layer(model, ds):
    f = OUT / f"{model}_{ds}.json"
    if not f.exists():
        return None
    d = json.load(open(f))
    return int(d["best_layer"])


def get_direction(model, ds, layer):
    """Train mean-diff direction at given layer."""
    train = load_split(model, ds, "train")
    if train is None or layer not in train["hidden_states"]:
        return None
    H = to_f32(train["hidden_states"][layer])
    y = train["labels"].numpy()
    return extract_separation_direction(H, y)


# ──────────────────────────────────────────────
# Exp 1: cross-model direction cosine
# Cannot compare directions across models directly (different hidden_dim),
# so instead compare downstream behavior: does each model's direction agree
# on RANK-ORDER of holdout samples? Use Spearman rank correlation of projections.
# ──────────────────────────────────────────────
from scipy.stats import spearmanr


def exp1_cross_model_agreement():
    print("\n========== Exp 1: cross-model agreement on holdout projection ==========")
    out = {}
    for ds in DATASETS:
        # for each model, project holdout onto its own train-direction at that model's best layer
        proj_per_model = {}
        labels = None
        types = None
        for m in MODELS:
            bl = best_layer(m, ds)
            d = get_direction(m, ds, bl)
            if d is None:
                continue
            ho = load_split(m, ds, "test_holdout")
            if ho is None:
                continue
            H = to_f32(ho["hidden_states"][bl])
            proj = project_onto_direction(H, d)
            proj_per_model[m] = proj
            if labels is None:
                labels = ho["labels"].numpy()
                types = np.array(ho["attack_types"])
        models_with_data = list(proj_per_model.keys())
        n = len(models_with_data)
        rho_mat = np.zeros((n, n))
        for i, mi in enumerate(models_with_data):
            for j, mj in enumerate(models_with_data):
                rho, _ = spearmanr(proj_per_model[mi], proj_per_model[mj])
                rho_mat[i, j] = rho
        out[ds] = {
            "models": models_with_data,
            "spearman": rho_mat.tolist(),
        }
        print(f"\n  [{ds}] Spearman rho of holdout projections:")
        print(f"           " + "  ".join(f"{m[:12]:>12}" for m in models_with_data))
        for i, m in enumerate(models_with_data):
            print(f"   {m[:12]:>12} " + "  ".join(f"{rho_mat[i,j]:>12.3f}" for j in range(n)))
    json.dump(out, open(OUT / "phase1_exp1_cross_model_agreement.json", "w"), indent=2)
    print(f"\n  saved -> phase1_exp1_cross_model_agreement.json")


# ──────────────────────────────────────────────
# Exp 2: per-type holdout breakdown for 4 models
# ──────────────────────────────────────────────
def exp2_per_type_breakdown():
    print("\n========== Exp 2: per-type holdout breakdown ==========")
    out = {}
    out_path = OUT / "phase1_exp2_per_type_4models.json"
    for m in MODELS:
        out[m] = {}
        for ds in DATASETS:
            bl = best_layer(m, ds)
            if bl is None:
                print(f"  [{m}/{ds}] no metrics yet, skip")
                continue
            train = load_split(m, ds, "train")
            holdout = load_split(m, ds, "test_holdout")
            if train is None or holdout is None:
                continue
            y_train = train["labels"].numpy()
            y_ho = holdout["labels"].numpy()
            types_ho = np.array(holdout["attack_types"])

            H_tr = to_f32(train["hidden_states"][bl])
            H_ho = to_f32(holdout["hidden_states"][bl])
            d = extract_separation_direction(H_tr, y_train)
            proj_n = project_onto_direction(H_tr[y_train == 0], d)
            proj_a = project_onto_direction(H_tr[y_train == 1], d)
            threshold = (proj_n.mean() + proj_a.mean()) / 2
            proj_ho = project_onto_direction(H_ho, d)

            X_tr = H_tr.numpy()
            X_ho = H_ho.numpy()
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(X_tr, y_train)

            per_type = {}
            for t in np.unique(types_ho):
                mask = types_ho == t
                if mask.sum() == 0:
                    continue
                tl = int(y_ho[mask][0] == 1)
                per_type[t] = {
                    "n": int(mask.sum()),
                    "true_label": tl,
                    "probe_acc": float((clf.predict(X_ho[mask]) == y_ho[mask]).mean()),
                    "direction_acc": float(((proj_ho[mask] > threshold).astype(int) == y_ho[mask]).mean()),
                    "projection_mean": float(np.nanmean(proj_ho[mask])),
                    "projection_std": float(np.nanstd(proj_ho[mask])),
                    "threshold": float(threshold),
                }
            out[m][ds] = {"layer": bl, "per_type": per_type}
            print(f"\n  [{m}/{ds}] @ layer {bl}:")
            for t, v in per_type.items():
                marker = "❌" if v["true_label"] == 1 and v["projection_mean"] < threshold else "✓"
                print(f"    {marker} {t:30s} n={v['n']:>4} probe={v['probe_acc']:.3f} "
                      f"dir={v['direction_acc']:.3f} proj={v['projection_mean']:+.2f}")
            json.dump(out, open(out_path, "w"), indent=2)  # save after every model/dataset
    print(f"\n  saved -> phase1_exp2_per_type_4models.json")


# ──────────────────────────────────────────────
# Exp 7: layer dynamics from layer_sweep
# ──────────────────────────────────────────────
def exp7_layer_dynamics():
    print("\n========== Exp 7: layer dynamics ==========")
    out = {}
    for m in MODELS:
        out[m] = {}
        for ds in DATASETS:
            f = OUT / f"{m}_{ds}.json"
            if not f.exists():
                continue
            d = json.load(open(f))
            sweep = d["layer_sweep"]
            layers = sorted(int(l) for l in sweep.keys())
            cv_auroc = [sweep[str(l)]["cv_auroc"] for l in layers]
            cv_acc = [sweep[str(l)]["cv_accuracy"] for l in layers]
            out[m][ds] = {
                "layers": layers,
                "cv_auroc": cv_auroc,
                "cv_accuracy": cv_acc,
                "best_layer": d["best_layer"],
                "known_acc": d["known_accuracy"],
                "holdout_acc": d["holdout_accuracy"],
            }
            print(f"  {m:30s} {ds:10s}  layers={layers}  best={d['best_layer']:>2}  "
                  f"holdout_auroc(known→ho)={d.get('known_auroc',0):.3f}→{d.get('holdout_auroc',0):.3f}")
    json.dump(out, open(OUT / "phase1_exp7_layer_dynamics.json", "w"), indent=2)
    print(f"\n  saved -> phase1_exp7_layer_dynamics.json")


# ──────────────────────────────────────────────
# Exp A: minority-only train
# ──────────────────────────────────────────────
MINORITY_KNOWN = {
    "cicids2017": {"FTP-Patator", "SSH-Patator", "Web Attack – Brute Force", "Web Attack – XSS",
                   "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest"},
    # NOTE: For UNSW we don't really have a clear "majority/minority" split since types are more balanced;
    # we'll skip UNSW for Exp A.
}
MAJORITY_DROP = {
    "cicids2017": {"DoS Hulk", "PortScan", "DDoS"},
}


def exp_a_minority_train():
    print("\n========== Exp A: minority-only training (CICIDS2017) ==========")
    out = {}
    ds = "cicids2017"
    for m in MODELS:
        bl = best_layer(m, ds)
        if bl is None:
            print(f"  [{m}] no metrics yet, skip"); continue
        train = load_split(m, ds, "train")
        holdout = load_split(m, ds, "test_holdout")
        known = load_split(m, ds, "test_known")
        if train is None:
            continue
        H_tr = to_f32(train["hidden_states"][bl])
        y_tr = train["labels"].numpy()
        types_tr = np.array(train["attack_types"])

        # Full direction (baseline)
        d_full = extract_separation_direction(H_tr, y_tr)

        # Minority training mask: keep all normal + minority attacks
        keep_attack = np.array([t in MINORITY_KNOWN[ds] for t in types_tr])
        keep = (types_tr == "Normal") | keep_attack
        if keep_attack.sum() < 30:
            print(f"  [{m}] not enough minority attacks ({keep_attack.sum()}); skip")
            continue
        H_min = H_tr[keep]
        y_min = y_tr[keep]
        types_min = types_tr[keep]
        d_min = extract_separation_direction(H_min, y_min)

        cs = cos_sim(d_full, d_min)

        # Project holdout onto each direction; report direction accuracy
        H_ho = to_f32(holdout["hidden_states"][bl])
        y_ho = holdout["labels"].numpy()
        types_ho = np.array(holdout["attack_types"])

        for tag, d in [("full", d_full), ("minority", d_min)]:
            proj_n = project_onto_direction(H_tr[y_tr == 0], d)
            proj_a = project_onto_direction(H_tr[y_tr == 1], d)
            thr = (proj_n.mean() + proj_a.mean()) / 2
            proj_ho = project_onto_direction(H_ho, d)
            preds = (proj_ho > thr).astype(int)
            acc = float((preds == y_ho).mean())
            per = {}
            for t in np.unique(types_ho):
                mk = types_ho == t
                if mk.sum() == 0:
                    continue
                per[t] = {
                    "n": int(mk.sum()),
                    "proj_mean": float(np.nanmean(proj_ho[mk])),
                    "dir_acc": float(((proj_ho[mk] > thr).astype(int) == y_ho[mk]).mean()),
                }
            out.setdefault(m, {})[tag] = {
                "n_train_attacks": int((y_min if tag == "minority" else y_tr).sum()),
                "holdout_dir_acc": acc,
                "per_type": per,
                "threshold": float(thr),
            }
        out[m]["cos_sim_full_vs_minority"] = cs
        print(f"\n  [{m}] cos(full, minority) = {cs:.3f}")
        print(f"    full:     n_atk={int(y_tr.sum())}  ho_dir_acc={out[m]['full']['holdout_dir_acc']:.3f}")
        print(f"    minority: n_atk={out[m]['minority']['n_train_attacks']}  ho_dir_acc={out[m]['minority']['holdout_dir_acc']:.3f}")
        print(f"    per-type holdout dir_acc (full / minority):")
        full_pt = out[m]["full"]["per_type"]
        min_pt = out[m]["minority"]["per_type"]
        for t in full_pt:
            f, mn = full_pt[t], min_pt.get(t, {})
            print(f"      {t:25s} n={f['n']:>4}  full={f['dir_acc']:.3f} ({f['proj_mean']:+.2f})  "
                  f"min={mn.get('dir_acc',0):.3f} ({mn.get('proj_mean',0):+.2f})")
    json.dump(out, open(OUT / "phase1_expA_minority_train.json", "w"), indent=2)
    print(f"\n  saved -> phase1_expA_minority_train.json")


# ──────────────────────────────────────────────
# Exp D: cross-dataset direction transfer
# Only meaningful within the SAME model (same hidden_dim).
# ──────────────────────────────────────────────
def exp_d_cross_dataset():
    print("\n========== Exp D: cross-dataset direction transfer ==========")
    out = {}
    for m in MODELS:
        bl_c = best_layer(m, "cicids2017")
        bl_u = best_layer(m, "unsw_nb15")
        if bl_c is None or bl_u is None:
            print(f"  [{m}] missing metrics for one dataset, skip"); continue
        # We need both directions at COMPARABLE layers. Use best layer for each.
        d_c = get_direction(m, "cicids2017", bl_c)
        d_u = get_direction(m, "unsw_nb15", bl_u)
        if d_c is None or d_u is None:
            continue

        # Cross-dataset cos: compare directions only when they live at the same layer
        # (different best-layers across datasets are common — also do at a shared layer)
        # Use last-but-one layer for fairness; here, use min(bl_c, bl_u).
        shared_layer = min(bl_c, bl_u)
        d_c_shared = get_direction(m, "cicids2017", shared_layer)
        d_u_shared = get_direction(m, "unsw_nb15", shared_layer)

        cs_best = cos_sim(d_c, d_u) if bl_c == bl_u else None
        cs_shared = cos_sim(d_c_shared, d_u_shared) if d_c_shared is not None and d_u_shared is not None else None

        # Project UNSW with CICIDS direction & vice versa, see attack/normal separation
        # use shared layer for fair comparison
        rec = {
            "best_layer_cicids": bl_c,
            "best_layer_unsw": bl_u,
            "shared_layer": shared_layer,
            "cos_sim_best": cs_best,
            "cos_sim_shared_layer": cs_shared,
        }

        # Use shared_layer reps for projection
        for src_ds, tgt_ds, d_src in [
            ("cicids2017", "unsw_nb15", d_c_shared),
            ("unsw_nb15", "cicids2017", d_u_shared),
        ]:
            tgt_train = load_split(m, tgt_ds, "train")
            if tgt_train is None:
                continue
            H_tgt = to_f32(tgt_train["hidden_states"][shared_layer])
            y_tgt = tgt_train["labels"].numpy()
            proj = project_onto_direction(H_tgt, d_src)
            # AUROC of separation
            from sklearn.metrics import roc_auc_score
            auroc = float(roc_auc_score(y_tgt, proj))
            rec[f"transfer_{src_ds}->{tgt_ds}_train_auroc"] = auroc
        out[m] = rec
        print(f"\n  [{m}]")
        print(f"    cicids best layer = {bl_c}, unsw best layer = {bl_u}")
        print(f"    cos(cicids_dir, unsw_dir) @ shared layer {shared_layer} = {cs_shared:+.3f}")
        for k, v in rec.items():
            if "transfer" in k:
                print(f"    {k} = {v:.3f}")
    json.dump(out, open(OUT / "phase1_expD_cross_dataset.json", "w"), indent=2)
    print(f"\n  saved -> phase1_expD_cross_dataset.json")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None,
                    help="Run only listed exps: 1 2 7 a d")
    args = ap.parse_args()

    fns = {
        "1": exp1_cross_model_agreement,
        "2": exp2_per_type_breakdown,
        "7": exp7_layer_dynamics,
        "a": exp_a_minority_train,
        "d": exp_d_cross_dataset,
    }
    selected = args.only or list(fns.keys())
    for k in selected:
        if k.lower() in fns:
            fns[k.lower()]()
    print("\n========== Phase 1 done ==========")
