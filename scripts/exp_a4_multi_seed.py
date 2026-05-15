"""A4: multi-seed bootstrap of probe gap for each (model, dataset).

Re-run probe (StandardScaler + LogReg, 5-fold CV) with 3 different seeds
{42, 7, 99} on cached reps. The seed affects:
  - StratifiedKFold split (cross_val_score)
  - LogReg internal random_state
  - create_generalization_splits via global numpy seed (CANNOT change post-hoc — splits are fixed by extracted .pt files; we only re-shuffle the CV).

So this only reports VARIANCE OF CV ESTIMATE for the same (already-extracted) data,
not full re-extraction variance. Still important for reporting.

For the held-out test sets, we re-fit the probe with seed and re-evaluate.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score

BASE = Path(__file__).resolve().parent.parent
REPS = BASE / "results" / "representations"
OUT = BASE / "results" / "metrics"

MODELS = ["Llama-3.1-8B-Instruct", "Mistral-7B-Instruct-v0.3",
          "Qwen3-8B", "gemma-2-9b-it"]
DATASETS = ["cicids2017", "unsw_nb15"]
SEEDS = [42, 7, 99]


def to_f32(t):
    return t.float() if t.dtype in (torch.bfloat16, torch.float16) else t


def best_layer(model, ds):
    f = OUT / f"{model}_{ds}.json"
    if not f.exists():
        return None
    return int(json.load(open(f))["best_layer"])


def run(model, ds, seed):
    bl = best_layer(model, ds)
    if bl is None:
        return None
    train = torch.load(REPS / ds / model / "train.pt", weights_only=False)
    known = torch.load(REPS / ds / model / "test_known.pt", weights_only=False)
    holdout = torch.load(REPS / ds / model / "test_holdout.pt", weights_only=False)
    X_tr = to_f32(train["hidden_states"][bl]).numpy()
    y_tr = train["labels"].numpy()
    X_kn = to_f32(known["hidden_states"][bl]).numpy()
    y_kn = known["labels"].numpy()
    X_ho = to_f32(holdout["hidden_states"][bl]).numpy()
    y_ho = holdout["labels"].numpy()

    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(C=1.0, max_iter=2000,
                                            random_state=seed, n_jobs=-1))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_auc = float(cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc").mean())

    pipe.fit(X_tr, y_tr)
    auc_kn = float(roc_auc_score(y_kn, pipe.predict_proba(X_kn)[:, 1]))
    acc_kn = float(accuracy_score(y_kn, pipe.predict(X_kn)))
    auc_ho = float(roc_auc_score(y_ho, pipe.predict_proba(X_ho)[:, 1]))
    acc_ho = float(accuracy_score(y_ho, pipe.predict(X_ho)))
    return {
        "seed": seed, "best_layer": bl,
        "cv_auroc": cv_auc,
        "known_auroc": auc_kn, "known_acc": acc_kn,
        "holdout_auroc": auc_ho, "holdout_acc": acc_ho,
        "gap_auroc": auc_kn - auc_ho,
        "gap_acc": acc_kn - acc_ho,
    }


def main():
    big = {}
    for m in MODELS:
        big[m] = {}
        for ds in DATASETS:
            big[m][ds] = []
            for seed in SEEDS:
                r = run(m, ds, seed)
                if r is None:
                    continue
                big[m][ds].append(r)
                print(f"  {m:30s} {ds:10s} seed={seed}  cv={r['cv_auroc']:.4f}  "
                      f"known={r['known_acc']:.3f}  holdout={r['holdout_acc']:.3f}  "
                      f"gap={r['gap_acc']:+.3f}")

    print("\n========== Summary (mean ± std over seeds) ==========")
    summary = {}
    for m, dd in big.items():
        summary[m] = {}
        for ds, runs in dd.items():
            if not runs:
                continue
            ka = np.array([r['known_acc'] for r in runs])
            ho = np.array([r['holdout_acc'] for r in runs])
            gap = np.array([r['gap_acc'] for r in runs])
            summary[m][ds] = {
                "known_acc_mean": float(ka.mean()), "known_acc_std": float(ka.std()),
                "holdout_acc_mean": float(ho.mean()), "holdout_acc_std": float(ho.std()),
                "gap_mean": float(gap.mean()), "gap_std": float(gap.std()),
            }
            print(f"  {m:30s} {ds:10s}  known {ka.mean():.3f}±{ka.std():.4f}  "
                  f"holdout {ho.mean():.3f}±{ho.std():.4f}  "
                  f"gap {gap.mean():+.3f}±{gap.std():.4f}")

    json.dump({"runs": big, "summary": summary},
              open(OUT / "phase2_a4_multi_seed.json", "w"), indent=2)
    print("\nsaved -> phase2_a4_multi_seed.json")


if __name__ == "__main__":
    main()
