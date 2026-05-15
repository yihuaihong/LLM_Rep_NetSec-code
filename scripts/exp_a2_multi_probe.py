"""A2: try 4 probe types — LR, MLP, LDA, PCA->LR — on cached reps.

For each (model, dataset, probe_type):
  - Train at best layer
  - Report 5-fold CV AUROC, in-domain test_known AUROC, holdout AUROC, gap

This blocks the reviewer comment "you used a weak probe; a stronger probe
would have found a real concept".
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

BASE = Path(__file__).resolve().parent.parent
REPS = BASE / "results" / "representations"
OUT = BASE / "results" / "metrics"

MODELS = ["Qwen3-8B-Base", "Meta-Llama-3-8B-Instruct"]  # extras only — main 4 already saved
DATASETS = ["cicids2017", "unsw_nb15"]


def to_f32(t):
    return t.float() if t.dtype in (torch.bfloat16, torch.float16) else t


def best_layer(model, ds):
    f = OUT / f"{model}_{ds}.json"
    return int(json.load(open(f))["best_layer"]) if f.exists() else None


def make_probe(name):
    if name == "lr":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=1.0, max_iter=2000, n_jobs=-1, random_state=42))
    if name == "lda":
        return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    if name == "pca_lr":
        return make_pipeline(StandardScaler(), PCA(n_components=128, random_state=42),
                             LogisticRegression(C=1.0, max_iter=2000, n_jobs=-1, random_state=42))
    if name == "mlp":
        return make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(256,), max_iter=300,
                                           random_state=42, early_stopping=True))
    raise ValueError(name)


def run_one(model, ds, probe_name):
    bl = best_layer(model, ds)
    train = torch.load(REPS / ds / model / "train.pt", weights_only=False)
    known = torch.load(REPS / ds / model / "test_known.pt", weights_only=False)
    holdout = torch.load(REPS / ds / model / "test_holdout.pt", weights_only=False)
    X_tr = to_f32(train["hidden_states"][bl]).numpy()
    y_tr = train["labels"].numpy()
    X_kn = to_f32(known["hidden_states"][bl]).numpy()
    y_kn = known["labels"].numpy()
    X_ho = to_f32(holdout["hidden_states"][bl]).numpy()
    y_ho = holdout["labels"].numpy()

    probe = make_probe(probe_name)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    try:
        cv_auc = float(cross_val_score(probe, X_tr, y_tr, cv=cv, scoring="roc_auc").mean())
    except Exception as e:
        cv_auc = float("nan")

    probe.fit(X_tr, y_tr)
    if hasattr(probe, "predict_proba"):
        try:
            kn_proba = probe.predict_proba(X_kn)[:, 1]
            ho_proba = probe.predict_proba(X_ho)[:, 1]
        except Exception:
            kn_proba = probe.decision_function(X_kn)
            ho_proba = probe.decision_function(X_ho)
    else:
        kn_proba = probe.decision_function(X_kn)
        ho_proba = probe.decision_function(X_ho)
    auc_kn = float(roc_auc_score(y_kn, kn_proba))
    auc_ho = float(roc_auc_score(y_ho, ho_proba))
    acc_kn = float(accuracy_score(y_kn, probe.predict(X_kn)))
    acc_ho = float(accuracy_score(y_ho, probe.predict(X_ho)))
    return {
        "best_layer": bl,
        "cv_auroc": cv_auc,
        "known_auroc": auc_kn, "known_acc": acc_kn,
        "holdout_auroc": auc_ho, "holdout_acc": acc_ho,
        "gap_auroc": auc_kn - auc_ho,
        "gap_acc": acc_kn - acc_ho,
    }


def main():
    out_path = OUT / "phase2_a2_multi_probe.json"
    big = json.load(open(out_path)) if out_path.exists() else {}  # append to existing
    print(f"  {'model':30s} {'ds':10s} {'probe':>8}  {'cv_auc':>7}  {'kn_acc':>7}  {'ho_acc':>7}  {'gap':>7}")
    print("-" * 95)
    for m in MODELS:
        if m not in big:
            big[m] = {}
        for ds in DATASETS:
            if ds not in big[m]:
                big[m][ds] = {}
            for pname in ["lr", "lda", "pca_lr", "mlp"]:
                if pname in big[m][ds]:
                    continue  # already done
                try:
                    r = run_one(m, ds, pname)
                except Exception as e:
                    print(f"  [{m}/{ds}/{pname}] error: {e}")
                    continue
                big[m][ds][pname] = r
                print(f"  {m:30s} {ds:10s} {pname:>8}  "
                      f"{r['cv_auroc']:>7.4f}  {r['known_acc']:>7.3f}  "
                      f"{r['holdout_acc']:>7.3f}  {r['gap_acc']:>+7.3f}")
                json.dump(big, open(out_path, "w"), indent=2)  # checkpoint after each
    print("\nsaved -> phase2_a2_multi_probe.json")


if __name__ == "__main__":
    main()
