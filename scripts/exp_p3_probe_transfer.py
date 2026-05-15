"""P3: probe-based (logistic regression) cross-dataset transfer.

V4 used the simpler mean-diff direction. Within-source AUROC of mean-diff
was only 0.7-0.74. A reviewer can argue the weak direction is what fails
to transfer. We therefore re-run the V4 experiment with the FULL logistic
regression probe.

Pipeline:
  1. Fit StandardScaler + LogReg on cic2017 train hidden states (per layer).
  2. predict_proba on the new dataset's hidden states (zero_shot.pt).
  3. Report AUROC.

This tightens our V4 conclusion: even with the strongest probe we have,
direction does not transfer.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parent.parent
REPS = BASE / "results" / "representations"
OUT = BASE / "results" / "metrics"

MODELS = ["Meta-Llama-3-8B-Instruct", "Mistral-7B-Instruct-v0.3",
          "Qwen3-8B-Base", "gemma-2-9b-it"]
SOURCE_DATASETS = ["cicids2017", "unsw_nb15"]
NEW_DATASETS = ["cicids2018", "iot23", "ctu13"]


def to_f32(t):
    return t.float() if t.dtype in (torch.bfloat16, torch.float16) else t


def fit_probe(model_name, src_ds, layer):
    train = torch.load(REPS / src_ds / model_name / "train.pt", weights_only=False)
    X = to_f32(train["hidden_states"][layer]).numpy()
    y = train["labels"].numpy()
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(C=1.0, max_iter=2000, n_jobs=-1, random_state=42))
    pipe.fit(X, y)
    return pipe


def run_for_model(model):
    out = {}
    # Pre-load source caches for layer intersections
    src_caches = {}
    for src in SOURCE_DATASETS:
        p = REPS / src / model / "train.pt"
        if p.exists():
            src_caches[src] = torch.load(p, weights_only=False)
    if not src_caches:
        return None
    src_layers = set.intersection(*[set(c["hidden_states"].keys()) for c in src_caches.values()])

    for new_ds in NEW_DATASETS:
        zs_path = REPS / new_ds / model / "zero_shot.pt"
        if not zs_path.exists():
            print(f"  [{model}/{new_ds}] no zero_shot.pt"); continue
        zs = torch.load(zs_path, weights_only=False)
        zs_layers = set(zs["hidden_states"].keys())
        layers = sorted(zs_layers & src_layers)
        y_new = zs["labels"].numpy()
        out[new_ds] = {"layers": layers, "n_normal": int((y_new == 0).sum()),
                       "n_attack": int((y_new == 1).sum()), "per_layer": {}}
        for L in layers:
            X_new = to_f32(zs["hidden_states"][L]).numpy()
            rec = {}
            for src in SOURCE_DATASETS:
                if src not in src_caches: continue
                pipe = fit_probe(model, src, L)
                proba = pipe.predict_proba(X_new)[:, 1]
                rec[f"{src}_probe_auroc"] = float(roc_auc_score(y_new, proba))
            out[new_ds]["per_layer"][L] = rec
    return out


def main():
    big = {}
    for m in MODELS:
        out = run_for_model(m)
        if out is None: continue
        big[m] = out
        print(f"\n=== {m} ===")
        for ds, d in out.items():
            print(f"\n  [{ds}]  n={d['n_normal']}+{d['n_attack']}")
            print(f"    {'L':>3}  {'cic2017_probe':>14}  {'unsw_probe':>10}")
            for L in d["layers"]:
                v = d["per_layer"][L]
                cic = v.get('cicids2017_probe_auroc', float('nan'))
                uns = v.get('unsw_nb15_probe_auroc', float('nan'))
                print(f"    {L:>3}  {cic:>14.3f}  {uns:>10.3f}")
    json.dump(big, open(OUT / "phase2_p3_probe_transfer.json", "w"), indent=2)
    print("\nsaved -> phase2_p3_probe_transfer.json")


if __name__ == "__main__":
    main()
