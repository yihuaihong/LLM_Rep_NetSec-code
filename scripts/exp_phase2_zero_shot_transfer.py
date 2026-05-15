"""Zero-shot direction transfer: project new-dataset reps through cicids2017
or unsw_nb15 attack_direction and check AUROC.

Datasets tested: cicids2018, iot23, ctu13
Source directions: cicids2017 (same model), unsw_nb15 (same model), random control.
Also: in-domain logistic probe AUROC (upper bound for that dataset's separability).

Hypothesis test:
  - If transfer AUROC ≈ chance (~0.5) for ALL new datasets:
    → "attack direction" is dataset-specific lexical hash. Hard.
  - If schema-match (cicids2017→2018) is high but schema-mismatch (→iot23/ctu13)
    is chance: → it's schema-level lexical hash, still no abstract concept.
  - If schema-mismatch is also high (>0.7): → real abstract attack concept.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

from utils.probe_utils import extract_separation_direction, project_onto_direction

BASE = Path(__file__).resolve().parent.parent
REPS = BASE / "results" / "representations"
OUT = BASE / "results" / "metrics"

NEW_DATASETS = ["cicids2018", "iot23", "ctu13"]
SOURCE_DATASETS = ["cicids2017", "unsw_nb15"]


def to_f32(t):
    """Cast to float32 (handles bfloat16 + float16)."""
    return t.float() if t.dtype in (torch.bfloat16, torch.float16) else t


_dir_cache = {}
def get_source_dir(model, src_ds, layer):
    key = (model, src_ds, layer)
    if key in _dir_cache:
        return _dir_cache[key]
    train = torch.load(REPS / src_ds / model / "train.pt", weights_only=False)
    H = to_f32(train["hidden_states"][layer])
    y = train["labels"].numpy()
    d = extract_separation_direction(H, y)
    _dir_cache[key] = d
    return d


def auroc_safe(y, scores):
    if len(set(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def in_domain_probe_auroc(H, y, n_folds=3):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1))
    scores = cross_val_score(pipe, H.numpy(), y, cv=n_folds, scoring="roc_auc", n_jobs=n_folds)
    return float(scores.mean())


def run_for_model(model, fast_probe=True):
    out = {}
    rng = np.random.RandomState(0)
    # Pre-load source caches once to determine valid shared layers
    src_caches = {}
    for src in SOURCE_DATASETS:
        p = REPS / src / model / "train.pt"
        if p.exists():
            src_caches[src] = torch.load(p, weights_only=False)
    src_layers = set()
    if src_caches:
        src_layers = set.intersection(*[set(c["hidden_states"].keys()) for c in src_caches.values()])
    for new_ds in NEW_DATASETS:
        zs_path = REPS / new_ds / model / "zero_shot.pt"
        if not zs_path.exists():
            print(f"  [{model}/{new_ds}] no zero_shot.pt, skip"); continue
        zs = torch.load(zs_path, weights_only=False)
        zs_layers = set(zs["hidden_states"].keys())
        layers = sorted(zs_layers & src_layers) if src_layers else sorted(zs_layers)
        y_new = zs["labels"].numpy()
        types_new = np.array(zs["attack_types"])
        out[new_ds] = {"layers": layers, "n_normal": int((y_new == 0).sum()),
                       "n_attack": int((y_new == 1).sum()),
                       "attack_types": dict(zip(*np.unique(types_new, return_counts=True))),
                       "per_layer": {}}
        out[new_ds]["attack_types"] = {k: int(v) for k, v in out[new_ds]["attack_types"].items()}

        for L in layers:
            H_new = to_f32(zs["hidden_states"][L])
            rec = {}
            # in-domain probe (CV on the zero_shot reps themselves)
            if fast_probe:
                rec["in_domain_probe_auroc"] = in_domain_probe_auroc(H_new, y_new)
            # source directions
            for src in SOURCE_DATASETS:
                d = get_source_dir(model, src, L)
                proj = project_onto_direction(H_new, d)
                rec[f"{src}_dir_auroc"] = auroc_safe(y_new, proj)
            # random control
            rd = torch.randn(H_new.shape[1], generator=torch.Generator().manual_seed(int(L)+7), dtype=torch.float32)
            rd = rd / rd.norm()
            proj_rand = project_onto_direction(H_new, rd)
            rec["random_dir_auroc"] = auroc_safe(y_new, proj_rand)
            out[new_ds]["per_layer"][L] = rec
    return out


def pretty_print(out, model):
    print(f"\n=== {model} ===")
    for ds, d in out.items():
        print(f"\n  [{ds}]  n_normal={d['n_normal']}  n_attack={d['n_attack']}")
        print(f"    attack types: {d['attack_types']}")
        print(f"    {'L':>3}  {'in-domain':>10}  {'cic2017→':>10}  {'unsw→':>8}  {'random':>8}")
        for L in d["layers"]:
            r = d["per_layer"][L]
            print(f"    {L:>3}  "
                  f"{r.get('in_domain_probe_auroc', float('nan')):>10.3f}  "
                  f"{r.get('cicids2017_dir_auroc', float('nan')):>10.3f}  "
                  f"{r.get('unsw_nb15_dir_auroc', float('nan')):>8.3f}  "
                  f"{r.get('random_dir_auroc', float('nan')):>8.3f}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["Qwen3-8B-Base"])
    args = ap.parse_args()

    big = {}
    for m in args.models:
        out = run_for_model(m)
        big[m] = out
        pretty_print(out, m)

    out_path = OUT / "phase2_zero_shot_transfer.json"
    json.dump(big, open(out_path, "w"), indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
