"""Single-type direction experiment.

For each (model, dataset), for each attack type T in train, build:
  d_T = mean( hidden_states[type==T] ) − mean( hidden_states[type=='Normal'] ),  normalized.

Then compute pairwise cosine matrix across types.

Interpretation:
- If cos(d_X, d_Y) is HIGH (>0.7) for all pairs → unified "attack" axis exists
  in the model's representation space. Lexical-hash hypothesis fails.
- If cos is LOW (<0.5) → each attack type teaches a different direction;
  attack_direction is essentially an arithmetic average of unrelated hashes.

Use models: Llama-3.1-8B-Instruct, Qwen3-8B (the most interesting comparison).
Each model's best layer.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parent.parent
REPS = BASE / "results" / "representations"
OUT = BASE / "results" / "metrics"


def to_f32(t):
    return t.float() if t.dtype == torch.bfloat16 else t


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(
        a.float().unsqueeze(0), b.float().unsqueeze(0)
    ).item())


def best_layer(model, ds):
    f = OUT / f"{model}_{ds}.json"
    return int(json.load(open(f))["best_layer"])


def run(model, ds, min_n=20):
    bl = best_layer(model, ds)
    train = torch.load(REPS / ds / model / "train.pt", weights_only=False)
    H = to_f32(train["hidden_states"][bl])
    types = np.array(train["attack_types"])

    type_counts = Counter(types)
    eligible_attack_types = [
        t for t, n in type_counts.items()
        if t != "Normal" and n >= min_n
    ]
    print(f"\n=== {model} / {ds}  layer={bl} ===")
    print(f"  type counts: {dict(type_counts)}")
    print(f"  eligible attack types (n>={min_n}): {eligible_attack_types}")

    normal_mask = (types == "Normal")
    H_normal = H[normal_mask]
    n_mean = H_normal.mean(0)
    print(f"  n_normal = {normal_mask.sum()}")

    # Build direction per type
    dirs = {}
    for t in eligible_attack_types:
        mask = (types == t)
        if mask.sum() < min_n:
            continue
        atk_mean = H[mask].mean(0)
        d = atk_mean - n_mean
        d = d / d.norm()
        dirs[t] = d

    # Also compute the FULL direction (all-attack vs normal)
    all_atk = (types != "Normal")
    d_full = H[all_atk].mean(0) - n_mean
    d_full = d_full / d_full.norm()
    dirs["__ALL__"] = d_full

    # Pairwise cosine matrix
    keys = list(dirs.keys())
    n = len(keys)
    M = np.zeros((n, n))
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            M[i, j] = cos(dirs[ki], dirs[kj])

    # Pretty print
    print(f"\n  Pairwise cos matrix:")
    print(f"  {'':25s} " + " ".join(f"{k[:12]:>12s}" for k in keys))
    for i, ki in enumerate(keys):
        print(f"  {ki[:25]:25s} " + " ".join(f"{M[i,j]:>+12.3f}" for j in range(n)))

    # Print summary stats (off-diag only, attack-attack)
    attack_keys = [k for k in keys if k != "__ALL__"]
    pairwise = []
    for i in range(len(attack_keys)):
        for j in range(i+1, len(attack_keys)):
            pairwise.append(M[keys.index(attack_keys[i]), keys.index(attack_keys[j])])
    if pairwise:
        print(f"\n  Pairwise attack-attack cos: mean={np.mean(pairwise):+.3f}  median={np.median(pairwise):+.3f}  "
              f"min={np.min(pairwise):+.3f}  max={np.max(pairwise):+.3f}")

    # cos with __ALL__
    cos_with_all = {k: float(M[keys.index(k), keys.index("__ALL__")]) for k in attack_keys}
    print(f"\n  cos with __ALL__ direction:")
    for k, c in sorted(cos_with_all.items(), key=lambda x: -x[1]):
        print(f"    {k:30s}  {c:+.3f}")

    return {
        "model": model,
        "dataset": ds,
        "layer": bl,
        "type_counts": dict(type_counts),
        "keys": keys,
        "cos_matrix": M.tolist(),
        "pairwise_attack_attack": {
            "mean": float(np.mean(pairwise)) if pairwise else None,
            "median": float(np.median(pairwise)) if pairwise else None,
            "min": float(np.min(pairwise)) if pairwise else None,
            "max": float(np.max(pairwise)) if pairwise else None,
        },
        "cos_with_all": cos_with_all,
    }


def main():
    out = {}
    for model in ["Llama-3.1-8B-Instruct", "Qwen3-8B", "Mistral-7B-Instruct-v0.3", "gemma-2-9b-it"]:
        out[model] = {}
        for ds in ["cicids2017", "unsw_nb15"]:
            try:
                out[model][ds] = run(model, ds, min_n=20)
            except Exception as e:
                print(f"  [{model}/{ds}] error: {e}")
    json.dump(out, open(OUT / "phase1_single_type_directions.json", "w"), indent=2)
    print("\nsaved -> phase1_single_type_directions.json")


if __name__ == "__main__":
    main()
