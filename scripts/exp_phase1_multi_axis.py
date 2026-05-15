"""Multi-axis cosine: probe attack_direction's projection onto multiple
feature axes built from normal-only samples.

Each axis = mean(top-k normal sorted by feature) − mean(bot-k normal sorted by feature).

For each (model, dataset), we already have train hidden states cached. The
challenge: we need to know which raw numeric values correspond to which cached
sample. The cache stores attack_types but not raw numeric features.

Workaround: re-run create_generalization_splits with the SAME seed + max samples,
which gives us the same (deterministic) train DataFrame. Cached reps[i] then
correspond to row i of that DataFrame.

We use Llama-3.1-8B-Instruct + cicids2017 + Qwen3-8B + cicids2017 for now.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from pathlib import Path

from utils.data_utils import load_dataset, format_dataset, create_generalization_splits
from utils.probe_utils import extract_separation_direction

BASE = Path(__file__).resolve().parent.parent
REPS = BASE / "results" / "representations"
OUT = BASE / "results" / "metrics"

DATASET_PATHS = {
    "cicids2017": "/scratch/yh6210/datasets/LLM_Rep_NetSec_datasets/cicids2017/",
    "unsw_nb15":  "/scratch/yh6210/datasets/LLM_Rep_NetSec_datasets/unsw_nb15/",
}
HOLDOUT = {
    "cicids2017": ["Heartbleed", "Infiltration", "Bot"],
    "unsw_nb15":  ["Shellcode", "Worms", "Backdoor"],
}

# Numeric features to test as candidate axes
CIC_AXES = [
    "Flow Bytes/s", "Flow Packets/s", "Flow Duration",
    "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd IAT Mean", "Bwd IAT Mean",
    "SYN Flag Count", "RST Flag Count", "ACK Flag Count", "PSH Flag Count",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "destination_port",
]
UNSW_AXES = [
    "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss",
    "spkts", "dpkts", "swin", "dwin", "smean", "dmean",
    "ct_srv_src", "ct_dst_sport_ltm", "destination_port",
]


def to_f32(t):
    return t.float() if t.dtype == torch.bfloat16 else t


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(
        a.float().unsqueeze(0), b.float().unsqueeze(0)
    ).item())


def best_layer(model, ds):
    f = OUT / f"{model}_{ds}.json"
    if not f.exists():
        return None
    return int(json.load(open(f))["best_layer"])


def reproduce_train_df(ds_name):
    """Re-run create_generalization_splits with seed=42 and return the train DataFrame.

    Skips format_dataset since we don't need the 'text' column for axis building —
    we only use raw numeric columns. format_dataset on 2.8M rows takes 10+ min.
    """
    df = load_dataset(ds_name, DATASET_PATHS[ds_name])
    splits = create_generalization_splits(
        df, holdout_attack_types=HOLDOUT[ds_name],
        max_samples_per_class=5000, seed=42,
    )
    return splits["train"].reset_index(drop=True)


def build_axis_for_feature(H, df, feat_name, k=500):
    """Return a unit-norm direction in hidden space corresponding to high vs low values of feat_name.

    H: (n, hidden_dim) tensor of hidden states for normal-only samples in df.
    df: corresponding DataFrame slice (normal-only, same indexing as H).
    feat_name: column in df to sort by.
    k: top/bot count.
    """
    if feat_name not in df.columns:
        return None
    vals = df[feat_name].values.astype(np.float64)
    valid = np.isfinite(vals)
    if valid.sum() < 2 * k:
        k = max(50, valid.sum() // 4)
    H = H[valid]
    vals = vals[valid]
    order = np.argsort(vals)
    bot_idx = order[:k]
    top_idx = order[-k:]
    bot_mean = H[bot_idx].mean(0)
    top_mean = H[top_idx].mean(0)
    d = top_mean - bot_mean
    n = d.norm()
    if n < 1e-8:
        return None
    return d / n, float(np.median(vals[bot_idx])), float(np.median(vals[top_idx]))


def run_one(model, ds, axis_features):
    bl = best_layer(model, ds)
    if bl is None:
        print(f"  [{model}/{ds}] no metrics, skip"); return None
    train_path = REPS / ds / model / "train.pt"
    if not train_path.exists():
        print(f"  [{model}/{ds}] no cached reps, skip"); return None

    cache = torch.load(train_path, weights_only=False)
    H = to_f32(cache["hidden_states"][bl])
    y = cache["labels"].numpy()
    types = np.array(cache["attack_types"])

    # Reproduce train df
    train_df = reproduce_train_df(ds)
    if len(train_df) != H.shape[0]:
        print(f"  [{model}/{ds}] mismatch: df={len(train_df)} reps={H.shape[0]}")
        return None

    # Verify alignment via attack_type
    cached_types = list(types)
    df_types = list(train_df["attack_type"].values)
    if cached_types != df_types:
        print(f"  [{model}/{ds}] WARN: type-by-row mismatch, first 5 cache: {cached_types[:5]} df: {df_types[:5]}")
        # Don't fail — but report mismatch count
        n_mismatch = sum(1 for a, b in zip(cached_types, df_types) if a != b)
        print(f"    n_mismatch = {n_mismatch}/{len(cached_types)}")

    # attack direction
    d_attack = extract_separation_direction(H, y)

    # Normal-only subset
    normal_mask = (y == 0)
    H_n = H[normal_mask]
    df_n = train_df.iloc[normal_mask].reset_index(drop=True)

    # Build axes
    print(f"\n=== {model} / {ds}  layer={bl}  n_train={len(train_df)}  n_normal={normal_mask.sum()} ===")
    print(f"  {'feature':30s} {'cos(attack, axis)':>20}  {'low_med':>10}  {'high_med':>10}")
    rows = []
    for feat in axis_features:
        res = build_axis_for_feature(H_n, df_n, feat, k=min(500, normal_mask.sum() // 4))
        if res is None:
            print(f"  {feat:30s} —  (no col / no variance)")
            continue
        d_axis, low_m, high_m = res
        c = cos(d_attack, d_axis)
        rows.append({"feature": feat, "cos": c, "low_med": low_m, "high_med": high_m})
        print(f"  {feat:30s} {c:>+20.4f}  {low_m:>10.3g}  {high_m:>10.3g}")
    rows.sort(key=lambda r: -abs(r["cos"]))
    return rows


def main():
    out = {}
    for model in ["Llama-3.1-8B-Instruct", "Qwen3-8B"]:
        out[model] = {}
        for ds, axes in [("cicids2017", CIC_AXES), ("unsw_nb15", UNSW_AXES)]:
            r = run_one(model, ds, axes)
            if r is None:
                continue
            out[model][ds] = r

    json.dump(out, open(OUT / "phase1_multi_axis_decomposition.json", "w"), indent=2)
    print("\nsaved -> phase1_multi_axis_decomposition.json")

    print("\n========== Top 5 axes per (model, ds) sorted by |cos| ==========")
    for m, dd in out.items():
        for ds, rows in dd.items():
            print(f"\n  [{m} / {ds}]")
            for r in rows[:5]:
                print(f"    {r['feature']:30s}  cos={r['cos']:+.4f}  (low={r['low_med']:.3g}, high={r['high_med']:.3g})")


if __name__ == "__main__":
    main()
