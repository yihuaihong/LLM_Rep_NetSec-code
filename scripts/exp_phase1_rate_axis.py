"""Exp B: cosine of attack direction with a 'flow rate' axis.

Procedure:
1. Take normal-only samples that are also in the train split.
2. Sort by Flow Bytes/s (CICIDS) or sbytes/duration (UNSW).
3. Top-k vs bottom-k normal: compute mean-diff direction => "rate axis".
4. Cosine sim with the train attack direction.

We need raw numerical values per cached sample. The cached .pt does not store the
original numeric features, only labels + attack_types + hidden_states. So we
re-load the parquet with the SAME random subsample by reproducing the seeded split.

Workaround: we instead build a rate axis from RAW NUMERIC VALUES IN THE PARQUET
on a fresh sample of normal-only rows, and project the attack direction onto a
PROXY: we train a logistic probe to predict "high-rate vs low-rate normal"
labels, giving us a "rate-direction" in the same hidden space, and then take
cosine with attack direction.

We use one model (Llama-3.1-8B-Instruct) because cross-model rate axis would
require re-extracting reps for each model on a new normal subsample => too slow.
"""
import os, sys, json, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from pathlib import Path

from utils.data_utils import load_dataset, format_dataset
from utils.probe_utils import (
    extract_separation_direction,
    project_onto_direction,
)
from utils.model_utils import load_model_and_tokenizer, get_layer_count, extract_hidden_states

BASE = Path(__file__).resolve().parent.parent
REPS = BASE / "results" / "representations"
OUT = BASE / "results" / "metrics"
CACHE_DIR = "/scratch/yh6210/transformers"

DATASET_PATHS = {
    "cicids2017": "/scratch/yh6210/datasets/LLM_Rep_NetSec_datasets/cicids2017/",
    "unsw_nb15":  "/scratch/yh6210/datasets/LLM_Rep_NetSec_datasets/unsw_nb15/",
}

RATE_FEATURE = {
    "cicids2017": "Flow Bytes/s",
    "unsw_nb15":  "sbytes",  # raw byte count proxy
}


def to_f32(t):
    return t.float() if t.dtype == torch.bfloat16 else t


def cos_sim(a, b):
    return float(torch.nn.functional.cosine_similarity(
        a.float().unsqueeze(0), b.float().unsqueeze(0)
    ).item())


def get_train_direction(model, ds, layer):
    train = torch.load(REPS / ds / model / "train.pt", weights_only=False)
    H = to_f32(train["hidden_states"][layer])
    y = train["labels"].numpy()
    return extract_separation_direction(H, y)


def build_rate_axis(model_name, dataset_name, layer, n_normal_per_bin=500, seed=123):
    """
    Build a rate axis at the given layer for a given model.
    Uses fresh normal samples (NOT from train cache, to avoid leakage).
    Returns (rate_direction, attack_direction_at_same_layer).
    """
    print(f"\n[{model_name}/{dataset_name}] building rate axis @ layer {layer}", flush=True)
    df = load_dataset(dataset_name, DATASET_PATHS[dataset_name])
    rate_col = RATE_FEATURE[dataset_name]
    if rate_col not in df.columns:
        raise RuntimeError(f"missing column {rate_col}")
    # Normal-only rows
    df_n = df[(df["is_attack"] == 0) & df[rate_col].notna() & np.isfinite(df[rate_col])].copy()
    rate = df_n[rate_col].values
    # Top-k and bottom-k by rate (k each)
    rng = np.random.RandomState(seed)
    sorted_idx = np.argsort(rate)
    k = min(n_normal_per_bin, len(sorted_idx) // 4)
    bot_idx = sorted_idx[:k]
    top_idx = sorted_idx[-k:]
    bot_df = df_n.iloc[bot_idx].copy()
    top_df = df_n.iloc[top_idx].copy()
    bot_df["bin"] = 0
    top_df["bin"] = 1
    sample_df = pd.concat([bot_df, top_df], ignore_index=True)
    sample_df = format_dataset(sample_df, dataset_name, fmt="natural_language")
    print(f"  built sample: {len(sample_df)} rows ({k} low-rate + {k} high-rate)")
    print(f"  low-rate {rate_col}: median={np.median(bot_df[rate_col]):.4g}, "
          f"high-rate median={np.median(top_df[rate_col]):.4g}")

    # Extract hidden states at this layer only
    print(f"  loading model {model_name} ...")
    model_path = os.path.join(CACHE_DIR, model_name)
    dtype = "bfloat16" if "gemma" in model_name.lower() else "float16"
    model, tokenizer = load_model_and_tokenizer(model_path, dtype=dtype, device="cuda")
    hs = extract_hidden_states(
        model, tokenizer, sample_df["text"].tolist(),
        layers=[layer], token_position="last", batch_size=32, max_seq_length=512,
    )
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    H = to_f32(hs[layer])
    bins = sample_df["bin"].values
    # rate axis = mean(top) - mean(bot)
    rate_dir = H[bins == 1].mean(0) - H[bins == 0].mean(0)
    rate_dir = rate_dir / rate_dir.norm()

    attack_dir = get_train_direction(model_name, dataset_name, layer)
    cs = cos_sim(rate_dir, attack_dir)
    print(f"  cos(attack_dir, rate_axis) @ layer {layer} = {cs:+.4f}")
    return {
        "model": model_name,
        "dataset": dataset_name,
        "layer": layer,
        "n_low_rate": k,
        "n_high_rate": k,
        "low_rate_median": float(np.median(bot_df[rate_col])),
        "high_rate_median": float(np.median(top_df[rate_col])),
        "cos_attack_vs_rate": cs,
        "rate_feature": rate_col,
    }


def main():
    # Just one model for cost — easy to extend later
    model = "Llama-3.1-8B-Instruct"

    out = []
    for ds in ["cicids2017", "unsw_nb15"]:
        # use the best layer for this model on this dataset
        f = OUT / f"{model}_{ds}.json"
        bl = int(json.load(open(f))["best_layer"])
        rec = build_rate_axis(model, ds, bl, n_normal_per_bin=500)
        out.append(rec)
    json.dump(out, open(OUT / "phase1_expB_rate_axis.json", "w"), indent=2)
    print("\nsaved -> phase1_expB_rate_axis.json")
    print("\nSummary:")
    for r in out:
        print(f"  {r['model']:30s} {r['dataset']:10s} layer={r['layer']:>2}  "
              f"cos(attack, rate) = {r['cos_attack_vs_rate']:+.4f}")


if __name__ == "__main__":
    main()
