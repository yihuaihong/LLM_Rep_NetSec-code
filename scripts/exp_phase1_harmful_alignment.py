"""Exp 6: cos sim of harmful direction vs CICIDS/UNSW attack direction.

For each model, at each shared layer between JBB and netsec reps:
  d_harm = mean(harmful) - mean(benign) on JBB reps
  d_cic  = mean(attack) - mean(normal)  on CICIDS train reps
  d_uns  = mean(attack) - mean(normal)  on UNSW train reps
  cos(harmful, cicids), cos(harmful, unsw), cos(cicids, unsw)

Models: Meta-Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.3, Qwen3-8B, gemma-2-9b-it
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path

from utils.probe_utils import extract_separation_direction

BASE = Path(__file__).resolve().parent.parent
REPS = BASE / "results" / "representations"
OUT = BASE / "results" / "metrics"

MODELS = [
    "Meta-Llama-3-8B-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "Qwen3-8B",
    "gemma-2-9b-it",
]


def to_f32(t):
    return t.float() if t.dtype == torch.bfloat16 else t


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(
        a.float().unsqueeze(0), b.float().unsqueeze(0)
    ).item())


def run_one(model):
    jbb_path = REPS / "jbb" / model / "all.pt"
    cic_path = REPS / "cicids2017" / model / "train.pt"
    uns_path = REPS / "unsw_nb15" / model / "train.pt"
    if not (jbb_path.exists() and cic_path.exists() and uns_path.exists()):
        print(f"  [{model}] missing one of jbb/cicids/unsw reps, skip")
        return None

    jbb = torch.load(jbb_path, weights_only=False)
    cic = torch.load(cic_path, weights_only=False)
    uns = torch.load(uns_path, weights_only=False)
    jbb_layers = set(jbb["hidden_states"].keys())
    netsec_layers = set(cic["hidden_states"].keys()) & set(uns["hidden_states"].keys())
    shared_layers = sorted(jbb_layers & netsec_layers)

    print(f"\n=== {model} ===")
    print(f"  shared layers ({len(shared_layers)}): {shared_layers}")
    print(f"  {'layer':>5}  {'cos(harmful,cic)':>18}  {'cos(harmful,uns)':>18}  {'cos(cic,uns)':>14}")
    print("  " + "-" * 70)

    out = {"model": model, "shared_layers": shared_layers, "per_layer": {}}
    for L in shared_layers:
        h_harm = to_f32(jbb["hidden_states"][L])
        d_harm = extract_separation_direction(h_harm, jbb["labels"].numpy())
        h_cic = to_f32(cic["hidden_states"][L])
        d_cic = extract_separation_direction(h_cic, cic["labels"].numpy())
        h_uns = to_f32(uns["hidden_states"][L])
        d_uns = extract_separation_direction(h_uns, uns["labels"].numpy())

        c_hc = cos(d_harm, d_cic)
        c_hu = cos(d_harm, d_uns)
        c_cu = cos(d_cic, d_uns)
        out["per_layer"][L] = {
            "cos_harmful_cicids": c_hc,
            "cos_harmful_unsw": c_hu,
            "cos_cicids_unsw": c_cu,
        }
        print(f"   {L:>3}     {c_hc:>+18.4f}  {c_hu:>+18.4f}  {c_cu:>+14.4f}")
    return out


def main():
    all_results = {}
    for m in MODELS:
        r = run_one(m)
        if r is not None:
            all_results[m] = r
    json.dump(all_results, open(OUT / "phase1_exp6_harmful_alignment_4models.json", "w"), indent=2)
    print("\nsaved -> phase1_exp6_harmful_alignment_4models.json")

    # Print summary table
    print("\n" + "=" * 80)
    print("Summary @ each model's middle/last layer")
    print("=" * 80)
    print(f"  {'model':30s} {'layer':>5}  {'cos(harm,cic)':>14}  {'cos(harm,uns)':>14}  {'cos(cic,uns)':>14}")
    for m, r in all_results.items():
        layers = r["shared_layers"]
        last_L = layers[-1]
        d = r["per_layer"][last_L]
        print(f"  {m:30s} {last_L:>5}  "
              f"{d['cos_harmful_cicids']:>+14.4f}  "
              f"{d['cos_harmful_unsw']:>+14.4f}  "
              f"{d['cos_cicids_unsw']:>+14.4f}")


if __name__ == "__main__":
    main()
