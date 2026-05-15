"""Compare cos(harmful, attack) on Qwen3-8B-Base vs Qwen3-8B-Instruct.

Assumes:
  - results/representations/jbb/Qwen3-8B/all.pt (instruct, with chat template)
  - results/representations/jbb/Qwen3-8B/all_raw.pt (instruct, NO chat template)
  - results/representations/jbb/Qwen3-8B-Base/all_raw.pt (base, NO chat template)
  - results/representations/cicids2017/Qwen3-8B/{train,test_known,test_holdout}.pt
  - results/representations/cicids2017/Qwen3-8B-Base/{train,test_known,test_holdout}.pt
  - same for unsw_nb15

For each variant, compute layer-wise:
  cos(harmful_dir, cicids_attack_dir)
  cos(harmful_dir, unsw_attack_dir)
  cos(cicids_attack_dir, unsw_attack_dir)
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


def to_f32(t):
    return t.float() if t.dtype == torch.bfloat16 else t


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(
        a.float().unsqueeze(0), b.float().unsqueeze(0)
    ).item())


def compute_dirs(model, jbb_filename):
    """Return dict layer -> {d_harm, d_cic, d_uns}."""
    jbb_path = REPS / "jbb" / model / jbb_filename
    cic_path = REPS / "cicids2017" / model / "train.pt"
    uns_path = REPS / "unsw_nb15" / model / "train.pt"
    if not (jbb_path.exists() and cic_path.exists() and uns_path.exists()):
        print(f"  [{model}/{jbb_filename}] missing one of: {jbb_path.exists()=} {cic_path.exists()=} {uns_path.exists()=}")
        return None
    jbb = torch.load(jbb_path, weights_only=False)
    cic = torch.load(cic_path, weights_only=False)
    uns = torch.load(uns_path, weights_only=False)
    layers = sorted(set(jbb["hidden_states"].keys()) & set(cic["hidden_states"].keys()) & set(uns["hidden_states"].keys()))
    out = {}
    for L in layers:
        d_h = extract_separation_direction(to_f32(jbb["hidden_states"][L]), jbb["labels"].numpy())
        d_c = extract_separation_direction(to_f32(cic["hidden_states"][L]), cic["labels"].numpy())
        d_u = extract_separation_direction(to_f32(uns["hidden_states"][L]), uns["labels"].numpy())
        out[L] = {
            "cos_harm_cic": cos(d_h, d_c),
            "cos_harm_uns": cos(d_h, d_u),
            "cos_cic_uns": cos(d_c, d_u),
        }
    return out


def main():
    variants = [
        ("Qwen3-8B",      "all.pt",     "instruct (chat-template)"),
        ("Qwen3-8B",      "all_raw.pt", "instruct (raw)"),
        ("Qwen3-8B-Base", "all_raw.pt", "base (raw)"),
    ]
    results = {}
    for model, jbb_file, label in variants:
        print(f"\n=== {label} ===  ({model} / {jbb_file})")
        d = compute_dirs(model, jbb_file)
        if d is None:
            continue
        results[label] = d
        print(f"  {'L':>3}  {'cos(h,cic)':>12}  {'cos(h,uns)':>12}  {'cos(cic,uns)':>14}")
        for L, v in d.items():
            print(f"  {L:>3}  {v['cos_harm_cic']:>+12.4f}  {v['cos_harm_uns']:>+12.4f}  {v['cos_cic_uns']:>+14.4f}")

    json.dump(results, open(OUT / "phase1_exp6_qwen3_base_vs_instruct.json", "w"), indent=2)
    print("\nsaved -> phase1_exp6_qwen3_base_vs_instruct.json")


if __name__ == "__main__":
    main()
