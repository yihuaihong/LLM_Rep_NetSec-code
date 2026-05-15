"""B5: project cicids2017 attack_direction onto JBB harmful reps.

For each model, at each shared layer:
  d_cic = mean-diff direction from cicids2017 train
  proj = JBB hidden states @ d_cic
  AUROC vs harmful label

If cic_dir captures generic "danger / unusual content", AUROC > 0.5 expected.
If cic_dir is purely netsec-specific lexical, AUROC ~ 0.5.

Run on 4 instruct models + Qwen3-8B-Base.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from utils.probe_utils import extract_separation_direction, project_onto_direction

BASE = Path(__file__).resolve().parent.parent
REPS = BASE / "results" / "representations"
OUT = BASE / "results" / "metrics"

MODELS = ["Meta-Llama-3-8B-Instruct", "Mistral-7B-Instruct-v0.3",
          "Qwen3-8B", "Qwen3-8B-Base", "gemma-2-9b-it"]


def to_f32(t):
    return t.float() if t.dtype in (torch.bfloat16, torch.float16) else t


def run_one(model):
    cic_path = REPS / "cicids2017" / model / "train.pt"
    uns_path = REPS / "unsw_nb15"  / model / "train.pt"
    if not cic_path.exists() or not uns_path.exists():
        return None
    cic = torch.load(cic_path, weights_only=False)
    uns = torch.load(uns_path, weights_only=False)
    # try raw + chat-template JBB
    out = {}
    for tag, name in [("chat", "all.pt"), ("raw", "all_raw.pt")]:
        jbb_path = REPS / "jbb" / model / name
        if not jbb_path.exists():
            continue
        jbb = torch.load(jbb_path, weights_only=False)
        y_jbb = jbb["labels"].numpy()
        layers = sorted(set(cic["hidden_states"].keys())
                        & set(uns["hidden_states"].keys())
                        & set(jbb["hidden_states"].keys()))
        rec = {}
        for L in layers:
            d_cic = extract_separation_direction(to_f32(cic["hidden_states"][L]),
                                                 cic["labels"].numpy())
            d_uns = extract_separation_direction(to_f32(uns["hidden_states"][L]),
                                                 uns["labels"].numpy())
            H_jbb = to_f32(jbb["hidden_states"][L])
            proj_c = project_onto_direction(H_jbb, d_cic)
            proj_u = project_onto_direction(H_jbb, d_uns)
            # random control
            rd = torch.randn(H_jbb.shape[1],
                             generator=torch.Generator().manual_seed(int(L)+11),
                             dtype=torch.float32)
            rd = rd / rd.norm()
            proj_r = project_onto_direction(H_jbb, rd)
            rec[int(L)] = {
                "auroc_cic_dir_on_harmful": float(roc_auc_score(y_jbb, proj_c)),
                "auroc_uns_dir_on_harmful": float(roc_auc_score(y_jbb, proj_u)),
                "auroc_random_on_harmful":  float(roc_auc_score(y_jbb, proj_r)),
            }
        out[tag] = rec
    return out


def main():
    big = {}
    for m in MODELS:
        r = run_one(m)
        if r is None:
            continue
        big[m] = r
        print(f"\n=== {m} ===")
        for tag, layers in r.items():
            print(f"  [JBB extraction = {tag}]")
            print(f"    {'L':>3}  {'cic→harm':>9}  {'uns→harm':>9}  {'random':>8}")
            for L in sorted(layers, key=int):
                v = layers[L]
                print(f"    {L:>3}  {v['auroc_cic_dir_on_harmful']:>9.3f}  "
                      f"{v['auroc_uns_dir_on_harmful']:>9.3f}  "
                      f"{v['auroc_random_on_harmful']:>8.3f}")
    json.dump(big, open(OUT / "phase2_b5_cic_dir_on_harmful.json", "w"), indent=2)
    print("\nsaved -> phase2_b5_cic_dir_on_harmful.json")


if __name__ == "__main__":
    main()
