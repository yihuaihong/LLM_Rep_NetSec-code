"""Cross-dataset direction transfer on Qwen3-8B-Base (raw, no chat template).

For each layer L:
  d_cic_L = mean(cicids attack) - mean(cicids normal)  on Qwen3-Base train reps
  d_uns_L = mean(unsw   attack) - mean(unsw   normal)  on Qwen3-Base train reps

Then test cross-projection AUROC:
  cic→uns: project Qwen3-Base unsw train onto d_cic_L; AUROC vs unsw labels
  uns→cic: project Qwen3-Base cicids train onto d_uns_L; AUROC vs cicids labels

Compare with within-dataset AUROC (which should be near 1.0).

Compare with the same numbers on Qwen3-Instruct (chat-template) — those were
cic→uns AUROC = 0.49 (chance) at best layer 18.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from utils.probe_utils import extract_separation_direction, project_onto_direction

BASE = Path(__file__).resolve().parent.parent
REPS = BASE / "results" / "representations"
OUT = BASE / "results" / "metrics"

MODELS = ["Qwen3-8B-Base", "Qwen3-8B"]


def to_f32(t):
    return t.float() if t.dtype == torch.bfloat16 else t


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(
        a.float().unsqueeze(0), b.float().unsqueeze(0)
    ).item())


def auroc(y, scores):
    return float(roc_auc_score(y, scores))


def run(model):
    cic = torch.load(REPS / "cicids2017" / model / "train.pt", weights_only=False)
    uns = torch.load(REPS / "unsw_nb15"  / model / "train.pt", weights_only=False)
    layers = sorted(set(cic["hidden_states"].keys()) & set(uns["hidden_states"].keys()))
    y_cic = cic["labels"].numpy()
    y_uns = uns["labels"].numpy()

    print(f"\n=== {model} ===")
    print(f"  layers: {layers}")
    print(f"  {'L':>3}  "
          f"{'within_cic':>10}  {'within_uns':>10}  "
          f"{'cic_dir on uns':>15}  {'uns_dir on cic':>15}  "
          f"{'cos(d_cic,d_uns)':>17}")

    rec = {}
    for L in layers:
        H_c = to_f32(cic["hidden_states"][L])
        H_u = to_f32(uns["hidden_states"][L])
        d_c = extract_separation_direction(H_c, y_cic)
        d_u = extract_separation_direction(H_u, y_uns)

        # Within-dataset AUROC (should be near 1.0)
        proj_cc = project_onto_direction(H_c, d_c)
        proj_uu = project_onto_direction(H_u, d_u)
        auroc_within_c = auroc(y_cic, proj_cc)
        auroc_within_u = auroc(y_uns, proj_uu)

        # Cross-dataset
        proj_uc = project_onto_direction(H_u, d_c)   # cicids dir on unsw reps
        proj_cu = project_onto_direction(H_c, d_u)   # unsw dir on cicids reps
        auroc_cic_to_uns = auroc(y_uns, proj_uc)
        auroc_uns_to_cic = auroc(y_cic, proj_cu)

        c = cos(d_c, d_u)
        rec[L] = {
            "within_cic_auroc": auroc_within_c,
            "within_uns_auroc": auroc_within_u,
            "cic_to_uns_auroc": auroc_cic_to_uns,
            "uns_to_cic_auroc": auroc_uns_to_cic,
            "cos_d_cic_d_uns": c,
        }
        print(f"  {L:>3}  {auroc_within_c:>10.3f}  {auroc_within_u:>10.3f}  "
              f"{auroc_cic_to_uns:>15.3f}  {auroc_uns_to_cic:>15.3f}  "
              f"{c:>+17.4f}")

    return rec


def main():
    out = {}
    for m in MODELS:
        try:
            out[m] = run(m)
        except FileNotFoundError as e:
            print(f"  [{m}] missing reps: {e}")
    json.dump(out, open(OUT / "phase1_qwen3_base_cross_dataset.json", "w"), indent=2)
    print("\nsaved -> phase1_qwen3_base_cross_dataset.json")


if __name__ == "__main__":
    main()
