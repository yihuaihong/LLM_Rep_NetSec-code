"""Exp 6 with RAW (no chat template) JBB extraction across 4 instruct models.

For each model, compute:
  cos(harmful_dir_chat, cic/uns_attack_dir)
  cos(harmful_dir_raw,  cic/uns_attack_dir)

If raw shows much larger |cos| than chat-template version, then the chat
template was hiding (or flipping) a real shared "danger" axis that exists in
the model's pretrained representation.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pathlib import Path
from utils.probe_utils import extract_separation_direction

BASE = Path(__file__).resolve().parent.parent
REPS = BASE / "results" / "representations"
OUT = BASE / "results" / "metrics"

MODELS = [
    "Llama-3.1-8B-Instruct",
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


def run(model):
    cic = torch.load(REPS / "cicids2017" / model / "train.pt", weights_only=False)
    uns = torch.load(REPS / "unsw_nb15"  / model / "train.pt", weights_only=False)
    jbb_chat = torch.load(REPS / "jbb" / model / "all.pt",     weights_only=False)
    jbb_raw  = torch.load(REPS / "jbb" / model / "all_raw.pt", weights_only=False)

    layers = sorted(set(cic["hidden_states"].keys())
                    & set(uns["hidden_states"].keys())
                    & set(jbb_chat["hidden_states"].keys())
                    & set(jbb_raw["hidden_states"].keys()))

    print(f"\n=== {model} ===")
    print(f"  shared layers: {layers}")
    hdr = "  " + "L".rjust(3) + "  "
    for tag in ["chat→cic", "raw→cic", "chat→uns", "raw→uns"]:
        hdr += tag.rjust(11) + "  "
    print(hdr)
    print("  " + "-" * 68)

    out = {}
    for L in layers:
        d_cic = extract_separation_direction(to_f32(cic["hidden_states"][L]), cic["labels"].numpy())
        d_uns = extract_separation_direction(to_f32(uns["hidden_states"][L]), uns["labels"].numpy())
        d_h_chat = extract_separation_direction(to_f32(jbb_chat["hidden_states"][L]), jbb_chat["labels"].numpy())
        d_h_raw  = extract_separation_direction(to_f32(jbb_raw["hidden_states"][L]),  jbb_raw["labels"].numpy())

        c_chat_cic = cos(d_h_chat, d_cic)
        c_raw_cic  = cos(d_h_raw,  d_cic)
        c_chat_uns = cos(d_h_chat, d_uns)
        c_raw_uns  = cos(d_h_raw,  d_uns)
        out[L] = {
            "cos_chat_cic": c_chat_cic, "cos_raw_cic": c_raw_cic,
            "cos_chat_uns": c_chat_uns, "cos_raw_uns": c_raw_uns,
        }
        row = f"  {L:>3}  "
        for v in [c_chat_cic, c_raw_cic, c_chat_uns, c_raw_uns]:
            row += f"{v:>+11.4f}  "
        print(row)
    return out


def main():
    out = {}
    for m in MODELS:
        try:
            out[m] = run(m)
        except FileNotFoundError as e:
            print(f"  [{m}] missing reps: {e}")
    json.dump(out, open(OUT / "phase2_raw_alignment_4models.json", "w"), indent=2)
    print("\nsaved -> phase2_raw_alignment_4models.json")


if __name__ == "__main__":
    main()
