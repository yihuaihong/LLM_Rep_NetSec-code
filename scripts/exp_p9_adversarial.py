"""P9: adversarial perturbation — minimum text-edit to flip probe prediction.

Take a holdout sample S that the probe correctly classifies as attack.
For each numeric field f in the input, sweep f's value through a range and
track when the probe flips to "normal". The smallest |Δf| (in standardized
units) needed for a flip = how brittle the probe is.

Hypothesis: if probe direction is a lexical hash, flipping a single field
flips the prediction with very small text edit. If the direction reflects
a real concept, flipping should require multiple coherent edits.

Run on Qwen3-8B-Base, cicids2017 holdout (Bot, Heartbleed, Infiltration).
We perturb 5 representative numeric fields per sample and measure the
flip threshold per field.
"""
import os, sys, json, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from pathlib import Path

from utils.model_utils import load_model_and_tokenizer, set_random_seed, flush, extract_hidden_states
from utils.data_utils import load_dataset, create_generalization_splits, format_log_natural_language

BASE = Path(__file__).resolve().parent.parent
REPS = BASE / "results" / "representations"
OUT = BASE / "results" / "metrics"
CACHE_DIR = "/scratch/yh6210/transformers"

DATASET_PATHS = {
    "cicids2017": "/scratch/yh6210/datasets/LLM_Rep_NetSec_datasets/cicids2017/",
}
HOLDOUT = ["Heartbleed", "Infiltration", "Bot"]


def to_f32(t):
    return t.float() if t.dtype in (torch.bfloat16, torch.float16) else t


def get_probe_direction(model, layer):
    """Use mean-diff direction (consistent with paper)."""
    train = torch.load(REPS / "cicids2017" / model / "train.pt", weights_only=False)
    H = to_f32(train["hidden_states"][layer])
    y = train["labels"].numpy()
    mu_a = H[y == 1].mean(0)
    mu_n = H[y == 0].mean(0)
    threshold = ((mu_a + mu_n) / 2 @ (mu_a - mu_n)) / (mu_a - mu_n).norm()
    d = (mu_a - mu_n) / (mu_a - mu_n).norm()
    # threshold for projection: midpoint of class means projected onto d
    # proj_normal_mean = mu_n @ d, proj_attack_mean = mu_a @ d
    proj_n = (mu_n @ d).item()
    proj_a = (mu_a @ d).item()
    thr = (proj_n + proj_a) / 2
    return d, thr, proj_n, proj_a


def sample_holdout(model_name, n_per_type=5, seed=0):
    """Get text representations of holdout samples."""
    df = load_dataset("cicids2017", DATASET_PATHS["cicids2017"])
    splits = create_generalization_splits(df, holdout_attack_types=HOLDOUT,
                                          max_samples_per_class=5000, seed=42)
    holdout = splits["test_holdout"]
    rng = np.random.RandomState(seed)
    rows = []
    for t in HOLDOUT + ["Normal"]:
        sub = holdout[holdout.attack_type == t]
        if len(sub) == 0: continue
        sub = sub.sample(min(len(sub), n_per_type), random_state=rng)
        for _, r in sub.iterrows():
            rows.append(r)
    return pd.DataFrame(rows)


PERTURB_FIELDS = [
    "Init_Win_bytes_forward",
    "destination_port",
    "PSH Flag Count",
    "Flow Bytes/s",
    "Flow Duration",
]


def perturb_value(row, field, factor):
    """Return (text, perturbed_row). factor multiplies the field value."""
    new_row = row.copy()
    if field in new_row.index and pd.notna(new_row[field]):
        new_row[field] = new_row[field] * factor
    text = format_log_natural_language(new_row, "cicids2017")
    return text


@torch.no_grad()
def project_one(model, tokenizer, text, layer, direction, max_len=512):
    toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to("cuda")
    out = model(**toks, output_hidden_states=True)
    h = out.hidden_states[layer][0, -1].float().cpu()
    return float((h @ direction).item())


def main():
    model_name = "Qwen3-8B-Base"
    set_random_seed(42)
    print(f"Loading {model_name} ...")
    model, tokenizer = load_model_and_tokenizer(
        f"{CACHE_DIR}/{model_name}", cache_dir=CACHE_DIR, dtype="float16", device="cuda")

    # Best layer for cicids2017 + Qwen3-Base
    metrics = json.load(open(OUT / f"{model_name}_cicids2017.json"))
    L = int(metrics["best_layer"])
    print(f"  best layer = {L}")
    direction, thr, proj_n_mean, proj_a_mean = get_probe_direction(model_name, L)
    direction = direction.float()
    print(f"  decision boundary at proj = {thr:.3f}")
    print(f"  mean(normal) proj = {proj_n_mean:.3f}, mean(attack) proj = {proj_a_mean:.3f}")

    # Sample holdout
    samples_df = sample_holdout(model_name, n_per_type=4)
    samples_df = samples_df.reset_index(drop=True)
    print(f"  {len(samples_df)} holdout samples loaded")

    factors = [0.0, 0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 100.0]

    out = {"model": model_name, "layer": L, "threshold": thr, "fields": PERTURB_FIELDS,
           "factors": factors, "samples": []}

    for i, row in samples_df.iterrows():
        # Original projection
        text_orig = format_log_natural_language(row, "cicids2017")
        proj_orig = project_one(model, tokenizer, text_orig, L, direction)
        sample_rec = {
            "type": row["attack_type"], "true_label": int(row["is_attack"]),
            "orig_proj": proj_orig,
            "orig_classification": "attack" if proj_orig > thr else "normal",
            "perturbations": {},
        }
        print(f"\n[{i+1}/{len(samples_df)}] type={row['attack_type']}  "
              f"orig_proj={proj_orig:+.2f} ({sample_rec['orig_classification']})")
        for field in PERTURB_FIELDS:
            if field not in row.index or pd.isna(row[field]):
                continue
            orig_val = float(row[field])
            curve = []
            for f in factors:
                text = perturb_value(row, field, f)
                p = project_one(model, tokenizer, text, L, direction)
                curve.append({"factor": f, "new_val": orig_val * f, "proj": p,
                              "classification": "attack" if p > thr else "normal"})
            sample_rec["perturbations"][field] = {"orig_val": orig_val, "curve": curve}
            # Did any factor flip the classification?
            orig_class = sample_rec["orig_classification"]
            flipped = [c for c in curve if c["classification"] != orig_class]
            if flipped:
                # Find smallest |log(factor)| flip
                flipped_sorted = sorted(flipped, key=lambda c: abs(np.log(max(c["factor"], 1e-10))))
                first_flip = flipped_sorted[0]
                print(f"    {field:25s} orig={orig_val:.2g}  → flips at factor={first_flip['factor']:.4g} "
                      f"(new={first_flip['new_val']:.2g}, proj={first_flip['proj']:+.2f})")
            else:
                print(f"    {field:25s} orig={orig_val:.2g}  → no flip across factors")
        out["samples"].append(sample_rec)

    json.dump(out, open(OUT / "phase2_p9_adversarial.json", "w"), indent=2)
    print(f"\nsaved -> phase2_p9_adversarial.json")

    del model, tokenizer; flush()


if __name__ == "__main__":
    main()
