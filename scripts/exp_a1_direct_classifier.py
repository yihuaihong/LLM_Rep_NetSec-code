"""A1: LLM as direct classifier (no probe).

Ask the model directly: "Is this connection malicious? Answer (yes/no):"
Use log p(yes) - log p(no) as score; AUROC against true labels.

If model is high AUROC zero-shot:
    -> probe ceiling exists; our negative result is "probe doesn't extract it"
If model is chance AUROC zero-shot:
    -> model truly doesn't have the concept; supports our V3 finding

Datasets:
  cic2017 holdout, unsw_nb15 holdout, cicids2018, iot23, ctu13.
For each, sample up to 1000 mixed (normal + attack).
"""
import argparse, os, sys, json, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from pathlib import Path

from utils.model_utils import load_model_and_tokenizer, set_random_seed, flush
from utils.data_utils import (
    load_dataset, format_dataset, create_generalization_splits
)

BASE = Path(__file__).resolve().parent.parent
REPS = BASE / "results" / "representations"
OUT  = BASE / "results" / "metrics"
CACHE_DIR = "/scratch/yh6210/transformers"

DATASET_PATHS = {
    "cicids2017": "/scratch/yh6210/datasets/LLM_Rep_NetSec_datasets/cicids2017/",
    "unsw_nb15":  "/scratch/yh6210/datasets/LLM_Rep_NetSec_datasets/unsw_nb15/",
}
HOLDOUT = {
    "cicids2017": ["Heartbleed", "Infiltration", "Bot"],
    "unsw_nb15":  ["Shellcode", "Worms", "Backdoor"],
}

INSTRUCT_PROMPT = (
    "You are a network security analyst. Below is a network connection log.\n"
    "Decide if this connection is benign or malicious.\n"
    "Answer with only one word: \"yes\" if malicious, \"no\" if benign.\n\n"
    "{log}\n\n"
    "Answer:"
)
BASE_PROMPT = (
    "Network connection logs can be benign or malicious.\n"
    "{log}\n"
    "Question: Is this connection malicious? Answer (yes/no):"
)


def make_prompt(log, tokenizer, is_instruct):
    if is_instruct:
        msgs = [{"role": "user", "content": INSTRUCT_PROMPT.format(log=log)}]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return BASE_PROMPT.format(log=log)


def get_yes_no_token_ids(tokenizer):
    yes_ids, no_ids = [], []
    for t in [" yes", "yes", " Yes", "Yes"]:
        ids = tokenizer.encode(t, add_special_tokens=False)
        if len(ids) == 1: yes_ids.append(ids[0])
    for t in [" no", "no", " No", "No"]:
        ids = tokenizer.encode(t, add_special_tokens=False)
        if len(ids) == 1: no_ids.append(ids[0])
    return list(set(yes_ids)), list(set(no_ids))


@torch.no_grad()
def score_prompts(model, tokenizer, prompts, yes_ids, no_ids, batch_size=8):
    diffs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        toks = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        toks = {k: v.cuda() for k, v in toks.items()}
        out = model(**toks)
        last_idx = toks["attention_mask"].sum(dim=1) - 1
        logits = out.logits[torch.arange(len(batch)), last_idx]
        log_p = torch.log_softmax(logits.float(), dim=-1)
        log_yes = torch.logsumexp(log_p[:, yes_ids], dim=-1)
        log_no  = torch.logsumexp(log_p[:, no_ids],  dim=-1)
        diffs.extend((log_yes - log_no).cpu().tolist())
    return diffs


def get_dataset_samples(name, n_per_class=500, source_priority=("zero_shot", "splits")):
    """Return list of {text, is_attack, attack_type}. Tries multiple sources."""
    # First try zero_shot.pt (already-formatted texts) — for new datasets
    for m_dir in (REPS / name).glob("*"):
        zs = m_dir / "zero_shot.pt"
        if zs.exists() and "zero_shot" in source_priority:
            d = torch.load(zs, weights_only=False)
            if "texts" in d:
                texts = d["texts"]
                labels = d["labels"].numpy()
                types = list(d["attack_types"])
                rows = list(zip(texts, labels, types))
                # Stratify by class
                rng = np.random.RandomState(0)
                normals = [r for r in rows if r[1] == 0]
                attacks = [r for r in rows if r[1] == 1]
                rng.shuffle(normals); rng.shuffle(attacks)
                normals = normals[:n_per_class]
                attacks = attacks[:n_per_class]
                return [{"text": t, "is_attack": int(y), "attack_type": at}
                        for t, y, at in normals + attacks]
    # Fall back to load_dataset for cic2017/unsw
    if name in DATASET_PATHS and "splits" in source_priority:
        df = load_dataset(name, DATASET_PATHS[name])
        df = format_dataset(df, name, fmt="natural_language")
        splits = create_generalization_splits(
            df, holdout_attack_types=HOLDOUT[name],
            max_samples_per_class=5000, seed=42,
        )
        # For cic2017/unsw: combine known-test + holdout-test (both are "test"; we want a representative sample)
        out = []
        rng = np.random.RandomState(1)
        for tag in ["test_known", "test_holdout"]:
            sub = splits[tag]
            normals = sub[sub.is_attack == 0]
            attacks = sub[sub.is_attack == 1]
            normals = normals.sample(min(len(normals), n_per_class // 2), random_state=rng)
            attacks = attacks.sample(min(len(attacks), n_per_class // 2), random_state=rng)
            for _, row in pd.concat([normals, attacks]).iterrows():
                out.append({"text": row["text"], "is_attack": int(row["is_attack"]),
                            "attack_type": row["attack_type"], "split": tag})
        return out
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--datasets", nargs="+",
                    default=["cicids2017", "unsw_nb15", "cicids2018", "iot23", "ctu13"])
    ap.add_argument("--n_per_class", type=int, default=500)
    args = ap.parse_args()

    set_random_seed(42)
    is_instruct = "Base" not in args.model

    # Build prompts before loading the model
    print(f"\n[A1] Building prompts for {args.model} (instruct={is_instruct})")
    dataset_samples = {}
    for ds in args.datasets:
        samples = get_dataset_samples(ds, n_per_class=args.n_per_class)
        if not samples:
            print(f"  [{ds}] no samples found, skip")
            continue
        n_atk = sum(1 for s in samples if s["is_attack"] == 1)
        print(f"  [{ds}] {len(samples)} samples (atk={n_atk})")
        dataset_samples[ds] = samples

    # Load model
    print(f"\nloading {args.model} ...")
    dtype = "bfloat16" if "gemma" in args.model.lower() else "float16"
    model, tokenizer = load_model_and_tokenizer(
        f"{CACHE_DIR}/{args.model}", cache_dir=CACHE_DIR, dtype=dtype, device="cuda",
    )
    yes_ids, no_ids = get_yes_no_token_ids(tokenizer)
    print(f"  yes_ids={yes_ids}  no_ids={no_ids}")

    # Run
    from sklearn.metrics import roc_auc_score, accuracy_score
    big = {"model": args.model, "is_instruct": is_instruct, "datasets": {}}
    for ds, samples in dataset_samples.items():
        prompts = [make_prompt(s["text"], tokenizer, is_instruct) for s in samples]
        print(f"\nscoring {ds} ({len(prompts)} prompts) ...")
        diffs = score_prompts(model, tokenizer, prompts, yes_ids, no_ids)
        diffs_np = np.array(diffs)
        labels = np.array([s["is_attack"] for s in samples])
        auc = float(roc_auc_score(labels, diffs_np))
        # Accuracy: positive logit-diff -> "yes"
        pred = (diffs_np > 0).astype(int)
        acc = float(accuracy_score(labels, pred))
        # Frac of "yes"
        frac_yes = float((diffs_np > 0).mean())
        big["datasets"][ds] = {
            "n": len(samples),
            "auroc": auc,
            "accuracy_at_zero": acc,
            "frac_yes_at_zero": frac_yes,
            "mean_diff": float(diffs_np.mean()),
            "n_attack": int((labels == 1).sum()),
            # Per-attack-type breakdown
            "per_type": {},
        }
        types = np.array([s["attack_type"] for s in samples])
        for t in np.unique(types):
            mask = types == t
            arr = diffs_np[mask]
            big["datasets"][ds]["per_type"][t] = {
                "n": int(mask.sum()),
                "true_label": int(labels[mask][0] == 1) if mask.sum() > 0 else None,
                "mean_logit_diff": float(arr.mean()),
                "frac_yes": float((arr > 0).mean()),
            }
        print(f"  AUROC={auc:.3f}  acc={acc:.3f}  frac_yes={frac_yes:.3f}  mean_diff={diffs_np.mean():+.3f}")
        for t, v in big["datasets"][ds]["per_type"].items():
            print(f"    {t:30s} n={v['n']:>4} mean_diff={v['mean_logit_diff']:+.2f}  frac_yes={v['frac_yes']:.2f}")

    out_path = OUT / f"phase2_a1_direct_classifier_{args.model}.json"
    json.dump(big, open(out_path, "w"), indent=2)
    print(f"\nsaved -> {out_path}")
    del model, tokenizer; flush()


if __name__ == "__main__":
    main()
