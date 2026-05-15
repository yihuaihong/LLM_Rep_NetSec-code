"""Exp 5: activation steering with the attack direction.

Procedure:
1. Build a yes/no Q&A prompt:
     "You are a network security analyst. Is this connection malicious?
      Answer with only 'yes' or 'no'.
      Log: <log>
      Answer:"

2. For a set of holdout samples (mix of normal + each holdout type), run the
   model and look at logit("yes") - logit("no") at the next-token position.
   This gives the model's RAW "attack belief" without any probe.

3. Add α * direction at every transformer layer (residual stream addition is
   the standard steering hook). Re-run, recompute the logit gap.

4. Compare 3 directions:
     - attack_direction (CICIDS train mean-diff)
     - harmful_direction (JBB)
     - random_direction (control)

   For each, sweep α in {-5, -2, 0, +2, +5}.

   - If attack_direction PUSHES yes/no monotonically → model uses this axis
     for actual output behavior. Direction is "real" for the model.
   - If only the harmful_direction pushes "no" (refusal) → the netsec direction
     and the harmful direction are functionally distinct.

We use one model per run (--model). Default: Qwen3-8B (the interesting case).
"""
import os, sys, json, gc, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path

from utils.model_utils import load_model_and_tokenizer, get_layer_count, get_layer_module
from utils.probe_utils import extract_separation_direction

BASE = Path(__file__).resolve().parent.parent
REPS = BASE / "results" / "representations"
OUT = BASE / "results" / "metrics"
CACHE_DIR = "/scratch/yh6210/transformers"


def to_f32(t):
    return t.float() if t.dtype == torch.bfloat16 else t


PROMPT_TEMPLATE = (
    "You are a network security analyst. Below is a network connection log.\n"
    "Decide if this connection is benign or malicious.\n"
    "Answer with only one word: \"yes\" if malicious, \"no\" if benign.\n\n"
    "{log}\n\n"
    "Answer:"
)


def make_prompt(log, tokenizer):
    user = PROMPT_TEMPLATE.format(log=log)
    msgs = [{"role": "user", "content": user}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def get_yes_no_token_ids(tokenizer):
    """Return likely token ids for 'yes' and 'no'."""
    yes_ids = []
    no_ids = []
    for t in [" yes", "yes", " Yes", "Yes", " YES", "YES"]:
        ids = tokenizer.encode(t, add_special_tokens=False)
        if len(ids) == 1:
            yes_ids.append(ids[0])
    for t in [" no", "no", " No", "No", " NO", "NO"]:
        ids = tokenizer.encode(t, add_special_tokens=False)
        if len(ids) == 1:
            no_ids.append(ids[0])
    return list(set(yes_ids)), list(set(no_ids))


def get_directions(model_name, dataset_name):
    """Return (attack_dir, harmful_dir, random_dir) at the model's best layer."""
    metrics = json.load(open(OUT / f"{model_name}_{dataset_name}.json"))
    bl = int(metrics["best_layer"])

    netsec = torch.load(REPS / dataset_name / model_name / "train.pt", weights_only=False)
    attack_dir = extract_separation_direction(
        to_f32(netsec["hidden_states"][bl]), netsec["labels"].numpy()
    )

    jbb_path = REPS / "jbb" / model_name / "all.pt"
    harmful_dir = None
    if jbb_path.exists():
        jbb = torch.load(jbb_path, weights_only=False)
        if bl in jbb["hidden_states"]:
            harmful_dir = extract_separation_direction(
                to_f32(jbb["hidden_states"][bl]), jbb["labels"].numpy()
            )

    g = torch.Generator().manual_seed(123)
    random_dir = torch.randn(attack_dir.shape, generator=g, dtype=torch.float32)
    random_dir = random_dir / random_dir.norm()
    return bl, attack_dir, harmful_dir, random_dir


@torch.no_grad()
def logit_yes_minus_no(model, tokenizer, prompts, yes_ids, no_ids, batch_size=4):
    """For each prompt, compute log p(yes) - log p(no) at the next token."""
    diffs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        toks = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        toks = {k: v.cuda() for k, v in toks.items()}
        out = model(**toks)
        # last non-pad token position per sequence
        last_idx = toks["attention_mask"].sum(dim=1) - 1
        logits = out.logits[torch.arange(len(batch)), last_idx]   # (B, V)
        log_p = torch.log_softmax(logits.float(), dim=-1)
        log_yes = torch.logsumexp(log_p[:, yes_ids], dim=-1)
        log_no  = torch.logsumexp(log_p[:, no_ids], dim=-1)
        diffs.extend((log_yes - log_no).cpu().tolist())
    return diffs


def install_steer_hook(model, alpha, direction, target_layer):
    """Add alpha * direction to ONE specific layer's residual stream output.

    This is the standard "activation addition" / CAA-style intervention.
    """
    direction = direction.cuda()
    layer = get_layer_module(model, target_layer)

    def hook(module, args, kwargs, output):
        if isinstance(output, tuple):
            hs = output[0]
            hs = hs + alpha * direction.to(hs.dtype)
            return (hs,) + output[1:]
        else:
            return output + alpha * direction.to(output.dtype)

    handle = layer.register_forward_hook(hook, with_kwargs=True)
    return [handle]


def remove_hooks(handles):
    for h in handles:
        h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-8B")
    ap.add_argument("--dataset", default="cicids2017")
    ap.add_argument("--n_per_type", type=int, default=20,
                    help="number of holdout samples per attack type")
    ap.add_argument("--alphas", nargs="+", type=float, default=[-5.0, -2.0, 0.0, 2.0, 5.0])
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    dtype = "bfloat16" if "gemma" in args.model.lower() else "float16"
    model, tokenizer = load_model_and_tokenizer(
        f"{CACHE_DIR}/{args.model}", cache_dir=CACHE_DIR, dtype=dtype, device="cuda",
    )
    yes_ids, no_ids = get_yes_no_token_ids(tokenizer)
    print(f"  yes token ids: {yes_ids}")
    print(f"  no  token ids: {no_ids}")
    if not yes_ids or not no_ids:
        raise RuntimeError("Could not find single-token yes/no in tokenizer; need fallback.")

    # Load directions
    bl, attack_dir, harmful_dir, random_dir = get_directions(args.model, args.dataset)
    print(f"  best_layer = {bl}")
    print(f"  attack_dir norm = {attack_dir.norm():.4f}")
    if harmful_dir is not None:
        print(f"  harmful_dir norm = {harmful_dir.norm():.4f}")

    # Pick samples — mix of holdout types (incl. Normal) — use raw text
    # We don't have raw text in cached reps, so reconstruct from parquet
    import pandas as pd
    from utils.data_utils import load_dataset, format_dataset, create_generalization_splits
    DATASET_PATHS = {
        "cicids2017": "/scratch/yh6210/datasets/LLM_Rep_NetSec_datasets/cicids2017/",
        "unsw_nb15":  "/scratch/yh6210/datasets/LLM_Rep_NetSec_datasets/unsw_nb15/",
    }
    HOLDOUT = {
        "cicids2017": ["Heartbleed", "Infiltration", "Bot"],
        "unsw_nb15":  ["Shellcode", "Worms", "Backdoor"],
    }
    df = load_dataset(args.dataset, DATASET_PATHS[args.dataset])
    df = format_dataset(df, args.dataset, fmt="natural_language")
    splits = create_generalization_splits(
        df, holdout_attack_types=HOLDOUT[args.dataset],
        max_samples_per_class=5000, seed=42,
    )
    holdout_df = splits["test_holdout"]
    samples = []
    for t in sorted(holdout_df["attack_type"].unique()):
        sub = holdout_df[holdout_df["attack_type"] == t]
        n = min(args.n_per_type, len(sub))
        sub = sub.sample(n=n, random_state=0)
        for _, row in sub.iterrows():
            samples.append({"text": row["text"], "type": t, "is_attack": int(row["is_attack"])})
    print(f"  total samples = {len(samples)}")

    prompts = [make_prompt(s["text"], tokenizer) for s in samples]

    # Sweep — for each direction, for each alpha, run forward and collect yes/no diff
    results = {"model": args.model, "dataset": args.dataset, "best_layer": bl,
               "alphas": args.alphas, "samples": samples,
               "directions": {}}

    dir_dict = {"attack": attack_dir, "random": random_dir}
    if harmful_dir is not None:
        dir_dict["harmful"] = harmful_dir

    for d_name, d_vec in dir_dict.items():
        print(f"\n=== Steering with {d_name}_direction ===")
        per_alpha = {}
        for alpha in args.alphas:
            handles = install_steer_hook(model, alpha, d_vec, target_layer=bl) if alpha != 0.0 else []
            try:
                diffs = logit_yes_minus_no(model, tokenizer, prompts, yes_ids, no_ids, batch_size=4)
            finally:
                remove_hooks(handles)
            per_alpha[float(alpha)] = diffs
            # Aggregate per type
            types = [s["type"] for s in samples]
            arr = np.array(diffs)
            print(f"  α={alpha:+.1f}  mean={arr.mean():+.3f}  ", end="")
            for t in sorted(set(types)):
                m = arr[np.array(types) == t]
                print(f"{t}={m.mean():+.2f} ", end="")
            print()
        results["directions"][d_name] = per_alpha

    out_path = OUT / f"phase1_exp5_steering_{args.model}_{args.dataset}.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
