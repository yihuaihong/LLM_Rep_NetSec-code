"""Steering v2 — careful hyperparameter sweep on Qwen3-8B and Qwen3-8B-Base.

Improvements over v1:
1. Sweep the LAYER at which to add direction (mid layers often work better
   than late ones for behavioral effects).
2. Use ABSOLUTE alpha grounded in residual stream norm (typical hidden-state
   norm ~10-100 for 8B models). v1 used α∈{-5..5} on unit direction, which
   is way smaller than the natural hidden norm. Try α ∈ {0, ±10, ±20, ±50}.
3. For Qwen3-Base: use a completion-style prompt, not chat template.
4. Compare 3 direction sources (attack from cicids, attack from unsw, harmful
   from JBB raw) and a control (random + 0).
5. Report effect at the FIRST 'is malicious yes/no' decision token AND at the
   second token (in case the answer takes 2 tokens to land).
6. Multi-layer steering (CAA-style: add at every layer ≥ start_layer).
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


# Two prompt formats — instruct uses chat template, base uses raw completion.
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
    yes_tokens = [" yes", "yes", " Yes", "Yes"]
    no_tokens  = [" no",  "no",  " No",  "No"]
    yes_ids, no_ids = [], []
    for t in yes_tokens:
        ids = tokenizer.encode(t, add_special_tokens=False)
        if len(ids) == 1:
            yes_ids.append(ids[0])
    for t in no_tokens:
        ids = tokenizer.encode(t, add_special_tokens=False)
        if len(ids) == 1:
            no_ids.append(ids[0])
    return list(set(yes_ids)), list(set(no_ids))


@torch.no_grad()
def get_residual_norm(model, tokenizer, prompt, layer):
    """Sample residual stream norm at given layer for one prompt."""
    norm_box = {"v": None}

    layer_mod = get_layer_module(model, layer)
    def hook(module, args, kwargs, output):
        hs = output[0] if isinstance(output, tuple) else output
        # last token of last sequence
        norm_box["v"] = float(hs[0, -1].float().norm().item())
    h = layer_mod.register_forward_hook(hook, with_kwargs=True)
    toks = tokenizer(prompt, return_tensors="pt").to("cuda")
    model(**toks)
    h.remove()
    return norm_box["v"]


@torch.no_grad()
def yes_minus_no_logit(model, tokenizer, prompts, yes_ids, no_ids, batch_size=4):
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
        log_no  = torch.logsumexp(log_p[:, no_ids], dim=-1)
        diffs.extend((log_yes - log_no).cpu().tolist())
    return diffs


def install_steer_hook_single(model, alpha, direction, target_layer):
    direction = direction.cuda()
    layer = get_layer_module(model, target_layer)

    def hook(module, args, kwargs, output):
        if isinstance(output, tuple):
            hs = output[0]
            hs = hs + alpha * direction.to(hs.dtype)
            return (hs,) + output[1:]
        else:
            return output + alpha * direction.to(output.dtype)

    return [layer.register_forward_hook(hook, with_kwargs=True)]


def install_steer_hook_multi(model, alpha, direction, start_layer):
    """Add α*direction at every layer ≥ start_layer (CAA-style)."""
    direction = direction.cuda()
    n_layers = get_layer_count(model)
    handles = []
    for i in range(start_layer, n_layers):
        layer = get_layer_module(model, i)
        def make_hook(_):
            def hook(module, args, kwargs, output):
                if isinstance(output, tuple):
                    hs = output[0]
                    hs = hs + alpha * direction.to(hs.dtype)
                    return (hs,) + output[1:]
                else:
                    return output + alpha * direction.to(output.dtype)
            return hook
        handles.append(layer.register_forward_hook(make_hook(i), with_kwargs=True))
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


def get_directions(model_name, jbb_filename):
    """Return dict layer -> direction tensors for cic_attack, uns_attack, harmful (raw)."""
    cic = torch.load(REPS / "cicids2017" / model_name / "train.pt", weights_only=False)
    uns = torch.load(REPS / "unsw_nb15"  / model_name / "train.pt", weights_only=False)
    jbb_path = REPS / "jbb" / model_name / jbb_filename
    jbb = None
    if jbb_path.exists():
        jbb = torch.load(jbb_path, weights_only=False)

    layers = sorted(cic["hidden_states"].keys())
    out = {}
    for L in layers:
        d_cic = extract_separation_direction(to_f32(cic["hidden_states"][L]), cic["labels"].numpy())
        d_uns = extract_separation_direction(to_f32(uns["hidden_states"][L]), uns["labels"].numpy())
        d_harm = None
        if jbb is not None and L in jbb["hidden_states"]:
            d_harm = extract_separation_direction(to_f32(jbb["hidden_states"][L]), jbb["labels"].numpy())
        out[L] = {"cic_attack": d_cic, "uns_attack": d_uns, "harmful": d_harm}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-8B-Base")
    ap.add_argument("--dataset", default="cicids2017")
    ap.add_argument("--jbb_file", default=None,
                    help="Override JBB filename. Default: all_raw.pt for *-Base, all.pt otherwise")
    ap.add_argument("--n_per_type", type=int, default=15)
    ap.add_argument("--target_layers", nargs="+", type=int, default=None,
                    help="Layers to test as steering insertion point. Default: 5 layers spread across model.")
    ap.add_argument("--alphas", nargs="+", type=float, default=[-5.0, -2.0, 0.0, 2.0, 5.0])
    ap.add_argument("--multi_layer", action="store_true",
                    help="If set, steer at all layers >= target_layer (CAA-style).")
    ap.add_argument("--alpha_relative", action="store_true",
                    help="If set, multiply alphas by mean residual norm at target layer.")
    ap.add_argument("--all_layers", action="store_true",
                    help="If set, sweep ALL cached layers (overrides --target_layers).")
    args = ap.parse_args()

    is_instruct = "Base" not in args.model
    if args.jbb_file is None:
        args.jbb_file = "all.pt" if is_instruct else "all_raw.pt"

    print(f"Loading {args.model}  (instruct={is_instruct})  jbb={args.jbb_file}")
    dtype = "bfloat16" if "gemma" in args.model.lower() else "float16"
    model, tokenizer = load_model_and_tokenizer(
        f"{CACHE_DIR}/{args.model}", cache_dir=CACHE_DIR, dtype=dtype, device="cuda",
    )
    n_layers = get_layer_count(model)
    print(f"  n_layers = {n_layers}")
    yes_ids, no_ids = get_yes_no_token_ids(tokenizer)
    print(f"  yes ids ({len(yes_ids)}): {yes_ids}")
    print(f"  no  ids ({len(no_ids)}): {no_ids}")

    # Directions
    dir_dict = get_directions(args.model, args.jbb_file)
    cached_layers = sorted(dir_dict.keys())
    print(f"  cached layers: {cached_layers}")
    if args.all_layers:
        args.target_layers = list(cached_layers)
    elif args.target_layers is None:
        # Pick 5 layers across cached layers: 25%, 40%, 50%, 70%, 90%
        idx = [int(round(p * (len(cached_layers) - 1))) for p in [0.25, 0.40, 0.50, 0.70, 0.90]]
        args.target_layers = sorted(set(cached_layers[i] for i in idx))
    print(f"  target steering layers: {args.target_layers}")

    # Build samples
    print("Loading raw dataset for prompts...")
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
    print(f"  total samples: {len(samples)}")
    types_arr = np.array([s["type"] for s in samples])
    prompts = [make_prompt(s["text"], tokenizer, is_instruct) for s in samples]

    # Sample residual norm at each target layer (use first prompt)
    layer_norms = {}
    for L in args.target_layers:
        layer_norms[L] = get_residual_norm(model, tokenizer, prompts[0], L)
    print("  residual stream norms (first sample):")
    for L, n in layer_norms.items():
        print(f"    L{L}: {n:.2f}")

    # Build random direction (unit norm, same shape as a netsec direction)
    sample_dir = next(d for d in dir_dict[args.target_layers[0]].values() if d is not None)
    g = torch.Generator().manual_seed(42)
    rand_dir = torch.randn(sample_dir.shape, generator=g, dtype=torch.float32)
    rand_dir = rand_dir / rand_dir.norm()

    results = {
        "model": args.model,
        "dataset": args.dataset,
        "is_instruct": is_instruct,
        "jbb_file": args.jbb_file,
        "n_layers": n_layers,
        "target_layers": list(args.target_layers),
        "alphas": list(args.alphas),
        "alpha_relative": args.alpha_relative,
        "multi_layer": args.multi_layer,
        "layer_norms": layer_norms,
        "samples": samples,
        "results": {},
    }

    # Baseline (no steering)
    print("\n=== Baseline (no steering) ===")
    base = yes_minus_no_logit(model, tokenizer, prompts, yes_ids, no_ids)
    results["baseline_yesno_diff"] = base
    arr = np.array(base)
    print(f"  mean = {arr.mean():+.3f}  ", end="")
    for t in sorted(set(types_arr)):
        m = arr[types_arr == t]
        print(f"{t}={m.mean():+.2f} ", end="")
    print()

    # Sweep — for each (target_layer, direction_name, alpha), measure
    for target_layer in args.target_layers:
        results["results"][int(target_layer)] = {}
        dirs_at_layer = dir_dict[target_layer]
        norm_scale = layer_norms[target_layer] if args.alpha_relative else 1.0

        for d_name, d_vec in [
            ("cic_attack",  dirs_at_layer["cic_attack"]),
            ("uns_attack",  dirs_at_layer["uns_attack"]),
            ("harmful",     dirs_at_layer["harmful"]),
            ("random",      rand_dir),
        ]:
            if d_vec is None:
                continue
            print(f"\n=== Steering layer={target_layer} {'(multi)' if args.multi_layer else '(single)'} "
                  f"direction={d_name}{' [α relative]' if args.alpha_relative else ''} ===")
            per_alpha = {}
            for alpha in args.alphas:
                eff_alpha = alpha * norm_scale
                if alpha == 0.0:
                    handles = []
                elif args.multi_layer:
                    handles = install_steer_hook_multi(model, eff_alpha, d_vec, target_layer)
                else:
                    handles = install_steer_hook_single(model, eff_alpha, d_vec, target_layer)
                try:
                    diffs = yes_minus_no_logit(model, tokenizer, prompts, yes_ids, no_ids)
                finally:
                    remove_hooks(handles)
                per_alpha[float(alpha)] = diffs
                arr = np.array(diffs)
                print(f"  α={alpha:+.1f} (eff={eff_alpha:+.2f})  mean={arr.mean():+.3f}  ", end="")
                for t in sorted(set(types_arr)):
                    m = arr[types_arr == t]
                    print(f"{t}={m.mean():+.2f} ", end="")
                print()
            results["results"][int(target_layer)][d_name] = per_alpha

    out_path = OUT / f"phase1_steering_v2_{args.model}_{args.dataset}.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
