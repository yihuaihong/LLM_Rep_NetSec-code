"""Extract JBB harmful + benign hidden states for any model.

Usage:
  python3 scripts/extract_harmful_harmless_multimodel.py --model Qwen3-8B
  python3 scripts/extract_harmful_harmless_multimodel.py --model gemma-2-9b-it
  python3 scripts/extract_harmful_harmless_multimodel.py --model Mistral-7B-Instruct-v0.3

Saves to results/representations/jbb/<model>/all.pt
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from datasets import load_dataset

from utils.model_utils import load_model_and_tokenizer, extract_hidden_states, set_random_seed, flush

CACHE_DIR = "/scratch/yh6210/transformers"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--no_chat_template", action="store_true",
                    help="feed raw goal text without chat template (use for base models)")
    ap.add_argument("--suffix", default="",
                    help="suffix appended to output filename (e.g. '_raw')")
    args = ap.parse_args()

    set_random_seed(42)
    out_dir = Path(__file__).resolve().parent.parent / "results" / "representations" / "jbb" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"all{args.suffix}.pt"
    out_path = out_dir / out_name
    if out_path.exists():
        print(f"already exists: {out_path}")
        return

    print(f"Loading {args.model} ...")
    dtype = "bfloat16" if "gemma" in args.model.lower() else "float16"
    model, tokenizer = load_model_and_tokenizer(
        f"{CACHE_DIR}/{args.model}",
        cache_dir=CACHE_DIR,
        dtype=dtype,
        device="cuda",
    )

    print("Loading JBB-Behaviors ...")
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    harmful = [(g, c, 1) for g, c in zip(ds["harmful"]["Goal"], ds["harmful"]["Category"])]
    benign  = [(g, c, 0) for g, c in zip(ds["benign"]["Goal"],  ds["benign"]["Category"])]
    items = harmful + benign
    goals      = [x[0] for x in items]
    categories = np.array([x[1] for x in items], dtype=object)
    labels     = np.array([x[2] for x in items], dtype=np.int64)

    print(f"  total: {len(items)}  (harmful={sum(labels==1)}, benign={sum(labels==0)})")

    if args.no_chat_template:
        print("Using RAW goal text (no chat template) ...")
        formatted = list(goals)
    else:
        print("Formatting with chat template ...")
        formatted = []
        for g in goals:
            msgs = [{"role": "user", "content": g}]
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            formatted.append(text)

    print("Extracting hidden states from all layers ...")
    hidden = extract_hidden_states(
        model, tokenizer, formatted,
        layers=None,
        token_position="last",
        batch_size=8,
        max_seq_length=256,
    )

    out = {
        "hidden_states": hidden,
        "labels": torch.from_numpy(labels),
        "categories": categories,
        "texts": goals,
    }
    torch.save(out, out_path)
    print(f"Saved to {out_path}")
    print(f"  layers: {sorted(hidden.keys())}")
    print(f"  per-layer shape: {hidden[sorted(hidden.keys())[0]].shape}")

    del model, tokenizer
    flush()


if __name__ == "__main__":
    main()
