"""
Extract Llama-3-8B-Instruct hidden states on JBB-Behaviors (harmful + benign).

Outputs:
  results/representations/jbb/Meta-Llama-3-8B-Instruct/all.pt
    keys: hidden_states (dict layer->tensor), labels, categories, texts

The extracted representations cover ALL 33 hidden_states layers (0..32) so any
subset of 4 layers can be picked at PDF compile time.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from datasets import load_dataset

from utils.model_utils import load_model_and_tokenizer, extract_hidden_states, set_random_seed

MODEL_NAME = "Meta-Llama-3-8B-Instruct"
CACHE_DIR = "/scratch/yh6210/transformers"
OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "representations" / "jbb" / MODEL_NAME


def main():
    set_random_seed(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_NAME} from {CACHE_DIR} ...")
    model, tokenizer = load_model_and_tokenizer(
        f"{CACHE_DIR}/{MODEL_NAME}",
        cache_dir=CACHE_DIR,
        dtype="float16",
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
    print(f"  harmful categories: {sorted(set(categories[labels==1]))}")

    # Apply chat template — important for instruct models
    print("Formatting with chat template ...")
    formatted = []
    for g in goals:
        msgs = [{"role": "user", "content": g}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        formatted.append(text)

    print("Extracting hidden states from all layers ...")
    hidden = extract_hidden_states(
        model, tokenizer, formatted,
        layers=None,             # all layers (0..32)
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
    out_path = OUT_DIR / "all.pt"
    torch.save(out, out_path)

    sample_layer = sorted(hidden.keys())[0]
    print(f"\nSaved to {out_path}")
    print(f"  layers: {sorted(hidden.keys())}")
    print(f"  per-layer shape: {hidden[sample_layer].shape}")
    print(f"  labels: {labels.shape}, categories: {categories.shape}")


if __name__ == "__main__":
    main()
