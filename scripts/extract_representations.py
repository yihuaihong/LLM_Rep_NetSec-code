"""
Batch extraction of hidden state representations from LLMs for network security logs.
Usage: python scripts/extract_representations.py --config configs/default.yaml
"""

import argparse
import os
import sys
import torch
import yaml
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_utils import load_model_and_tokenizer, extract_hidden_states, flush, set_random_seed
from utils.data_utils import load_dataset, format_dataset, create_generalization_splits


def main(args):
    set_random_seed(42)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["name"]
    cache_dir = cfg["model"]["cache_dir"]
    model_path = os.path.join(cache_dir, model_name)

    print(f"Loading model: {model_path}")
    model, tokenizer = load_model_and_tokenizer(
        model_path,
        dtype=cfg["model"]["dtype"],
        device=cfg["model"]["device"],
    )

    for ds_cfg in cfg["data"]["datasets"]:
        ds_name = ds_cfg["name"]
        ds_path = ds_cfg["path"]

        if not os.path.exists(ds_path):
            print(f"Dataset {ds_name} not found at {ds_path}, skipping.")
            continue

        print(f"\nLoading dataset: {ds_name}")
        df = load_dataset(ds_name, ds_path)
        df = format_dataset(df, ds_name, fmt=cfg["data"]["text_format"])

        holdout_types = cfg["data"]["holdout_attack_types"].get(ds_name, [])
        splits = create_generalization_splits(
            df,
            holdout_attack_types=holdout_types,
            max_samples_per_class=cfg["data"]["max_samples_per_class"],
        )

        rep_cfg = cfg["representation"]
        save_dir = Path(rep_cfg["cache_dir"]) / ds_name / model_name
        save_dir.mkdir(parents=True, exist_ok=True)

        for split_name, split_df in splits.items():
            if len(split_df) == 0:
                print(f"  Split {split_name} is empty, skipping.")
                continue

            print(f"  Extracting {split_name}: {len(split_df)} samples")
            texts = split_df["text"].tolist()
            labels = split_df["is_attack"].values
            attack_types = split_df["attack_type"].values

            layers = rep_cfg["layers"]
            if layers == "all":
                layers = None  # extract_hidden_states will use all layers

            hidden_states = extract_hidden_states(
                model, tokenizer, texts,
                layers=layers,
                token_position=rep_cfg["token_position"],
                batch_size=rep_cfg["batch_size"],
                max_seq_length=rep_cfg["max_seq_length"],
            )

            # Save representations and labels
            save_data = {
                "hidden_states": hidden_states,
                "labels": torch.tensor(labels),
                "attack_types": attack_types,
            }
            save_path = save_dir / f"{split_name}.pt"
            torch.save(save_data, save_path)
            print(f"  Saved to {save_path}")

            flush()

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    main(args)
