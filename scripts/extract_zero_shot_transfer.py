"""Extract Qwen3-8B-Base hidden states for 3 new datasets, for zero-shot transfer.

Dataset-specific loaders + text formatters. We sample N=1500 normal + 1500
attack rows per dataset (when available), format as natural language, and
extract reps at every layer that's also in cicids2017 cache (so we can
project through cicids2017's attack_direction at matching layers).

Output: results/representations/<dataset>/<model>/zero_shot.pt
  keys: hidden_states (dict L -> tensor(N, hidden_dim)),
        labels (0/1), attack_types (str), texts (str)
"""
import argparse, os, sys, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from pathlib import Path

from utils.model_utils import load_model_and_tokenizer, extract_hidden_states, set_random_seed, flush

BASE = Path(__file__).resolve().parent.parent
REPS = BASE / "results" / "representations"
DATA_ROOT = Path("/scratch/yh6210/datasets/LLM_Rep_NetSec_datasets")
CACHE_DIR = "/scratch/yh6210/transformers"


# ---------------- CIC-IDS-2018 ----------------
CIC2018_FEATURES = [
    ("Protocol",            "protocol"),
    ("Dst Port",            "destination port"),
    ("Flow Duration",       "flow duration"),
    ("Tot Fwd Pkts",        "total forward packets"),
    ("Tot Bwd Pkts",        "total backward packets"),
    ("TotLen Fwd Pkts",     "total forward bytes"),
    ("TotLen Bwd Pkts",     "total backward bytes"),
    ("Flow Byts/s",         "flow bytes per second"),
    ("Flow Pkts/s",         "flow packets per second"),
    ("Fwd IAT Mean",        "forward inter-arrival time mean"),
    ("Bwd IAT Mean",        "backward inter-arrival time mean"),
    ("SYN Flag Cnt",        "SYN flags"),
    ("RST Flag Cnt",        "RST flags"),
    ("ACK Flag Cnt",        "ACK flags"),
    ("PSH Flag Cnt",        "PSH flags"),
    ("Init Fwd Win Byts",   "initial forward window bytes"),
    ("Init Bwd Win Byts",   "initial backward window bytes"),
]


def load_cic2018(n_normal=1500, n_attack=1500, seed=42):
    csvs = sorted((DATA_ROOT / "cicids2018_full").glob("*.csv"))
    print(f"  reading {len(csvs)} CSVs (sampling on the fly)")
    rng = np.random.RandomState(seed)
    benigns, attacks = [], []
    for fp in csvs:
        # Read in chunks to keep memory OK
        for chunk in pd.read_csv(fp, chunksize=200_000, low_memory=False):
            chunk.columns = chunk.columns.str.strip()
            if "Label" not in chunk.columns:
                continue
            chunk["is_attack"] = (chunk["Label"] != "Benign").astype(int)
            chunk["attack_type"] = chunk["Label"].apply(lambda x: "Normal" if x == "Benign" else str(x))
            n_b = chunk[chunk.is_attack == 0]
            n_a = chunk[chunk.is_attack == 1]
            if len(n_b) > 0:
                benigns.append(n_b.sample(min(len(n_b), 200), random_state=rng))
            if len(n_a) > 0:
                attacks.append(n_a.sample(min(len(n_a), 400), random_state=rng))
    benigns_df = pd.concat(benigns, ignore_index=True)
    attacks_df = pd.concat(attacks, ignore_index=True)
    print(f"  benign pool {len(benigns_df)}, attack pool {len(attacks_df)}")
    print(f"  attack types: {attacks_df.attack_type.value_counts().to_dict()}")
    benigns_df = benigns_df.sample(min(len(benigns_df), n_normal), random_state=rng)
    attacks_df = attacks_df.sample(min(len(attacks_df), n_attack), random_state=rng)
    df = pd.concat([benigns_df, attacks_df], ignore_index=True)
    df = df.sample(frac=1, random_state=rng).reset_index(drop=True)
    return df


def format_cic2018(row):
    parts = ["Network connection log:"]
    for col, desc in CIC2018_FEATURES:
        if col in row.index:
            v = row[col]
            if pd.notna(v):
                parts.append(f"  {desc}: {v}")
    return "\n".join(parts)


# ---------------- IoT-23 ----------------
IOT23_FEATURES = [
    ("proto",          "protocol"),
    ("id.orig_p",      "source port"),
    ("id.resp_p",      "destination port"),
    ("duration",       "duration"),
    ("orig_bytes",     "originator bytes"),
    ("resp_bytes",     "responder bytes"),
    ("service",        "service"),
    ("conn_state",     "connection state"),
    ("history",        "history"),
    ("orig_pkts",      "originator packets"),
    ("resp_pkts",      "responder packets"),
    ("missed_bytes",   "missed bytes"),
    ("orig_ip_bytes",  "originator IP bytes"),
    ("resp_ip_bytes",  "responder IP bytes"),
]


def load_iot23(n_normal=1500, n_attack=1500, seed=42):
    files = sorted((DATA_ROOT / "iot23" / "data").glob("*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["is_attack"] = (df["label"] != "Benign").astype(int)
    df["attack_type"] = df["label"].apply(lambda x: "Normal" if x == "Benign" else str(x))
    rng = np.random.RandomState(seed)
    benigns = df[df.is_attack == 0]
    attacks = df[df.is_attack == 1]
    print(f"  benign pool {len(benigns)}, attack pool {len(attacks)}")
    print(f"  attack types: {attacks.attack_type.value_counts().to_dict()}")
    benigns = benigns.sample(min(len(benigns), n_normal), random_state=rng)
    # Take stratified sample across attack types (avoid Okiru dominating)
    types = attacks["attack_type"].value_counts().index.tolist()
    per_type = max(1, n_attack // len(types))
    samples = []
    for t in types:
        sub = attacks[attacks.attack_type == t]
        samples.append(sub.sample(min(len(sub), per_type), random_state=rng))
    attacks = pd.concat(samples, ignore_index=True)
    if len(attacks) > n_attack:
        attacks = attacks.sample(n_attack, random_state=rng)
    out = pd.concat([benigns, attacks], ignore_index=True)
    out = out.sample(frac=1, random_state=rng).reset_index(drop=True)
    return out


def format_iot23(row):
    parts = ["Network connection log:"]
    for col, desc in IOT23_FEATURES:
        if col in row.index:
            v = row[col]
            if pd.notna(v):
                parts.append(f"  {desc}: {v}")
    return "\n".join(parts)


# ---------------- CTU-13 (binetflow) ----------------
CTU13_FEATURES = [
    ("SrcPort",       "source port"),
    ("DstPort",       "destination port"),
    ("Duration",      "duration"),
    ("TotPkts",       "total packets"),
    ("TotBytes",      "total bytes"),
    ("SrcBytes",      "source bytes"),
    ("sTos",          "source ToS"),
    ("dTos",          "destination ToS"),
    ("PktByteRatio",  "packet/byte ratio"),
    ("BytePerPkt",    "bytes per packet"),
    ("SrcByteRatio",  "source byte ratio"),
]


def load_ctu13(n_normal=1500, n_attack=1500, seed=42):
    files = sorted((DATA_ROOT / "ctu13" / "data").glob("*.parquet"))
    print(f"  reading {len(files)} parquets")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"  total rows: {len(df)}, label_multi: {df['label_multi'].value_counts().to_dict()}")
    # Use ONLY Normal vs Botnet (skip Background — unlabeled mixed traffic)
    benigns = df[df["label_multi"] == "Normal"].copy()
    attacks = df[df["label_multi"] == "Botnet"].copy()
    benigns["attack_type"] = "Normal"
    benigns["is_attack"] = 0
    attacks["is_attack"] = 1
    # Use the verbose Label as "attack_type" — it has botnet variants
    attacks["attack_type"] = attacks["Label"].apply(lambda s: str(s).replace("flow=", ""))
    print(f"  Normal {len(benigns)},  Botnet {len(attacks)}")
    print(f"  Botnet variants: {attacks.attack_type.value_counts().head(8).to_dict()}")
    rng = np.random.RandomState(seed)
    benigns = benigns.sample(min(len(benigns), n_normal), random_state=rng)
    # Stratified across botnet variants
    variants = attacks["attack_type"].value_counts().index.tolist()
    per = max(1, n_attack // len(variants))
    samples = []
    for v in variants:
        sub = attacks[attacks.attack_type == v]
        samples.append(sub.sample(min(len(sub), per), random_state=rng))
    attacks = pd.concat(samples, ignore_index=True)
    if len(attacks) > n_attack:
        attacks = attacks.sample(n_attack, random_state=rng)
    out = pd.concat([benigns, attacks], ignore_index=True)
    out = out.sample(frac=1, random_state=rng).reset_index(drop=True)
    return out


def format_ctu13(row):
    parts = ["Network connection log:"]
    for col, desc in CTU13_FEATURES:
        if col in row.index:
            v = row[col]
            if pd.notna(v):
                parts.append(f"  {desc}: {v}")
    return "\n".join(parts)


# ---------------- Main ----------------
LOADERS = {
    "cicids2018": (load_cic2018, format_cic2018),
    "iot23":      (load_iot23,   format_iot23),
    "ctu13":      (load_ctu13,   format_ctu13),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True, choices=list(LOADERS.keys()))
    ap.add_argument("--n_normal", type=int, default=1500)
    ap.add_argument("--n_attack", type=int, default=1500)
    ap.add_argument("--layers", nargs="+", type=int, default=None,
                    help="If given, only extract these layers. Default: match cicids2017 cached layers.")
    args = ap.parse_args()

    set_random_seed(42)

    out_dir = REPS / args.dataset / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "zero_shot.pt"
    if out_path.exists():
        print(f"already exists: {out_path}")
        return

    # Default layers: match cicids2017 cached layers for this model
    if args.layers is None:
        ref_path = REPS / "cicids2017" / args.model / "layers.json"
        if ref_path.exists():
            import json
            args.layers = json.loads(ref_path.read_text())["layers"]
        else:
            print("WARN: no cicids2017 layers.json — using all layers")

    print(f"\n=== {args.dataset} extraction for {args.model} ===")
    loader, formatter = LOADERS[args.dataset]
    df = loader(n_normal=args.n_normal, n_attack=args.n_attack)
    df["text"] = df.apply(formatter, axis=1)
    print(f"  total samples: {len(df)}")
    print(f"  is_attack: {df.is_attack.value_counts().to_dict()}")
    print(f"  example text:\n{df.iloc[0]['text'][:400]}\n...")

    # Load model
    print(f"  loading {args.model} ...")
    dtype = "bfloat16" if "gemma" in args.model.lower() else "float16"
    model, tokenizer = load_model_and_tokenizer(
        f"{CACHE_DIR}/{args.model}", cache_dir=CACHE_DIR, dtype=dtype, device="cuda",
    )

    import os as _os
    _bs = int(_os.environ.get("BATCH_SIZE", "32"))
    print(f"  extracting at layers {args.layers} (batch_size={_bs})")
    hs = extract_hidden_states(
        model, tokenizer, df["text"].tolist(),
        layers=args.layers, token_position="last", batch_size=_bs, max_seq_length=512,
    )

    save = {
        "hidden_states": hs,
        "labels": torch.tensor(df["is_attack"].values),
        "attack_types": df["attack_type"].values,
        "texts": df["text"].tolist(),
    }
    torch.save(save, out_path)
    print(f"saved {out_path}")
    print(f"  layers: {sorted(hs.keys())}")
    print(f"  per-layer shape: {hs[sorted(hs.keys())[0]].shape}")

    del model, tokenizer
    flush()


if __name__ == "__main__":
    main()
