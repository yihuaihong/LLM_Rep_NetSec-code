"""
Data utilities for loading, parsing, and formatting network security log datasets.
Supports CICIDS2017, UNSW-NB15, and extensible to other datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


# ──────────────────────────────────────────────
# Dataset Loaders
# ──────────────────────────────────────────────

def load_cicids2017(data_dir: str) -> pd.DataFrame:
    """Load CICIDS2017 dataset (parquet or CSV) and return unified DataFrame.

    Expected data path (HuggingFace download):
        data_dir/Network-Flows/CICIDS_Flow.parquet
    Or CSV files directly under data_dir.
    """
    data_dir = Path(data_dir)

    # Try parquet first (HuggingFace format from rdpahalavan/CIC-IDS2017)
    parquet_path = data_dir / "Network-Flows" / "CICIDS_Flow.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    else:
        # Fallback: CSV files
        dfs = []
        for csv_file in sorted(data_dir.glob("*.csv")):
            d = pd.read_csv(csv_file, encoding="utf-8", low_memory=False)
            d.columns = d.columns.str.strip()
            dfs.append(d)
        df = pd.concat(dfs, ignore_index=True)

    # Unify labels - handle both column naming conventions
    label_col = "attack_label" if "attack_label" in df.columns else "Label"
    df["is_attack"] = (df[label_col] != "BENIGN").astype(int)
    df["attack_type"] = df[label_col].apply(lambda x: "Normal" if x == "BENIGN" else str(x))
    df["source_dataset"] = "cicids2017"

    # Clean infinities and NaNs in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df


def load_unsw_nb15(data_dir: str) -> pd.DataFrame:
    """Load UNSW-NB15 dataset (parquet or CSV) and return unified DataFrame.

    Expected data path (HuggingFace download):
        data_dir/Network-Flows/UNSW_Flow.parquet
    Or CSV files directly under data_dir.
    """
    data_dir = Path(data_dir)

    # Try parquet first (HuggingFace format from rdpahalavan/UNSW-NB15)
    parquet_path = data_dir / "Network-Flows" / "UNSW_Flow.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    else:
        # Fallback: CSV files
        dfs = []
        for csv_file in sorted(data_dir.glob("*.csv")):
            d = pd.read_csv(csv_file, encoding="utf-8", low_memory=False)
            d.columns = d.columns.str.strip()
            dfs.append(d)
        df = pd.concat(dfs, ignore_index=True)

    # Unify labels - handle both column naming conventions
    if "attack_label" in df.columns:
        df["is_attack"] = (df["attack_label"] != "normal").astype(int)
        df["attack_type"] = df["attack_label"].apply(lambda x: "Normal" if x == "normal" else str(x).capitalize())
    elif "label" in df.columns:
        df["is_attack"] = df["label"].astype(int)
        if "attack_cat" in df.columns:
            df["attack_type"] = df["attack_cat"].fillna("Normal").str.strip()
        else:
            df["attack_type"] = df["is_attack"].map({0: "Normal", 1: "Attack"})
    df["source_dataset"] = "unsw_nb15"

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df


DATASET_LOADERS = {
    "cicids2017": load_cicids2017,
    "unsw_nb15": load_unsw_nb15,
}


def load_dataset(name: str, data_dir: str) -> pd.DataFrame:
    """Load a dataset by name."""
    if name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_LOADERS.keys())}")
    return DATASET_LOADERS[name](data_dir)


# ──────────────────────────────────────────────
# Text Formatting (convert log rows to natural language for LLM input)
# ──────────────────────────────────────────────

# Key features for CICIDS2017 text formatting
# Supports both parquet (HuggingFace) and original CSV column names
CICIDS_TEXT_FEATURES = [
    ("protocol", "protocol"),
    ("source_port", "source port"),
    ("destination_port", "destination port"),
    ("Flow Duration", "flow duration"),
    ("Total Fwd Packets", "total forward packets"),
    ("Total Backward Packets", "total backward packets"),
    ("Total Length of Fwd Packets", "total forward bytes"),
    ("Total Length of Bwd Packets", "total backward bytes"),
    ("Flow Bytes/s", "flow bytes per second"),
    ("Flow Packets/s", "flow packets per second"),
    ("Fwd IAT Mean", "forward inter-arrival time mean"),
    ("Bwd IAT Mean", "backward inter-arrival time mean"),
    ("SYN Flag Count", "SYN flags"),
    ("RST Flag Count", "RST flags"),
    ("ACK Flag Count", "ACK flags"),
    ("PSH Flag Count", "PSH flags"),
    ("Init_Win_bytes_forward", "initial forward window bytes"),
    ("Init_Win_bytes_backward", "initial backward window bytes"),
]

# Key features for UNSW-NB15 text formatting
# Supports both parquet (HuggingFace) and original CSV column names
UNSW_TEXT_FEATURES = [
    ("protocol", "protocol"),
    ("source_port", "source port"),
    ("destination_port", "destination port"),
    ("dur", "duration"),
    ("sbytes", "source bytes"),
    ("dbytes", "destination bytes"),
    ("sttl", "source TTL"),
    ("dttl", "destination TTL"),
    ("sloss", "source packets lost"),
    ("dloss", "destination packets lost"),
    ("service", "service"),
    ("state", "connection state"),
    ("spkts", "source packets"),
    ("dpkts", "destination packets"),
    ("swin", "source TCP window"),
    ("dwin", "destination TCP window"),
    ("smean", "source packet size mean"),
    ("dmean", "destination packet size mean"),
    ("ct_srv_src", "connections to same service from source"),
    ("ct_dst_sport_ltm", "connections to same dest port"),
]


def format_log_natural_language(row: pd.Series, dataset_name: str) -> str:
    """Convert a log row into a natural language description for LLM input."""
    if dataset_name == "cicids2017":
        features = CICIDS_TEXT_FEATURES
    elif dataset_name == "unsw_nb15":
        features = UNSW_TEXT_FEATURES
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    parts = ["Network connection log:"]
    for col_name, desc in features:
        if col_name in row.index:
            val = row[col_name]
            if pd.notna(val):
                parts.append(f"  {desc}: {val}")

    return "\n".join(parts)


def format_log_key_value(row: pd.Series, dataset_name: str) -> str:
    """Convert a log row into key=value format for LLM input."""
    if dataset_name == "cicids2017":
        features = CICIDS_TEXT_FEATURES
    elif dataset_name == "unsw_nb15":
        features = UNSW_TEXT_FEATURES
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    parts = []
    for col_name, desc in features:
        if col_name in row.index:
            val = row[col_name]
            if pd.notna(val):
                parts.append(f"{desc}={val}")

    return " | ".join(parts)


def format_dataset(df: pd.DataFrame, dataset_name: str, fmt: str = "natural_language") -> pd.DataFrame:
    """Add a 'text' column to the DataFrame with formatted log text."""
    format_fn = format_log_natural_language if fmt == "natural_language" else format_log_key_value
    df = df.copy()
    df["text"] = df.apply(lambda row: format_fn(row, dataset_name), axis=1)
    return df


# ──────────────────────────────────────────────
# Train/Test Splits with Attack-Type Holdout
# ──────────────────────────────────────────────

def create_generalization_splits(
    df: pd.DataFrame,
    holdout_attack_types: list[str],
    max_samples_per_class: Optional[int] = None,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> dict:
    """
    Create train/test splits where certain attack types are held out entirely.

    Returns:
        dict with keys:
        - "train": DataFrame (normal + known attacks, for training)
        - "test_known": DataFrame (normal + known attacks, for standard eval)
        - "test_holdout": DataFrame (normal + held-out attacks, for zero-day eval)
    """
    rng = np.random.RandomState(seed)

    # Separate normal, known attacks, and holdout attacks
    df_normal = df[df["is_attack"] == 0].copy()
    df_attacks = df[df["is_attack"] == 1].copy()

    holdout_mask = df_attacks["attack_type"].isin(holdout_attack_types)
    df_holdout_attacks = df_attacks[holdout_mask].copy()
    df_known_attacks = df_attacks[~holdout_mask].copy()

    # Subsample if needed
    if max_samples_per_class is not None:
        if len(df_normal) > max_samples_per_class:
            df_normal = df_normal.sample(n=max_samples_per_class, random_state=rng)
        if len(df_known_attacks) > max_samples_per_class:
            df_known_attacks = df_known_attacks.sample(n=max_samples_per_class, random_state=rng)
        if len(df_holdout_attacks) > max_samples_per_class:
            df_holdout_attacks = df_holdout_attacks.sample(n=max_samples_per_class, random_state=rng)

    # Split normal into 3 parts: train, test_known, test_holdout
    normal_idx = df_normal.index.tolist()
    rng.shuffle(normal_idx)
    n = len(normal_idx)
    n_train = int(n * (1 - test_ratio * 2))
    n_test_known = int(n * test_ratio)
    normal_train = df_normal.loc[normal_idx[:n_train]]
    normal_test_known = df_normal.loc[normal_idx[n_train:n_train + n_test_known]]
    normal_test_holdout = df_normal.loc[normal_idx[n_train + n_test_known:]]

    # Split known attacks into train/test
    known_idx = df_known_attacks.index.tolist()
    rng.shuffle(known_idx)
    k_split = int(len(known_idx) * (1 - test_ratio))
    known_train = df_known_attacks.loc[known_idx[:k_split]]
    known_test = df_known_attacks.loc[known_idx[k_split:]]

    return {
        "train": pd.concat([normal_train, known_train], ignore_index=True),
        "test_known": pd.concat([normal_test_known, known_test], ignore_index=True),
        "test_holdout": pd.concat([normal_test_holdout, df_holdout_attacks], ignore_index=True),
    }


def _subsample_balanced(df: pd.DataFrame, max_per_class: int, rng: np.random.RandomState) -> pd.DataFrame:
    """Subsample each class to at most max_per_class samples."""
    groups = []
    for _, group in df.groupby("is_attack"):
        if len(group) > max_per_class:
            group = group.sample(n=max_per_class, random_state=rng)
        groups.append(group)
    return pd.concat(groups, ignore_index=True)
