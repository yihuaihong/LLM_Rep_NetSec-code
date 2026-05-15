"""Three quick experiments using cached reps:

1A. Direction cosine across prompt formats (Qwen3-Base on cicids2017):
    natural_language vs key_value
1B. Single-numeric-feature baseline: use only top axis feature(s) as input,
    train logistic regression. Compare in-domain + holdout AUROC vs LLM probe.
1C. Multi-dataset joint probe: train on UNION(cic2017_train, unsw_train),
    test on each dataset's holdout. Does joint training help generalization?
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score

from utils.probe_utils import extract_separation_direction
from utils.data_utils import load_dataset, create_generalization_splits

BASE = Path(__file__).resolve().parent.parent
REPS = BASE / "results" / "representations"
OUT = BASE / "results" / "metrics"

DATASET_PATHS = {
    "cicids2017": "/scratch/yh6210/datasets/LLM_Rep_NetSec_datasets/cicids2017/",
    "unsw_nb15":  "/scratch/yh6210/datasets/LLM_Rep_NetSec_datasets/unsw_nb15/",
}
HOLDOUT = {
    "cicids2017": ["Heartbleed", "Infiltration", "Bot"],
    "unsw_nb15":  ["Shellcode", "Worms", "Backdoor"],
}


def to_f32(t):
    return t.float() if t.dtype in (torch.bfloat16, torch.float16) else t


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(
        a.float().unsqueeze(0), b.float().unsqueeze(0)
    ).item())


# ============================================================
# 1A: format direction cosine
# ============================================================
def exp_1a():
    print("\n=== 1A: direction cosine across prompt formats ===")
    out = {}
    model = "Qwen3-8B-Base"
    for ds in ["cicids2017", "unsw_nb15"]:
        nat = REPS / ds / model / "train.pt"
        kv  = REPS / f"{ds}_key_value" / model / "train.pt"
        if not nat.exists() or not kv.exists():
            print(f"  [{ds}] missing reps"); continue
        n = torch.load(nat, weights_only=False)
        k = torch.load(kv, weights_only=False)
        layers = sorted(set(n["hidden_states"].keys()) & set(k["hidden_states"].keys()))
        rec = {}
        print(f"  [{ds}]  layers: {layers}")
        for L in layers:
            d_n = extract_separation_direction(to_f32(n["hidden_states"][L]), n["labels"].numpy())
            d_k = extract_separation_direction(to_f32(k["hidden_states"][L]), k["labels"].numpy())
            c = cos(d_n, d_k)
            rec[int(L)] = c
            print(f"    L{L:>3}: cos(d_nat, d_kv) = {c:+.4f}")
        out[ds] = rec
    json.dump(out, open(OUT / "phase2_1a_format_direction_cos.json", "w"), indent=2)
    print("  saved phase2_1a_format_direction_cos.json")


# ============================================================
# 1B: single-feature baseline (no LLM)
# ============================================================
def exp_1b():
    print("\n=== 1B: numeric-feature-only baselines (no LLM) ===")
    out = {}
    for ds in ["cicids2017", "unsw_nb15"]:
        df = load_dataset(ds, DATASET_PATHS[ds])
        splits = create_generalization_splits(
            df, holdout_attack_types=HOLDOUT[ds],
            max_samples_per_class=5000, seed=42,
        )
        train_df = splits["train"].reset_index(drop=True)
        known_df = splits["test_known"].reset_index(drop=True)
        ho_df = splits["test_holdout"].reset_index(drop=True)

        # Pick numeric columns (exclude label, attack_type, etc.)
        num_cols = [c for c in train_df.columns
                    if pd.api.types.is_numeric_dtype(train_df[c])
                    and c not in ("is_attack",)]
        print(f"\n  [{ds}]  using {len(num_cols)} numeric features")

        # Replace inf/nan with median
        train_df[num_cols] = train_df[num_cols].replace([np.inf, -np.inf], np.nan)
        known_df[num_cols] = known_df[num_cols].replace([np.inf, -np.inf], np.nan)
        ho_df[num_cols]    = ho_df[num_cols].replace([np.inf, -np.inf], np.nan)
        med = train_df[num_cols].median()
        train_df[num_cols] = train_df[num_cols].fillna(med)
        known_df[num_cols] = known_df[num_cols].fillna(med)
        ho_df[num_cols]    = ho_df[num_cols].fillna(med)

        X_tr, y_tr = train_df[num_cols].values, train_df["is_attack"].values
        X_kn, y_kn = known_df[num_cols].values, known_df["is_attack"].values
        X_ho, y_ho = ho_df[num_cols].values, ho_df["is_attack"].values

        out[ds] = {}
        for tag, X_train_use in [
            ("all_features", X_tr),
        ]:
            pipe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, n_jobs=-1, random_state=42))
            pipe.fit(X_train_use, y_tr)
            kn_proba = pipe.predict_proba(X_kn)[:, 1]
            ho_proba = pipe.predict_proba(X_ho)[:, 1]
            kn_acc = accuracy_score(y_kn, pipe.predict(X_kn))
            ho_acc = accuracy_score(y_ho, pipe.predict(X_ho))
            kn_auc = roc_auc_score(y_kn, kn_proba)
            ho_auc = roc_auc_score(y_ho, ho_proba)
            out[ds][tag] = {
                "n_features": int(X_train_use.shape[1]),
                "known_acc": float(kn_acc), "known_auroc": float(kn_auc),
                "holdout_acc": float(ho_acc), "holdout_auroc": float(ho_auc),
                "gap_acc": float(kn_acc - ho_acc),
                "feature_names": num_cols,
            }
            print(f"    [{tag}] n={X_train_use.shape[1]}  known_acc={kn_acc:.3f}  ho_acc={ho_acc:.3f}  gap={kn_acc-ho_acc:+.3f}")

        # Also: top-1 single-axis (the most probe-aligned numeric feature from B1)
        top1 = {
            "cicids2017": "Init_Win_bytes_forward",
            "unsw_nb15":  "destination_port",
        }.get(ds)
        if top1 and top1 in num_cols:
            X_tr1 = train_df[[top1]].values
            X_kn1 = known_df[[top1]].values
            X_ho1 = ho_df[[top1]].values
            pipe1 = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=42))
            pipe1.fit(X_tr1, y_tr)
            kn_proba = pipe1.predict_proba(X_kn1)[:, 1]
            ho_proba = pipe1.predict_proba(X_ho1)[:, 1]
            kn_acc = accuracy_score(y_kn, pipe1.predict(X_kn1))
            ho_acc = accuracy_score(y_ho, pipe1.predict(X_ho1))
            out[ds][f"top1_{top1}"] = {
                "n_features": 1, "feature_names": [top1],
                "known_acc": float(kn_acc), "holdout_acc": float(ho_acc),
                "known_auroc": float(roc_auc_score(y_kn, kn_proba)),
                "holdout_auroc": float(roc_auc_score(y_ho, ho_proba)),
                "gap_acc": float(kn_acc - ho_acc),
            }
            print(f"    [top1: {top1}] known_acc={kn_acc:.3f}  ho_acc={ho_acc:.3f}  gap={kn_acc-ho_acc:+.3f}")
    json.dump(out, open(OUT / "phase2_1b_numeric_baseline.json", "w"), indent=2)
    print("  saved phase2_1b_numeric_baseline.json")


# ============================================================
# 1C: multi-dataset joint probe
# ============================================================
def exp_1c():
    print("\n=== 1C: joint train on cic2017+unsw, eval each dataset ===")
    out = {}
    for model in ["Qwen3-8B-Base", "Llama-3.1-8B-Instruct"]:
        out[model] = {}
        cic_train = torch.load(REPS / "cicids2017" / model / "train.pt", weights_only=False)
        uns_train = torch.load(REPS / "unsw_nb15" / model / "train.pt", weights_only=False)
        # Pick a layer that exists in both
        common_layers = sorted(set(cic_train["hidden_states"].keys()) & set(uns_train["hidden_states"].keys()))

        for L in common_layers:
            X_cic = to_f32(cic_train["hidden_states"][L]).numpy()
            X_uns = to_f32(uns_train["hidden_states"][L]).numpy()
            if X_cic.shape[1] != X_uns.shape[1]:
                continue
            y_cic = cic_train["labels"].numpy()
            y_uns = uns_train["labels"].numpy()
            X = np.concatenate([X_cic, X_uns], axis=0)
            y = np.concatenate([y_cic, y_uns], axis=0)

            pipe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, n_jobs=-1, random_state=42))
            pipe.fit(X, y)

            rec = {}
            for ds_name, ds_split, src_train in [("cicids2017", cic_train, cic_train),
                                                  ("unsw_nb15", uns_train, uns_train)]:
                # Test on this dataset's holdout
                ho = torch.load(REPS / ds_name / model / "test_holdout.pt", weights_only=False)
                kn = torch.load(REPS / ds_name / model / "test_known.pt", weights_only=False)
                X_ho = to_f32(ho["hidden_states"][L]).numpy()
                X_kn = to_f32(kn["hidden_states"][L]).numpy()
                y_ho = ho["labels"].numpy()
                y_kn = kn["labels"].numpy()
                ho_auc = float(roc_auc_score(y_ho, pipe.predict_proba(X_ho)[:, 1]))
                kn_auc = float(roc_auc_score(y_kn, pipe.predict_proba(X_kn)[:, 1]))
                rec[ds_name] = {
                    "joint_known_auroc": kn_auc, "joint_holdout_auroc": ho_auc,
                    "joint_gap_auroc": kn_auc - ho_auc,
                }
            out[model][L] = rec
        # Pretty print last (deepest) layer
        if common_layers:
            L_last = common_layers[-1]
            print(f"\n  [{model}]  joint probe @ L{L_last}:")
            for ds, v in out[model][L_last].items():
                print(f"    {ds}: known_auroc={v['joint_known_auroc']:.3f}  holdout_auroc={v['joint_holdout_auroc']:.3f}  gap={v['joint_gap_auroc']:+.3f}")
    json.dump(out, open(OUT / "phase2_1c_joint_probe.json", "w"), indent=2)
    print("\n  saved phase2_1c_joint_probe.json")


def main():
    exp_1a()
    exp_1b()
    exp_1c()


if __name__ == "__main__":
    main()
