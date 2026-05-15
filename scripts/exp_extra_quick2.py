"""More quick experiments using cached reps:

(5) Compute-aware probe — train probe with N={100, 500, 1000, 3000, all} train samples,
    see when probe accuracy plateaus. If small N already saturates → lexical hash.

(6) Cross-format probe transfer — train probe on natural_language reps, test on key_value reps
    (and vice versa). Tests format-invariance of probe direction.

(8) Cross-layer direction transfer — for each pair of layers (L, L'), compute
    cos(d_L, d_L') and use d_L to predict samples at L'. Tests layer interchangeability.

(9) TF-IDF cross-dataset transfer — fit TF-IDF + LR on cic2017 train, predict on
    cic2018 / iot23 / ctu13 raw text. Compares to V4 LLM result.
"""
import os, sys, json, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

from utils.probe_utils import extract_separation_direction, project_onto_direction
from utils.data_utils import load_dataset, format_dataset, create_generalization_splits

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


def best_layer(model, ds):
    f = OUT / f"{model}_{ds}.json"
    return int(json.load(open(f))["best_layer"]) if f.exists() else None


# ============================================================
# (5) Compute-aware probe sample-size curve
# ============================================================
def exp_5():
    print("\n=== (5) compute-aware probe sample-size curve ===")
    out = {}
    for model in ["Qwen3-8B-Base", "Llama-3.1-8B-Instruct"]:
        for ds in ["cicids2017", "unsw_nb15"]:
            bl = best_layer(model, ds)
            if bl is None: continue
            tr = torch.load(REPS / ds / model / "train.pt", weights_only=False)
            kn = torch.load(REPS / ds / model / "test_known.pt", weights_only=False)
            ho = torch.load(REPS / ds / model / "test_holdout.pt", weights_only=False)
            X_tr = to_f32(tr["hidden_states"][bl]).numpy()
            y_tr = tr["labels"].numpy()
            X_kn = to_f32(kn["hidden_states"][bl]).numpy()
            y_kn = kn["labels"].numpy()
            X_ho = to_f32(ho["hidden_states"][bl]).numpy()
            y_ho = ho["labels"].numpy()
            n_total = len(y_tr)
            print(f"\n  [{model}/{ds}] n_total={n_total}")
            print(f"    {'N':>5}  {'kn_auroc':>8} {'ho_auroc':>8} {'gap':>7}")
            sizes = [50, 100, 200, 500, 1000, 2000, 4000, n_total]
            sizes = sorted(set(s for s in sizes if s <= n_total))
            rec = []
            rng = np.random.RandomState(42)
            for N in sizes:
                # Stratified sample
                idx_pos = np.where(y_tr == 1)[0]
                idx_neg = np.where(y_tr == 0)[0]
                rng.shuffle(idx_pos); rng.shuffle(idx_neg)
                n_pos = min(N // 2, len(idx_pos))
                n_neg = min(N - n_pos, len(idx_neg))
                idx = np.concatenate([idx_pos[:n_pos], idx_neg[:n_neg]])
                X_sub = X_tr[idx]; y_sub = y_tr[idx]
                pipe = make_pipeline(StandardScaler(),
                                     LogisticRegression(C=1.0, max_iter=2000, n_jobs=-1, random_state=42))
                try:
                    pipe.fit(X_sub, y_sub)
                except Exception:
                    continue
                kn_auc = float(roc_auc_score(y_kn, pipe.predict_proba(X_kn)[:, 1]))
                ho_auc = float(roc_auc_score(y_ho, pipe.predict_proba(X_ho)[:, 1]))
                rec.append({"N": N, "kn_auroc": kn_auc, "ho_auroc": ho_auc,
                            "gap": kn_auc - ho_auc})
                print(f"    {N:>5}  {kn_auc:>8.3f} {ho_auc:>8.3f} {kn_auc-ho_auc:+7.3f}")
            out[f"{model}/{ds}"] = rec
    json.dump(out, open(OUT / "phase2_5_compute_aware_probe.json", "w"), indent=2)
    print("  saved phase2_5_compute_aware_probe.json")


# ============================================================
# (6) Cross-format probe transfer (Qwen3-Base)
# ============================================================
def exp_6():
    print("\n=== (6) cross-format probe transfer (Qwen3-Base) ===")
    out = {}
    model = "Qwen3-8B-Base"
    for ds in ["cicids2017", "unsw_nb15"]:
        nat = REPS / ds / model
        kv  = REPS / f"{ds}_key_value" / model
        if not (nat / "train.pt").exists() or not (kv / "train.pt").exists():
            continue
        tr_n = torch.load(nat / "train.pt", weights_only=False)
        tr_k = torch.load(kv / "train.pt", weights_only=False)
        ho_n = torch.load(nat / "test_holdout.pt", weights_only=False)
        ho_k = torch.load(kv / "test_holdout.pt", weights_only=False)
        kn_n = torch.load(nat / "test_known.pt", weights_only=False)
        kn_k = torch.load(kv / "test_known.pt", weights_only=False)

        layers = sorted(set(tr_n["hidden_states"].keys()) & set(tr_k["hidden_states"].keys()))
        rec = {}
        for L in layers:
            X_tn = to_f32(tr_n["hidden_states"][L]).numpy()
            X_tk = to_f32(tr_k["hidden_states"][L]).numpy()
            X_kn_n = to_f32(kn_n["hidden_states"][L]).numpy()
            X_ho_n = to_f32(ho_n["hidden_states"][L]).numpy()
            X_kn_k = to_f32(kn_k["hidden_states"][L]).numpy()
            X_ho_k = to_f32(ho_k["hidden_states"][L]).numpy()
            y_tr = tr_n["labels"].numpy()
            y_kn = kn_n["labels"].numpy()
            y_ho = ho_n["labels"].numpy()

            # Note: train set is the same logical sample but different formats -> labels match
            # Train probe on natural, test on key_value
            pipe_n = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, n_jobs=-1, random_state=42))
            pipe_n.fit(X_tn, y_tr)
            kn_n_test = float(roc_auc_score(y_kn, pipe_n.predict_proba(X_kn_n)[:, 1]))
            ho_n_test = float(roc_auc_score(y_ho, pipe_n.predict_proba(X_ho_n)[:, 1]))
            kn_k_test = float(roc_auc_score(y_kn, pipe_n.predict_proba(X_kn_k)[:, 1]))
            ho_k_test = float(roc_auc_score(y_ho, pipe_n.predict_proba(X_ho_k)[:, 1]))

            # Train on key_value, test on natural
            pipe_k = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, n_jobs=-1, random_state=42))
            pipe_k.fit(X_tk, y_tr)
            kn_k_self = float(roc_auc_score(y_kn, pipe_k.predict_proba(X_kn_k)[:, 1]))
            ho_k_self = float(roc_auc_score(y_ho, pipe_k.predict_proba(X_ho_k)[:, 1]))
            kn_n_xfer = float(roc_auc_score(y_kn, pipe_k.predict_proba(X_kn_n)[:, 1]))
            ho_n_xfer = float(roc_auc_score(y_ho, pipe_k.predict_proba(X_ho_n)[:, 1]))

            rec[int(L)] = {
                "nat_self_kn":  kn_n_test, "nat_self_ho":  ho_n_test,
                "nat_xfer_kn":  kn_k_test, "nat_xfer_ho":  ho_k_test,
                "kv_self_kn":   kn_k_self, "kv_self_ho":   ho_k_self,
                "kv_xfer_kn":   kn_n_xfer, "kv_xfer_ho":   ho_n_xfer,
            }
        # Print best layer
        if rec:
            best_L = max(rec, key=lambda L: rec[L]["nat_self_kn"])
            v = rec[best_L]
            print(f"\n  [{ds}] best L={best_L}")
            print(f"    nat→nat:  known={v['nat_self_kn']:.3f}  holdout={v['nat_self_ho']:.3f}")
            print(f"    nat→kv:   known={v['nat_xfer_kn']:.3f}  holdout={v['nat_xfer_ho']:.3f}")
            print(f"    kv→kv:    known={v['kv_self_kn']:.3f}  holdout={v['kv_self_ho']:.3f}")
            print(f"    kv→nat:   known={v['kv_xfer_kn']:.3f}  holdout={v['kv_xfer_ho']:.3f}")
        out[ds] = rec
    json.dump(out, open(OUT / "phase2_6_cross_format_probe.json", "w"), indent=2)
    print("  saved phase2_6_cross_format_probe.json")


# ============================================================
# (8) Cross-layer direction transfer (matrix)
# ============================================================
def exp_8():
    print("\n=== (8) cross-layer direction transfer ===")
    out = {}
    for model in ["Qwen3-8B-Base", "Llama-3.1-8B-Instruct"]:
        for ds in ["cicids2017"]:
            tr = torch.load(REPS / ds / model / "train.pt", weights_only=False)
            ho = torch.load(REPS / ds / model / "test_holdout.pt", weights_only=False)
            layers = sorted(tr["hidden_states"].keys())
            y_tr = tr["labels"].numpy()
            y_ho = ho["labels"].numpy()

            # Build all directions
            dirs = {}
            for L in layers:
                X = to_f32(tr["hidden_states"][L])
                dirs[L] = extract_separation_direction(X, y_tr)

            # For each (L_train, L_test) pair: project test reps at L_test through dir from L_train
            # AUROC on holdout
            mat = np.zeros((len(layers), len(layers)))
            cos_mat = np.zeros((len(layers), len(layers)))
            for i, L1 in enumerate(layers):
                d1 = dirs[L1]
                for j, L2 in enumerate(layers):
                    H = to_f32(ho["hidden_states"][L2])
                    proj = project_onto_direction(H, d1)
                    try:
                        auc = float(roc_auc_score(y_ho, proj))
                    except Exception:
                        auc = 0.5
                    mat[i, j] = auc
                    cos_mat[i, j] = cos(dirs[L1], dirs[L2])
            print(f"\n  [{model}/{ds}]  layers: {layers}")
            print("  AUROC matrix (L_train rows × L_test cols, holdout):")
            print("  " + " ".join(f"{l:>5}" for l in layers))
            for i, L in enumerate(layers):
                row = " ".join(f"{mat[i,j]:>5.2f}" for j in range(len(layers)))
                print(f"  {L:>2}  {row}")
            out[f"{model}/{ds}"] = {
                "layers": layers,
                "auroc_matrix": mat.tolist(),
                "cos_matrix": cos_mat.tolist(),
            }
    json.dump(out, open(OUT / "phase2_8_cross_layer.json", "w"), indent=2)
    print("  saved phase2_8_cross_layer.json")


# ============================================================
# (9) TF-IDF cross-dataset transfer
# ============================================================
def exp_9():
    print("\n=== (9) TF-IDF cross-dataset transfer ===")
    # Source: cicids2017 train (with current text)
    df_src = load_dataset("cicids2017", DATASET_PATHS["cicids2017"])
    df_src = format_dataset(df_src, "cicids2017", fmt="natural_language")
    splits_src = create_generalization_splits(
        df_src, holdout_attack_types=HOLDOUT["cicids2017"],
        max_samples_per_class=5000, seed=42,
    )
    train_text = splits_src["train"]["text"].tolist()
    train_y = splits_src["train"]["is_attack"].values

    # Fit TF-IDF + LR on cicids2017 train
    print("  fitting TF-IDF on cicids2017 train ...")
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=20000)
    X_tr_v = vec.fit_transform(train_text)
    pipe = LogisticRegression(C=1.0, max_iter=2000, n_jobs=-1, random_state=42).fit(X_tr_v, train_y)

    # Source self-test
    kn_text = splits_src["test_known"]["text"].tolist()
    kn_y = splits_src["test_known"]["is_attack"].values
    kn_auc = roc_auc_score(kn_y, pipe.predict_proba(vec.transform(kn_text))[:, 1])
    ho_text = splits_src["test_holdout"]["text"].tolist()
    ho_y = splits_src["test_holdout"]["is_attack"].values
    ho_auc = roc_auc_score(ho_y, pipe.predict_proba(vec.transform(ho_text))[:, 1])
    print(f"    cic2017 self-test:  known={kn_auc:.3f}  holdout={ho_auc:.3f}")

    out = {"source": "cicids2017", "vectorizer": "word_1-2grams",
           "self_known_auroc": float(kn_auc), "self_holdout_auroc": float(ho_auc),
           "cross_dataset": {}}

    # Now test on the new datasets (we need their texts)
    # For cicids2018, iot23, ctu13: use the zero_shot.pt's "texts" field
    for new_ds in ["cicids2018", "iot23", "ctu13"]:
        zs_path = REPS / new_ds / "Qwen3-8B-Base" / "zero_shot.pt"
        if not zs_path.exists():
            print(f"    [{new_ds}] no zero_shot.pt"); continue
        zs = torch.load(zs_path, weights_only=False)
        texts = zs["texts"]
        y = zs["labels"].numpy()
        Xv = vec.transform(texts)
        try:
            auc = float(roc_auc_score(y, pipe.predict_proba(Xv)[:, 1]))
        except Exception as e:
            print(f"    [{new_ds}] err {e}"); continue
        print(f"    cic2017 → {new_ds}:  AUROC={auc:.3f}  (n={len(y)}, n_atk={int(y.sum())})")
        out["cross_dataset"][new_ds] = {"n": int(len(y)), "n_attack": int(y.sum()),
                                          "auroc": auc}
    json.dump(out, open(OUT / "phase2_9_tfidf_transfer.json", "w"), indent=2)
    print("  saved phase2_9_tfidf_transfer.json")


def main():
    exp_5()
    exp_6()
    exp_8()
    exp_9()


if __name__ == "__main__":
    main()
