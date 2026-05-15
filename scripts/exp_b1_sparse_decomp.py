"""B1: sparse decomposition of attack_direction into single-feature axes.

For each (model, dataset):
  1. Build feature axes from train normals: axis_f = mean(top-k by f) - mean(bot-k by f), normalized.
  2. Stack axes into A (n_features, hidden_dim).
  3. Fit non-negative-L2 (Lasso) regression: d ≈ A.T @ w  (constrained to find sparse w).
  4. Report: (a) reconstruction R^2 with all features (b) R^2 with top-k features (k=1, 2, 3, 5)
     (c) chosen sparse coefficients.

Interpretation:
  - If R^2 with k=1 is already > 0.5: the direction is essentially one feature.
  - If R^2 with k=3 ≈ R^2 with all: simple linear combo of 3 features.
  - If R^2 stays low even at k=full: the direction has hidden-state structure
    not captured by any numeric feature axis (genuine high-dim hash, harder to debunk).
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import Lasso, LinearRegression
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
CIC_AXES = [
    "Flow Bytes/s", "Flow Packets/s", "Flow Duration",
    "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd IAT Mean", "Bwd IAT Mean",
    "SYN Flag Count", "RST Flag Count", "ACK Flag Count", "PSH Flag Count",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "destination_port",
]
UNSW_AXES = [
    "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss",
    "spkts", "dpkts", "swin", "dwin", "smean", "dmean",
    "ct_srv_src", "ct_dst_sport_ltm", "destination_port",
]


def to_f32(t):
    return t.float() if t.dtype in (torch.bfloat16, torch.float16) else t


def best_layer(model, ds):
    f = OUT / f"{model}_{ds}.json"
    return int(json.load(open(f))["best_layer"]) if f.exists() else None


def build_axes_for_normal(H_normal, df_normal, axis_features, k=500):
    """Return dict feature -> normalized direction."""
    axes = {}
    for f in axis_features:
        if f not in df_normal.columns:
            continue
        vals = df_normal[f].values.astype(np.float64)
        valid = np.isfinite(vals)
        if valid.sum() < 100:
            continue
        H = H_normal[valid]; vals = vals[valid]
        kk = min(k, valid.sum() // 4)
        order = np.argsort(vals)
        bot = order[:kk]; top = order[-kk:]
        d = H[top].mean(0) - H[bot].mean(0)
        n = d.norm().item()
        if n < 1e-9:
            continue
        axes[f] = d / n
    return axes


def reconstruct_r2(d, A, weights):
    pred = A.T @ weights
    err = d - pred
    var_d = (d - d.mean()).pow(2).sum().item() + 1e-9
    var_err = err.pow(2).sum().item()
    return 1 - var_err / var_d


def run_one(model, ds, axis_features):
    bl = best_layer(model, ds)
    if bl is None:
        return None
    train = torch.load(REPS / ds / model / "train.pt", weights_only=False)
    H = to_f32(train["hidden_states"][bl])
    y = train["labels"].numpy()

    # Reproduce train df for raw numeric values (skip text formatting -- fast)
    df_full = load_dataset(ds, DATASET_PATHS[ds])
    splits = create_generalization_splits(
        df_full, holdout_attack_types=HOLDOUT[ds],
        max_samples_per_class=5000, seed=42,
    )
    train_df = splits["train"].reset_index(drop=True)
    if len(train_df) != H.shape[0]:
        print(f"  [{model}/{ds}] mismatch df={len(train_df)} reps={H.shape[0]}")
        return None

    # Attack direction
    d = extract_separation_direction(H, y).double()

    normal_mask = (y == 0)
    H_n = H[normal_mask].double()
    df_n = train_df.iloc[normal_mask].reset_index(drop=True)
    axes = build_axes_for_normal(H_n, df_n, axis_features, k=500)
    if not axes:
        return None
    feat_names = list(axes.keys())
    A = torch.stack([axes[f].double() for f in feat_names], dim=0)  # (n_features, hidden_dim)

    # Cosine of direction with each axis
    cos = {f: float(torch.nn.functional.cosine_similarity(d.unsqueeze(0), axes[f].double().unsqueeze(0)).item())
           for f in feat_names}

    # Linear fit: d ≈ A.T @ w (least squares)
    A_np = A.cpu().numpy().T   # (hidden_dim, n_features)
    d_np = d.cpu().numpy()
    lr = LinearRegression(fit_intercept=False).fit(A_np, d_np)
    full_pred = A_np @ lr.coef_
    r2_full = 1 - ((d_np - full_pred) ** 2).sum() / ((d_np - d_np.mean()) ** 2).sum()

    # Sparse fit (Lasso) — sweep alpha so that we control the sparsity
    sparse_results = {}
    for alpha in [0.001, 0.005, 0.01, 0.05]:
        ls = Lasso(alpha=alpha, fit_intercept=False, max_iter=20000).fit(A_np, d_np)
        nz = int((np.abs(ls.coef_) > 1e-6).sum())
        pred = A_np @ ls.coef_
        r2 = 1 - ((d_np - pred) ** 2).sum() / ((d_np - d_np.mean()) ** 2).sum()
        # which features got non-zero coefs
        kept = sorted([(feat_names[i], float(ls.coef_[i]))
                       for i in range(len(feat_names)) if abs(ls.coef_[i]) > 1e-6],
                      key=lambda x: -abs(x[1]))
        sparse_results[float(alpha)] = {"n_nonzero": nz, "r2": float(r2), "kept": kept}

    # Greedy top-k via cosine: take top-k features by |cos|, fit LR on those only
    cos_sorted = sorted(cos.items(), key=lambda x: -abs(x[1]))
    greedy = {}
    for k in [1, 2, 3, 5]:
        picks = [name for name, _ in cos_sorted[:k]]
        idx = [feat_names.index(p) for p in picks]
        A_sub = A_np[:, idx]
        lr_k = LinearRegression(fit_intercept=False).fit(A_sub, d_np)
        pred = A_sub @ lr_k.coef_
        r2 = 1 - ((d_np - pred) ** 2).sum() / ((d_np - d_np.mean()) ** 2).sum()
        greedy[int(k)] = {"features": picks, "coef": [float(c) for c in lr_k.coef_], "r2": float(r2)}

    return {
        "best_layer": bl,
        "n_features_total": len(feat_names),
        "feature_names": feat_names,
        "cosines": cos,
        "r2_with_all": float(r2_full),
        "lasso_sparse_results": sparse_results,
        "greedy_topk": greedy,
    }


def main():
    out = {}
    for model in ["Llama-3.1-8B-Instruct", "Qwen3-8B", "Mistral-7B-Instruct-v0.3", "gemma-2-9b-it"]:
        out[model] = {}
        for ds, axes in [("cicids2017", CIC_AXES), ("unsw_nb15", UNSW_AXES)]:
            try:
                r = run_one(model, ds, axes)
            except Exception as e:
                import traceback; traceback.print_exc()
                continue
            if r is None: continue
            out[model][ds] = r
            print(f"\n=== {model} / {ds} (layer {r['best_layer']}, n_features={r['n_features_total']}) ===")
            print(f"  R^2 with all features: {r['r2_with_all']:.4f}")
            print(f"  R^2 with top-k (greedy by |cos|):")
            for k, v in r['greedy_topk'].items():
                print(f"    k={k}: R^2={v['r2']:.4f}   features={v['features']}")
            print(f"  Lasso sparsity sweep:")
            for alpha, v in r['lasso_sparse_results'].items():
                kept_str = ", ".join(f"{n}({c:+.2f})" for n, c in v['kept'][:5])
                print(f"    α={alpha}: n_nz={v['n_nonzero']}  R^2={v['r2']:.4f}  top: {kept_str}")
    json.dump(out, open(OUT / "phase2_b1_sparse_decomp.json", "w"), indent=2)
    print("\nsaved -> phase2_b1_sparse_decomp.json")


if __name__ == "__main__":
    main()
