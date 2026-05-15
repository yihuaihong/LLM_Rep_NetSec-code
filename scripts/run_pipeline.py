"""Unified per-model per-dataset pipeline.

Usage:
    PYTHONPATH=. python scripts/run_pipeline.py \
        --model Mistral-7B-Instruct-v0.3 --dataset unsw_nb15

Steps:
    1. Load + format dataset, build generalization splits
    2. Extract hidden states from a sweep of layers (proportional positions)
    3. Train logistic probe at each sampled layer, find best layer
    4. Evaluate best-layer probe on test_known and test_holdout
    5. Save metrics json + per-type projection figure
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from utils.data_utils import load_dataset, format_dataset, create_generalization_splits
from utils.model_utils import (
    load_model_and_tokenizer, extract_hidden_states, flush,
    set_random_seed, get_layer_count,
)
from utils.probe_utils import (
    train_logistic_probe, evaluate_probe,
    extract_separation_direction, project_onto_direction,
)
from utils.evaluation_utils import compute_generalization_gap

CACHE_DIR = "/scratch/yh6210/transformers"
DATASET_PATHS = {
    "unsw_nb15": "/scratch/yh6210/datasets/LLM_Rep_NetSec_datasets/unsw_nb15/",
    "cicids2017": "/scratch/yh6210/datasets/LLM_Rep_NetSec_datasets/cicids2017/",
}
HOLDOUTS = {
    "unsw_nb15": ["Shellcode", "Worms", "Backdoor"],
    "cicids2017": ["Bot", "Heartbleed", "Infiltration"],
}
MAX_SAMPLES = 5000
BATCH_SIZE = 16
MAX_SEQ_LEN = 512

BASE = Path(__file__).resolve().parent.parent


def pick_layers(num_layers: int, k: int = 9) -> list[int]:
    """Pick k roughly evenly-spaced layer indices including 0 and num_layers-1."""
    if num_layers <= k:
        return list(range(num_layers))
    pts = np.linspace(0, num_layers - 1, k).round().astype(int)
    return sorted(set(pts.tolist()))


def extract(model_name: str, dataset_name: str, reps_dir: Path, fmt: str = "natural_language"):
    split_files = {n: reps_dir / f"{n}.pt" for n in ("train", "test_known", "test_holdout")}
    if all(p.exists() for p in split_files.values()):
        print(f"[{model_name}/{dataset_name}] reps already cached, loading")
        return tuple(torch.load(split_files[n], weights_only=False)
                     for n in ("train", "test_known", "test_holdout"))

    missing = [n for n, p in split_files.items() if not p.exists()]
    print(f"[{model_name}/{dataset_name}] missing splits: {missing}")

    print(f"[{model_name}/{dataset_name}] loading dataset (fmt={fmt})...")
    df = load_dataset(dataset_name, DATASET_PATHS[dataset_name])
    df = format_dataset(df, dataset_name, fmt=fmt)
    splits = create_generalization_splits(
        df, holdout_attack_types=HOLDOUTS[dataset_name],
        max_samples_per_class=MAX_SAMPLES,
    )

    layers_json = reps_dir / "layers.json"
    layers, num_layers = None, None
    if layers_json.exists():
        info = json.loads(layers_json.read_text())
        layers, num_layers = info["layers"], info["num_layers"]
        print(f"[{model_name}] reusing cached layer choice: {layers}")

    model = tokenizer = None
    if missing:
        print(f"[{model_name}/{dataset_name}] loading model...")
        model_path = os.path.join(CACHE_DIR, model_name)
        dtype = "bfloat16" if "gemma" in model_name.lower() else "float16"
        model, tokenizer = load_model_and_tokenizer(model_path, dtype=dtype, device="cuda")
        if layers is None:
            num_layers = get_layer_count(model)
            print(f"[{model_name}] num_layers = {num_layers}")
            layers = pick_layers(num_layers, k=9)
            print(f"[{model_name}] sampled layers = {layers}")

    reps_dir.mkdir(parents=True, exist_ok=True)
    if layers is not None and not layers_json.exists():
        layers_json.write_text(json.dumps({"layers": layers, "num_layers": num_layers}))

    out = {}
    for name, sdf in splits.items():
        if split_files[name].exists():
            print(f"[{name}] cached, loading")
            out[name] = torch.load(split_files[name], weights_only=False)
            continue
        print(f"[{name}] extracting {len(sdf)} samples")
        hs = extract_hidden_states(
            model, tokenizer, sdf["text"].tolist(),
            layers=layers, token_position="last",
            batch_size=BATCH_SIZE, max_seq_length=MAX_SEQ_LEN,
        )
        save = {
            "hidden_states": hs,
            "labels": torch.tensor(sdf["is_attack"].values),
            "attack_types": sdf["attack_type"].values,
        }
        torch.save(save, split_files[name])
        out[name] = save
        flush()

    if not layers_json.exists() and layers is not None:
        layers_json.write_text(json.dumps({"layers": layers, "num_layers": num_layers}))

    if model is not None:
        del model, tokenizer
        flush()
    return out["train"], out["test_known"], out["test_holdout"]


def _to_np(t):
    # bfloat16 is unsupported by numpy; cast to float32 first.
    if t.dtype == torch.bfloat16:
        t = t.float()
    return t.numpy()


def evaluate(model_name, dataset_name, train, known, holdout, metrics_path, fig_path):
    y_train = train["labels"].numpy()
    y_known = known["labels"].numpy()
    y_holdout = holdout["labels"].numpy()
    types_holdout = np.array(holdout["attack_types"])

    layers = sorted(train["hidden_states"].keys())
    print(f"\n=== layer sweep ({model_name}/{dataset_name}) ===")
    sweep = {}
    for l in layers:
        X = _to_np(train["hidden_states"][l])
        r = train_logistic_probe(X, y_train)
        sweep[l] = {"cv_accuracy": r["cv_accuracy"], "cv_auroc": r["cv_auroc"]}
        print(f"  L{l:3d}: acc={r['cv_accuracy']:.4f} auroc={r['cv_auroc']:.4f}")

    best = max(sweep, key=lambda l: sweep[l]["cv_auroc"])
    X_tr = _to_np(train["hidden_states"][best])
    probe = train_logistic_probe(X_tr, y_train)
    clf = probe["model"]

    X_kn = _to_np(known["hidden_states"][best])
    X_ho = _to_np(holdout["hidden_states"][best])
    known_eval = evaluate_probe(clf, X_kn, y_known)
    holdout_eval = evaluate_probe(clf, X_ho, y_holdout)
    gap = compute_generalization_gap(known_eval, holdout_eval)

    H_tr = train["hidden_states"][best]
    H_ho = holdout["hidden_states"][best]
    if H_tr.dtype == torch.bfloat16:
        H_tr = H_tr.float()
    if H_ho.dtype == torch.bfloat16:
        H_ho = H_ho.float()
    direction = extract_separation_direction(H_tr, y_train)
    proj_n = project_onto_direction(H_tr[y_train == 0], direction)
    proj_a = project_onto_direction(H_tr[y_train == 1], direction)
    threshold = (proj_n.mean() + proj_a.mean()) / 2

    per_type = {}
    proj_ho = project_onto_direction(H_ho, direction)
    for t in np.unique(types_holdout):
        mask = types_holdout == t
        if mask.sum() == 0:
            continue
        true_label = int(y_holdout[mask][0] == 1)
        per_type[t] = {
            "n": int(mask.sum()),
            "true_label": true_label,
            "probe_acc": float((clf.predict(X_ho[mask]) == y_holdout[mask]).mean()),
            "direction_acc": float(((proj_ho[mask] > threshold).astype(int) == y_holdout[mask]).mean()),
            "projection_mean": float(np.nanmean(proj_ho[mask])),
        }

    print(f"\nbest_layer={best}")
    print(f"known: acc={known_eval['accuracy']:.4f} auroc={known_eval.get('auroc'):.4f}")
    print(f"holdout: acc={holdout_eval['accuracy']:.4f} auroc={holdout_eval.get('auroc'):.4f}")
    print(f"gap: acc={gap['accuracy_gap']:+.4f} auroc={gap['auroc_gap']:+.4f}")

    metrics = {
        "model": model_name, "dataset": dataset_name,
        "best_layer": int(best),
        "known_accuracy": known_eval["accuracy"],
        "known_auroc": known_eval.get("auroc"),
        "holdout_accuracy": holdout_eval["accuracy"],
        "holdout_auroc": holdout_eval.get("auroc"),
        "generalization_gap": gap,
        "layer_sweep": {int(l): v for l, v in sweep.items()},
        "per_type_holdout": per_type,
        "n_train": int(len(y_train)),
        "n_known": int(len(y_known)),
        "n_holdout": int(len(y_holdout)),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved {metrics_path}")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ls = sorted(sweep.keys())
    axes[0].plot(ls, [sweep[l]["cv_auroc"] for l in ls], "o-", color="darkorange", label="CV AUROC")
    axes[0].plot(ls, [sweep[l]["cv_accuracy"] for l in ls], "s--", color="steelblue", label="CV Acc")
    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("Score"); axes[0].grid(alpha=0.3)
    axes[0].set_title(f"{model_name} | {dataset_name} layer sweep"); axes[0].legend()

    types = list(per_type.keys())
    means = [per_type[t]["projection_mean"] for t in types]
    accs = [per_type[t]["direction_acc"] for t in types]
    ns = [per_type[t]["n"] for t in types]
    labels = [per_type[t]["true_label"] for t in types]
    order = np.argsort(means)
    types_o = [types[i] for i in order]
    means_o = [means[i] for i in order]
    colors = ["steelblue" if labels[i] == 0 else "darkorange" for i in order]
    ypos = np.arange(len(types_o))
    axes[1].barh(ypos, means_o, color=colors, alpha=0.8)
    for i, idx in enumerate(order):
        axes[1].text(means[idx], i, f"  acc={accs[idx]:.2f}, n={ns[idx]}", va="center", fontsize=8)
    axes[1].set_yticks(ypos); axes[1].set_yticklabels(types_o, fontsize=9)
    axes[1].axvline(x=0, color="gray", linestyle="--", alpha=0.4)
    axes[1].set_xlabel(f"Mean projection on attack direction (layer {best})")
    axes[1].set_title(f"{model_name} | {dataset_name} per-type holdout")
    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=130)
    plt.close(fig)
    print(f"saved {fig_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True, choices=list(DATASET_PATHS))
    ap.add_argument("--fmt", default="natural_language",
                    choices=["natural_language", "key_value"])
    args = ap.parse_args()

    set_random_seed(42)
    suffix = "" if args.fmt == "natural_language" else f"_{args.fmt}"
    reps_dir = BASE / "results" / "representations" / f"{args.dataset}{suffix}" / args.model
    metrics_path = BASE / "results" / "metrics" / f"{args.model}_{args.dataset}{suffix}.json"
    fig_path = BASE / "results" / "figures" / f"summary_{args.model}_{args.dataset}{suffix}.png"

    train, known, holdout = extract(args.model, args.dataset, reps_dir, fmt=args.fmt)
    evaluate(args.model, args.dataset, train, known, holdout, metrics_path, fig_path)
    print("DONE")


if __name__ == "__main__":
    main()
