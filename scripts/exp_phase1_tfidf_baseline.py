"""Exp C: TF-IDF + logistic baseline.

Reproduce the SAME train/test_known/test_holdout split that the LLM probe used
(seeded by load_dataset → format_dataset → create_generalization_splits with
seed=42, max_samples_per_class=5000), then run TF-IDF vectorizer + logistic
regression. Compare known→holdout gap with the LLM probe gap.

If TF-IDF baseline gets the SAME big gap on CICIDS, it's strong evidence that
the LLM is not contributing semantic abstraction.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from collections import Counter

from utils.data_utils import load_dataset, format_dataset, create_generalization_splits

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "results" / "metrics"
OUT.mkdir(parents=True, exist_ok=True)

DATASET_PATHS = {
    "cicids2017": "/scratch/yh6210/datasets/LLM_Rep_NetSec_datasets/cicids2017/",
    "unsw_nb15":  "/scratch/yh6210/datasets/LLM_Rep_NetSec_datasets/unsw_nb15/",
}
HOLDOUT = {
    "cicids2017": ["Heartbleed", "Infiltration", "Bot"],
    "unsw_nb15":  ["Shellcode", "Worms", "Backdoor"],
}
MAX_PER_CLASS = 5000
SEED = 42


def per_type_breakdown(probs, preds, y, types):
    out = {}
    for t in np.unique(types):
        mask = types == t
        if mask.sum() == 0:
            continue
        out[t] = {
            "n": int(mask.sum()),
            "true_label": int(y[mask][0] == 1),
            "acc": float(accuracy_score(y[mask], preds[mask])),
            "mean_proba_attack": float(np.nanmean(probs[mask])),
        }
    return out


def run_dataset(name):
    print(f"\n{'='*60}\n  TF-IDF baseline: {name}\n{'='*60}", flush=True)
    df = load_dataset(name, DATASET_PATHS[name])
    df = format_dataset(df, name, fmt="natural_language")
    splits = create_generalization_splits(
        df,
        holdout_attack_types=HOLDOUT[name],
        max_samples_per_class=MAX_PER_CLASS,
        seed=SEED,
    )
    train, known, holdout = splits["train"], splits["test_known"], splits["test_holdout"]
    print(f"  train n={len(train)} (atk={train.is_attack.sum()})")
    print(f"  known n={len(known)} (atk={known.is_attack.sum()})")
    print(f"  holdout n={len(holdout)} (atk={holdout.is_attack.sum()})")

    # TF-IDF on word-level + char ngrams (2-5) — the latter helps capture numeric
    # token patterns like "9.6" vs "63411".
    # Run two configs: word-only (cleaner), and word + char (richer).
    results = {}
    for cfg_name, vec in [
        ("word_unigrams_bigrams", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=20000)),
        ("char_3to5",            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=50000)),
    ]:
        Xt = vec.fit_transform(train["text"].tolist())
        Xk = vec.transform(known["text"].tolist())
        Xh = vec.transform(holdout["text"].tolist())
        clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1).fit(Xt, train["is_attack"].values)

        pk = clf.predict(Xk); pk_p = clf.predict_proba(Xk)[:, 1]
        ph = clf.predict(Xh); ph_p = clf.predict_proba(Xh)[:, 1]

        ka = accuracy_score(known["is_attack"].values, pk)
        ku = roc_auc_score(known["is_attack"].values, pk_p)
        ha = accuracy_score(holdout["is_attack"].values, ph)
        hu = roc_auc_score(holdout["is_attack"].values, ph_p)

        per_type = per_type_breakdown(
            ph_p, ph,
            holdout["is_attack"].values,
            holdout["attack_type"].values,
        )

        rec = {
            "vectorizer": cfg_name,
            "n_features": int(Xt.shape[1]),
            "known_accuracy": float(ka),
            "known_auroc": float(ku),
            "holdout_accuracy": float(ha),
            "holdout_auroc": float(hu),
            "gap_acc": float(ka - ha),
            "gap_auroc": float(ku - hu),
            "per_type": per_type,
        }
        results[cfg_name] = rec
        print(f"\n  [{cfg_name}] features={Xt.shape[1]}")
        print(f"    known:   acc={ka:.4f}  auroc={ku:.4f}")
        print(f"    holdout: acc={ha:.4f}  auroc={hu:.4f}")
        print(f"    gap:     acc={ka-ha:+.4f} auroc={ku-hu:+.4f}")
        for t, v in per_type.items():
            print(f"      {t:25s} n={v['n']:>4} acc={v['acc']:.3f} mean_p_atk={v['mean_proba_attack']:.3f}")
    return results


def main():
    out = {}
    for name in ["cicids2017", "unsw_nb15"]:
        out[name] = run_dataset(name)
    json.dump(out, open(OUT / "phase1_expC_tfidf_baseline.json", "w"), indent=2)
    print("\nsaved -> phase1_expC_tfidf_baseline.json")


if __name__ == "__main__":
    main()
