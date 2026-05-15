"""Build paper figures from JSON metric files.

Outputs to paper/figures/
"""
import json, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
METRICS = BASE / "results" / "metrics"
OUT = BASE / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# Fig 1: layer-dynamics — known vs holdout AUROC per layer for 4 models, 2 datasets
# ============================================================
def fig_layer_dynamics():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    models = ["Llama-3.1-8B-Instruct", "Mistral-7B-Instruct-v0.3", "Qwen3-8B", "gemma-2-9b-it"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for ax, ds in zip(axes, ["cicids2017", "unsw_nb15"]):
        for m, c in zip(models, colors):
            f = METRICS / f"{m}_{ds}.json"
            if not f.exists(): continue
            d = json.load(open(f))
            sweep = d["layer_sweep"]
            layers = sorted(int(l) for l in sweep.keys())
            cv = [sweep[str(l)]["cv_auroc"] for l in layers]
            short = m.replace("-Instruct", "").replace("Meta-", "").replace("v0.3", "")
            ax.plot(layers, cv, "o-", color=c, label=f"{short}", lw=1.5, markersize=5)
            best_l = d["best_layer"]
            ax.axvline(best_l, color=c, linestyle="--", alpha=0.25)
        ax.set_xlabel("Layer")
        ax.set_title(f"{'CIC-IDS2017' if ds=='cicids2017' else 'UNSW-NB15'}")
        ax.set_ylim(0.5, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Probe CV AUROC")
    axes[0].legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "fig_layer_dynamics.pdf", bbox_inches="tight")
    plt.savefig(OUT / "fig_layer_dynamics.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved fig_layer_dynamics.{pdf,png}")


# ============================================================
# Fig 2: cross-dataset transfer heatmap — best-layer cic2017→{cic18,iot23,ctu13} per model
# ============================================================
def fig_transfer_heatmap():
    d = None
    f = METRICS / "phase2_zero_shot_transfer.json"
    if not f.exists():
        print("skip transfer heatmap (no data)"); return
    d = json.load(open(f))
    # Keep only the 4 instruction-tuned 8-9B backbones used in the main text.
    KEEP = ["Llama-3.1-8B-Instruct", "Mistral-7B-Instruct-v0.3", "Qwen3-8B", "gemma-2-9b-it"]
    DISPLAY = {
        "Llama-3.1-8B-Instruct": "Llama-3.1-8B-Inst",
        "Mistral-7B-Instruct-v0.3": "Mistral-7B-Inst",
        "Qwen3-8B": "Qwen3-8B-Inst",
        "gemma-2-9b-it": "Gemma-2-9B-it",
    }
    models = [m for m in KEEP if m in d]
    if not models:
        print("skip transfer heatmap (no kept models)"); return
    datasets = ["cicids2018", "iot23", "ctu13"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 0.55 * max(len(models), 2) + 1.5),
                              gridspec_kw={"wspace": 0.18})
    in_dom = np.zeros((len(models), len(datasets)))
    transfer = np.zeros((len(models), len(datasets)))
    for i, m in enumerate(models):
        for j, ds in enumerate(datasets):
            r = d[m].get(ds, {}).get("per_layer", {})
            if not r: continue
            in_dom[i, j] = max(v["in_domain_probe_auroc"] for v in r.values())
            transfer[i, j] = max(v["cicids2017_dir_auroc"] for v in r.values())
    for k, (ax, mat, title) in enumerate([
        (axes[0], in_dom, "In-domain probe AUROC (upper bound)"),
        (axes[1], transfer, "CIC-2017 attack-direction transfer AUROC"),
    ]):
        im = ax.imshow(mat, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(
            [{"cicids2018": "CIC-2018", "iot23": "IoT-23", "ctu13": "CTU-13"}[d_] for d_ in datasets]
        )
        if k == 0:
            ax.set_yticks(range(len(models)))
            ax.set_yticklabels([DISPLAY.get(m, m) for m in models], fontsize=9)
        else:
            ax.set_yticks(range(len(models)))
            ax.set_yticklabels([""] * len(models))
        for i in range(len(models)):
            for j in range(len(datasets)):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        color="white" if mat[i, j] < 0.5 or mat[i, j] > 0.85 else "black",
                        fontsize=9)
        ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=axes, fraction=0.04, pad=0.02)
    plt.savefig(OUT / "fig_transfer_heatmap.pdf", bbox_inches="tight")
    plt.savefig(OUT / "fig_transfer_heatmap.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved fig_transfer_heatmap.{{pdf,png}} (models={[DISPLAY.get(m,m) for m in models]})")


# ============================================================
# Fig 3: V3 steering — yes/no logit gap vs alpha for one model (Qwen3-Base)
# ============================================================
def fig_steering():
    f = METRICS / "phase1_steering_v2_Qwen3-8B-Base_cicids2017.json"
    if not f.exists():
        print("skip steering (no data)"); return
    d = json.load(open(f))
    baseline = float(np.mean(d["baseline_yesno_diff"]))
    fig, axes = plt.subplots(1, len(d["target_layers"]), figsize=(15, 3.5),
                              sharey=True)
    if len(d["target_layers"]) == 1:
        axes = [axes]
    colors = {"cic_attack": "#e41a1c", "uns_attack": "#377eb8",
              "harmful": "#4daf4a", "random": "#999999"}
    for ax, L in zip(axes, d["target_layers"]):
        rec = d["results"][str(L)]
        for d_name, color in colors.items():
            if d_name not in rec: continue
            per_a = rec[d_name]
            alphas = sorted(float(a) for a in per_a.keys())
            means = [float(np.mean(per_a[str(a) if str(a) in per_a else f"{a:.1f}"])) for a in alphas]
            ax.plot(alphas, means, "o-", color=color, label=d_name, lw=1.4, markersize=5)
        ax.axhline(baseline, color="black", linestyle="--", alpha=0.4, label="baseline")
        ax.set_title(f"layer {L}")
        ax.set_xlabel(r"steering $\alpha$ (× layer norm)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"$\log p(yes) - \log p(no)$")
    axes[-1].legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(OUT / "fig_steering.pdf", bbox_inches="tight")
    plt.savefig(OUT / "fig_steering.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved fig_steering.{pdf,png}")


# ============================================================
# Fig 4: V1 TF-IDF vs LLM probe gap bar chart
# ============================================================
def fig_tfidf_vs_llm():
    f = METRICS / "phase1_expC_tfidf_baseline.json"
    if not f.exists():
        print("skip tfidf bar (no data)"); return
    d_tfidf = json.load(open(f))
    fig, ax = plt.subplots(figsize=(8, 3.5))
    methods = []
    cic_gaps = []
    uns_gaps = []
    # TF-IDF rows
    for vec_key, label in [("word_unigrams_bigrams", "TF-IDF word"), ("char_3to5", "TF-IDF char")]:
        methods.append(label)
        cic_gaps.append(d_tfidf["cicids2017"][vec_key]["gap_acc"])
        uns_gaps.append(d_tfidf["unsw_nb15"][vec_key]["gap_acc"])
    # LLM rows
    for m in ["Llama-3.1-8B-Instruct", "Mistral-7B-Instruct-v0.3", "Qwen3-8B", "gemma-2-9b-it"]:
        for ds, gaps in [("cicids2017", cic_gaps), ("unsw_nb15", uns_gaps)]:
            f2 = METRICS / f"{m}_{ds}.json"
            d2 = json.load(open(f2))
            gap = d2["known_accuracy"] - d2["holdout_accuracy"]
            if ds == "cicids2017":
                cic_gaps.append(gap)
            else:
                uns_gaps.append(gap)
        methods.append(m.replace("-Instruct", "").replace("v0.3", ""))
    x = np.arange(len(methods))
    w = 0.4
    ax.bar(x - w/2, cic_gaps, w, color="#e41a1c", label="CIC-IDS2017")
    ax.bar(x + w/2, uns_gaps, w, color="#377eb8", label="UNSW-NB15")
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Probe gap (known $-$ holdout acc)")
    ax.axhline(0, color="black", lw=0.6)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(OUT / "fig_tfidf_vs_llm.pdf", bbox_inches="tight")
    plt.savefig(OUT / "fig_tfidf_vs_llm.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved fig_tfidf_vs_llm.{pdf,png}")


def main():
    fig_layer_dynamics()
    fig_transfer_heatmap()
    fig_steering()
    fig_tfidf_vs_llm()
    print("\nAll figures written to", OUT)


if __name__ == "__main__":
    main()
