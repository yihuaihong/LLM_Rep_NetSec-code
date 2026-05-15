"""
PDF v2: Unified 4-layer (0/12/16/31) results for both domains
on Meta-Llama-3-8B-Instruct.

Each section: 2x2 panel of layers, generated freshly from cached representations.
- Network Security  (CICIDS2017)  — train + holdout × {Mean Diff, LDA}
- Harmful/Harmless  (JBB)         — train + holdout × {Mean Diff, LDA}

Cached representations required:
  results/representations/cicids2017/Meta-Llama-3-8B-Instruct/{train,test_known,test_holdout}.pt
  results/representations/jbb/Meta-Llama-3-8B-Instruct/all.pt
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from utils.probe_utils import extract_separation_direction, project_onto_direction

# ── config ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
NETSEC_REPS = BASE / "results" / "representations" / "cicids2017" / "Meta-Llama-3-8B-Instruct"
JBB_REPS    = BASE / "results" / "representations" / "jbb" / "Meta-Llama-3-8B-Instruct"
OUT_PDF     = BASE / "results" / "llama3_results_unified_v2.pdf"

LAYERS = [0, 12, 16, 31]
HOLDOUT_HARMFUL_CATS = {"Malware/Hacking", "Physical harm", "Sexual/Adult content"}

# ── plotting helpers ────────────────────────────────────────────────────────
def hist_two_classes(ax, p0, p1, title, label0, label1, color0="steelblue", color1="darkorange"):
    ax.hist(p0, bins=40, alpha=0.6, label=label0, density=True, color=color0)
    ax.hist(p1, bins=40, alpha=0.6, label=label1, density=True, color=color1)
    thr = (p0.mean() + p1.mean()) / 2
    pred = np.concatenate([p0, p1]) > thr
    truth = np.concatenate([np.zeros_like(p0), np.ones_like(p1)])
    acc = float(((pred == truth.astype(bool)).mean()))
    ax.text(0.02, 0.95, f"acc={acc:.3f}", transform=ax.transAxes,
            fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    ax.set_xlabel("projection", fontsize=9)
    ax.set_ylabel("density", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)


def lda_scatter(ax, X_view, y_view, title, label0, label1,
                X_fit=None, y_fit=None, normalize=True,
                color0="steelblue", color1="darkorange"):
    """LDA fit on (X_fit,y_fit) — falls back to (X_view,y_view) — then transform X_view."""
    Xf, yf = (X_fit, y_fit) if X_fit is not None else (X_view, y_view)

    if normalize:
        scaler = StandardScaler().fit(Xf)
        Xf  = scaler.transform(Xf)
        Xv  = scaler.transform(X_view)
    else:
        Xv = X_view

    lda = LinearDiscriminantAnalysis(n_components=1).fit(Xf, yf)
    pca = PCA(n_components=1).fit(Xf)
    coords = np.column_stack([lda.transform(Xv), pca.transform(Xv)])

    for cls, name, color in [(0, label0, color0), (1, label1, color1)]:
        mask = y_view == cls
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   alpha=0.45, s=10, label=name, color=color)
    ax.set_xlabel("LDA dim 1", fontsize=9)
    ax.set_ylabel("PCA dim 1", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, markerscale=2)


def panel_2x2(layers, title, render_one):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    for idx, layer in enumerate(layers):
        ax = axes[idx // 2, idx % 2]
        render_one(ax, layer)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def text_page(lines, title=None):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    if title:
        ax.text(0.5, 0.95, title, fontsize=18, fontweight="bold",
                ha="center", va="top")
    ax.text(0.05, 0.85, "\n".join(lines), fontsize=11, ha="left", va="top",
            family="monospace")
    return fig


# ── data prep ───────────────────────────────────────────────────────────────
def load_netsec():
    train   = torch.load(NETSEC_REPS / "train.pt", weights_only=False)
    known   = torch.load(NETSEC_REPS / "test_known.pt", weights_only=False)
    holdout = torch.load(NETSEC_REPS / "test_holdout.pt", weights_only=False)
    return train, known, holdout


def load_jbb():
    return torch.load(JBB_REPS / "all.pt", weights_only=False)


def split_jbb(jbb):
    """Split JBB into train (7 categories of harmful + matching benign) and holdout (3 categories)."""
    cats = jbb["categories"]
    labels = jbb["labels"].numpy() if isinstance(jbb["labels"], torch.Tensor) else jbb["labels"]

    is_harmful = labels == 1
    holdout_mask = np.array([c in HOLDOUT_HARMFUL_CATS for c in cats]) & is_harmful

    rng = np.random.default_rng(42)
    benign_idx = np.where(~is_harmful)[0]
    rng.shuffle(benign_idx)
    n_holdout = int(holdout_mask.sum())
    benign_holdout = benign_idx[:n_holdout]
    benign_train   = benign_idx[n_holdout:]

    harmful_train_mask = is_harmful & ~holdout_mask
    train_idx = np.concatenate([np.where(harmful_train_mask)[0], benign_train])
    holdout_idx = np.concatenate([np.where(holdout_mask)[0], benign_holdout])
    rng.shuffle(train_idx)
    rng.shuffle(holdout_idx)
    return train_idx, holdout_idx


def layer_arr(rep_dict, layer):
    """Get hidden state at a layer as float32 numpy."""
    return rep_dict["hidden_states"][layer].float().numpy()


# ── metrics summary ─────────────────────────────────────────────────────────
def cv_acc_auroc(X, y, seed=42):
    """5-fold CV logistic regression returning (acc, auroc)."""
    Xs = StandardScaler().fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs, aurocs = [], []
    for tr, te in cv.split(Xs, y):
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xs[tr], y[tr])
        p = clf.predict(Xs[te])
        accs.append((p == y[te]).mean())
        aurocs.append(np.nan if len(np.unique(y[te])) < 2
                      else float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(
                          y[te], clf.predict_proba(Xs[te])[:, 1])))
    return float(np.mean(accs)), float(np.mean(aurocs))


def fit_eval(X_train, y_train, X_test, y_test):
    """Fit LR on (X_train,y_train) standardized, evaluate on (X_test,y_test)."""
    from sklearn.metrics import roc_auc_score
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xte = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, y_train)
    pred = clf.predict(Xte)
    acc  = float((pred == y_test).mean())
    try:
        auroc = float(roc_auc_score(y_test, clf.predict_proba(Xte)[:, 1]))
    except Exception:
        auroc = float("nan")
    return acc, auroc


# ── main ────────────────────────────────────────────────────────────────────
def main():
    print("Loading representations...")
    ns_train, ns_known, ns_holdout = load_netsec()
    jbb = load_jbb()

    y_ns_train   = ns_train["labels"].numpy()
    y_ns_known   = ns_known["labels"].numpy()
    y_ns_holdout = ns_holdout["labels"].numpy()
    y_jbb        = jbb["labels"].numpy() if isinstance(jbb["labels"], torch.Tensor) else jbb["labels"]
    jbb_tr_idx, jbb_ho_idx = split_jbb(jbb)
    print(f"  netsec train={len(y_ns_train)} known={len(y_ns_known)} holdout={len(y_ns_holdout)}")
    print(f"  jbb total={len(y_jbb)}  train={len(jbb_tr_idx)} holdout={len(jbb_ho_idx)}")

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(OUT_PDF)) as pdf:
        # ── title ──
        fig = text_page([
            "Domains:",
            "  • Network Security  (CICIDS2017)",
            "  • Harmful / Harmless  (JBB-Behaviors)",
            "",
            f"Layers shown: {LAYERS}",
            "",
            "For each domain we present:",
            "  • Mean-Diff direction histograms   ×  {train, holdout}",
            "  • LDA scatter (LDA dim vs PCA dim) ×  {train, holdout}",
            "",
            "Holdout = generalization split:",
            "  • NetSec : holdout attack types = Heartbleed, Infiltration, Bot",
            "  • Harmful: holdout categories  = Malware/Hacking, Physical harm, Sexual/Adult content",
        ], title="LLM Representation Analysis  —  Meta-Llama-3-8B-Instruct")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ════════════════════════════════════════════════════════════════════
        # NETSEC — Mean Diff
        # ════════════════════════════════════════════════════════════════════
        def render_ns_meandiff_train(ax, layer):
            X = ns_train["hidden_states"][layer]
            d = extract_separation_direction(X, y_ns_train)
            p0 = project_onto_direction(X[y_ns_train == 0], d)
            p1 = project_onto_direction(X[y_ns_train == 1], d)
            hist_two_classes(ax, p0, p1, f"layer {layer}", "Normal", "Attack")

        def render_ns_meandiff_holdout(ax, layer):
            Xtr = ns_train["hidden_states"][layer]
            Xho = ns_holdout["hidden_states"][layer]
            d = extract_separation_direction(Xtr, y_ns_train)
            p0 = project_onto_direction(Xho[y_ns_holdout == 0], d)
            p1 = project_onto_direction(Xho[y_ns_holdout == 1], d)
            hist_two_classes(ax, p0, p1, f"layer {layer} (direction from train)",
                             "Normal", "Attack (unseen types)")

        pdf.savefig(panel_2x2(LAYERS,
            "NetSec — Mean-Diff direction — TRAIN (in-distribution)",
            render_ns_meandiff_train), bbox_inches="tight"); plt.close()
        pdf.savefig(panel_2x2(LAYERS,
            "NetSec — Mean-Diff direction — HOLDOUT (zero-day generalization)",
            render_ns_meandiff_holdout), bbox_inches="tight"); plt.close()

        # ════════════════════════════════════════════════════════════════════
        # NETSEC — LDA
        # ════════════════════════════════════════════════════════════════════
        def render_ns_lda_train(ax, layer):
            X = layer_arr(ns_train, layer)
            lda_scatter(ax, X, y_ns_train, f"layer {layer}", "Normal", "Attack")

        def render_ns_lda_holdout(ax, layer):
            Xtr = layer_arr(ns_train, layer)
            Xho = layer_arr(ns_holdout, layer)
            lda_scatter(ax, Xho, y_ns_holdout,
                        f"layer {layer} (LDA fit on train)",
                        "Normal", "Attack (unseen types)",
                        X_fit=Xtr, y_fit=y_ns_train)

        pdf.savefig(panel_2x2(LAYERS,
            "NetSec — LDA scatter — TRAIN (in-distribution)",
            render_ns_lda_train), bbox_inches="tight"); plt.close()
        pdf.savefig(panel_2x2(LAYERS,
            "NetSec — LDA scatter — HOLDOUT (zero-day generalization)",
            render_ns_lda_holdout), bbox_inches="tight"); plt.close()

        # ════════════════════════════════════════════════════════════════════
        # HARMFUL/HARMLESS — Mean Diff
        # ════════════════════════════════════════════════════════════════════
        def render_jbb_meandiff_train(ax, layer):
            X = jbb["hidden_states"][layer][jbb_tr_idx]
            y = y_jbb[jbb_tr_idx]
            d = extract_separation_direction(X, y)
            p0 = project_onto_direction(X[y == 0], d)
            p1 = project_onto_direction(X[y == 1], d)
            hist_two_classes(ax, p0, p1, f"layer {layer}", "Benign", "Harmful",
                             color0="seagreen", color1="firebrick")

        def render_jbb_meandiff_holdout(ax, layer):
            Xtr = jbb["hidden_states"][layer][jbb_tr_idx]
            ytr = y_jbb[jbb_tr_idx]
            Xho = jbb["hidden_states"][layer][jbb_ho_idx]
            yho = y_jbb[jbb_ho_idx]
            d = extract_separation_direction(Xtr, ytr)
            p0 = project_onto_direction(Xho[yho == 0], d)
            p1 = project_onto_direction(Xho[yho == 1], d)
            hist_two_classes(ax, p0, p1, f"layer {layer} (direction from train cats)",
                             "Benign", "Harmful (unseen cats)",
                             color0="seagreen", color1="firebrick")

        pdf.savefig(panel_2x2(LAYERS,
            "Harmful/Harmless — Mean-Diff direction — TRAIN (7 categories)",
            render_jbb_meandiff_train), bbox_inches="tight"); plt.close()
        pdf.savefig(panel_2x2(LAYERS,
            "Harmful/Harmless — Mean-Diff direction — HOLDOUT (3 unseen categories)",
            render_jbb_meandiff_holdout), bbox_inches="tight"); plt.close()

        # ════════════════════════════════════════════════════════════════════
        # HARMFUL/HARMLESS — LDA
        # ════════════════════════════════════════════════════════════════════
        def render_jbb_lda_train(ax, layer):
            X = layer_arr(jbb, layer)[jbb_tr_idx]
            y = y_jbb[jbb_tr_idx]
            lda_scatter(ax, X, y, f"layer {layer}", "Benign", "Harmful",
                        color0="seagreen", color1="firebrick")

        def render_jbb_lda_holdout(ax, layer):
            Xtr = layer_arr(jbb, layer)[jbb_tr_idx]
            Xho = layer_arr(jbb, layer)[jbb_ho_idx]
            ytr = y_jbb[jbb_tr_idx]
            yho = y_jbb[jbb_ho_idx]
            lda_scatter(ax, Xho, yho,
                        f"layer {layer} (LDA fit on train cats)",
                        "Benign", "Harmful (unseen cats)",
                        X_fit=Xtr, y_fit=ytr,
                        color0="seagreen", color1="firebrick")

        pdf.savefig(panel_2x2(LAYERS,
            "Harmful/Harmless — LDA scatter — TRAIN (7 categories)",
            render_jbb_lda_train), bbox_inches="tight"); plt.close()
        pdf.savefig(panel_2x2(LAYERS,
            "Harmful/Harmless — LDA scatter — HOLDOUT (3 unseen categories)",
            render_jbb_lda_holdout), bbox_inches="tight"); plt.close()

        # ════════════════════════════════════════════════════════════════════
        # Summary: probe accuracy across layers (train vs holdout) for both
        # ════════════════════════════════════════════════════════════════════
        print("Computing probe accuracies across all layers (this is fast)...")
        ns_layers = sorted(ns_train["hidden_states"].keys())
        jbb_layers = sorted(jbb["hidden_states"].keys())

        ns_curve = {}
        for L in ns_layers:
            Xtr = layer_arr(ns_train, L)
            Xho = layer_arr(ns_holdout, L)
            tr_acc, tr_auroc = cv_acc_auroc(Xtr, y_ns_train)
            ho_acc, ho_auroc = fit_eval(Xtr, y_ns_train, Xho, y_ns_holdout)
            ns_curve[L] = (tr_acc, ho_acc)

        jbb_curve = {}
        Xall_jbb_tr = {L: layer_arr(jbb, L)[jbb_tr_idx] for L in jbb_layers}
        Xall_jbb_ho = {L: layer_arr(jbb, L)[jbb_ho_idx] for L in jbb_layers}
        for L in jbb_layers:
            tr_acc, _ = cv_acc_auroc(Xall_jbb_tr[L], y_jbb[jbb_tr_idx])
            ho_acc, _ = fit_eval(Xall_jbb_tr[L], y_jbb[jbb_tr_idx],
                                 Xall_jbb_ho[L], y_jbb[jbb_ho_idx])
            jbb_curve[L] = (tr_acc, ho_acc)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        L1 = sorted(ns_curve)
        axes[0].plot(L1, [ns_curve[l][0] for l in L1], 'o-', label="train (CV)", color="steelblue")
        axes[0].plot(L1, [ns_curve[l][1] for l in L1], 's--', label="holdout", color="firebrick")
        for l in LAYERS:
            axes[0].axvline(l, color="gray", linestyle=":", alpha=0.5)
        axes[0].set_xlabel("layer"); axes[0].set_ylabel("accuracy"); axes[0].set_ylim(0.3, 1.05)
        axes[0].set_title("NetSec — probe accuracy"); axes[0].legend(); axes[0].grid(alpha=0.3)

        L2 = sorted(jbb_curve)
        axes[1].plot(L2, [jbb_curve[l][0] for l in L2], 'o-', label="train (CV)", color="seagreen")
        axes[1].plot(L2, [jbb_curve[l][1] for l in L2], 's--', label="holdout", color="firebrick")
        for l in LAYERS:
            axes[1].axvline(l, color="gray", linestyle=":", alpha=0.5)
        axes[1].set_xlabel("layer"); axes[1].set_ylabel("accuracy"); axes[1].set_ylim(0.3, 1.05)
        axes[1].set_title("Harmful/Harmless — probe accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)

        fig.suptitle("Probe accuracy across all layers  (dotted = layers shown above)",
                     fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── final summary table ──
        lines = [f"{'Layer':>6}  {'NetSec train':>14}  {'NetSec hold':>13}  "
                 f"{'JBB train':>11}  {'JBB hold':>10}"]
        lines.append("-" * 65)
        for L in LAYERS:
            ns_t, ns_h = ns_curve.get(L, (np.nan, np.nan))
            jb_t, jb_h = jbb_curve.get(L, (np.nan, np.nan))
            lines.append(f"{L:>6}  {ns_t:>14.3f}  {ns_h:>13.3f}  {jb_t:>11.3f}  {jb_h:>10.3f}")
        lines.append("")
        lines.append("Generalization gap (train − holdout):")
        for L in LAYERS:
            ns_t, ns_h = ns_curve.get(L, (0, 0))
            jb_t, jb_h = jbb_curve.get(L, (0, 0))
            lines.append(f"  layer {L:>2}:  NetSec gap = {ns_t-ns_h:+.3f}    "
                         f"JBB gap = {jb_t-jb_h:+.3f}")
        fig = text_page(lines, title="Summary  —  probe accuracy at the 4 chosen layers")
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

    # save curves to JSON for reuse
    out_metrics = BASE / "results" / "metrics" / "v2_curves.json"
    out_metrics.write_text(json.dumps({
        "netsec": {str(k): {"train": v[0], "holdout": v[1]} for k, v in ns_curve.items()},
        "jbb":    {str(k): {"train": v[0], "holdout": v[1]} for k, v in jbb_curve.items()},
        "layers_shown": LAYERS,
    }, indent=2))
    print(f"\nMetrics  -> {out_metrics}")
    print(f"PDF v2  -> {OUT_PDF}")


if __name__ == "__main__":
    main()
