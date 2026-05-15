"""
Compile key Llama-3 results into a single PDF:
- Network Security (CICIDS2017): Mean Diff + LDA at 4 layers, train + holdout
- Harmful/Harmless: Mean Diff + LDA at 4 layers (non-gen) + generalization results
"""

import sys
sys.path.insert(0, '..')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.image import imread
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from pathlib import Path

from utils.probe_utils import extract_separation_direction, project_onto_direction

# ── paths ──
BASE = Path(__file__).resolve().parent.parent
FIGS = BASE / 'results' / 'figures'
REPS = BASE / 'results' / 'representations' / 'cicids2017' / 'Meta-Llama-3-8B-Instruct'
OUT_PDF = BASE / 'results' / 'llama3_results_summary.pdf'

MODEL = 'Meta-Llama-3-8B-Instruct'
LAYERS = [0, 8, 16, 31]  # 4 representative layers


def load_netsec_data():
    """Load cached network security representations."""
    train = torch.load(REPS / 'train.pt', weights_only=False)
    holdout = torch.load(REPS / 'test_holdout.pt', weights_only=False)
    known = torch.load(REPS / 'test_known.pt', weights_only=False)
    return train, known, holdout


def make_direction_hist(ax, proj_class0, proj_class1, title, label0='Normal', label1='Attack', show_acc=False):
    """Draw mean-diff direction histogram on given axes."""
    ax.hist(proj_class0, bins=50, alpha=0.6, label=label0, density=True, color='steelblue')
    ax.hist(proj_class1, bins=50, alpha=0.6, label=label1, density=True, color='darkorange')
    if show_acc:
        threshold = (proj_class0.mean() + proj_class1.mean()) / 2
        all_proj = np.concatenate([proj_class0, proj_class1])
        all_labels = np.concatenate([np.zeros(len(proj_class0)), np.ones(len(proj_class1))])
        pred = (all_proj > threshold).astype(int)
        acc = (pred == all_labels).mean()
        ax.text(0.02, 0.95, f'Acc: {acc:.3f}', transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.set_xlabel('Projection Value', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)


def make_lda_scatter(ax, X, y, title, label_names=None, fit_X=None, fit_y=None):
    """Draw LDA scatter on given axes. If fit_X/fit_y given, fit on that and transform X."""
    if label_names is None:
        label_names = {0: 'Normal', 1: 'Attack'}

    lda = LinearDiscriminantAnalysis(n_components=1)
    if fit_X is not None and fit_y is not None:
        lda.fit(fit_X, fit_y)
    else:
        lda.fit(X, y)

    X_lda_1d = lda.transform(X)
    pca = PCA(n_components=1)
    if fit_X is not None:
        pca.fit(fit_X)
    else:
        pca.fit(X)
    X_pca_1d = pca.transform(X)
    X_2d = np.column_stack([X_lda_1d, X_pca_1d])

    colors = {0: 'steelblue', 1: 'darkorange'}
    for label_val in sorted(np.unique(y)):
        mask = y == label_val
        name = label_names.get(label_val, str(label_val))
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], label=name, alpha=0.4, s=8,
                   color=colors.get(label_val, 'gray'))

    ax.set_xlabel('LDA Dim 1', fontsize=9)
    ax.set_ylabel('PCA Dim 1', fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, markerscale=3)


def add_existing_image(pdf, img_path, title=None):
    """Add an existing PNG image as a page in the PDF."""
    img = imread(str(img_path))
    h, w = img.shape[:2]
    aspect = w / h
    fig_w = 12
    fig_h = fig_w / aspect
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(img)
    ax.axis('off')
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def main():
    print("Loading network security data...")
    train, known, holdout = load_netsec_data()

    y_train = train['labels'].numpy()
    y_known = known['labels'].numpy()
    y_holdout = holdout['labels'].numpy()

    with PdfPages(str(OUT_PDF)) as pdf:
        # ════════════════════════════════════════════════
        # Title page
        # ════════════════════════════════════════════════
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        ax.text(0.5, 0.65, 'LLM Representation Analysis', fontsize=28,
                ha='center', va='center', fontweight='bold')
        ax.text(0.5, 0.50, 'Meta-Llama-3-8B-Instruct', fontsize=20,
                ha='center', va='center', color='gray')
        ax.text(0.5, 0.35, 'Network Security Logs + Harmful/Harmless Instructions',
                fontsize=16, ha='center', va='center')
        ax.text(0.5, 0.22, f'Layers analyzed: {LAYERS}',
                fontsize=14, ha='center', va='center', color='steelblue')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # ════════════════════════════════════════════════
        # SECTION 1: Network Security - Train (known attacks, no generalization)
        # ════════════════════════════════════════════════
        print("Generating Network Security - Train plots...")

        # Page: Mean Diff Direction - Train, 4 layers
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Network Security — Mean Diff Direction — Train (Known Attacks)',
                     fontsize=14, fontweight='bold')
        for idx, layer in enumerate(LAYERS):
            ax = axes[idx // 2, idx % 2]
            X = train['hidden_states'][layer]
            direction = extract_separation_direction(X, y_train)
            proj_normal = project_onto_direction(X[y_train == 0], direction)
            proj_attack = project_onto_direction(X[y_train == 1], direction)
            make_direction_hist(ax, proj_normal, proj_attack,
                              f'Layer {layer}', show_acc=True)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Page: LDA Scatter - Train, 4 layers
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Network Security — LDA Scatter — Train (Known Attacks)',
                     fontsize=14, fontweight='bold')
        for idx, layer in enumerate(LAYERS):
            ax = axes[idx // 2, idx % 2]
            X = train['hidden_states'][layer].numpy()
            make_lda_scatter(ax, X, y_train, f'Layer {layer}')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # ════════════════════════════════════════════════
        # SECTION 2: Network Security - Holdout (zero-day generalization)
        # ════════════════════════════════════════════════
        print("Generating Network Security - Holdout (generalization) plots...")

        # Page: Mean Diff Direction - Holdout, 4 layers
        # Direction is computed from TRAIN, projected onto HOLDOUT
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Network Security — Mean Diff Direction — Holdout (Zero-Day Generalization)',
                     fontsize=14, fontweight='bold')
        for idx, layer in enumerate(LAYERS):
            ax = axes[idx // 2, idx % 2]
            X_tr = train['hidden_states'][layer]
            X_ho = holdout['hidden_states'][layer]
            direction = extract_separation_direction(X_tr, y_train)
            proj_normal = project_onto_direction(X_ho[y_holdout == 0], direction)
            proj_attack = project_onto_direction(X_ho[y_holdout == 1], direction)
            make_direction_hist(ax, proj_normal, proj_attack,
                              f'Layer {layer} (direction from train)', show_acc=True)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Page: LDA Scatter - Holdout, 4 layers (LDA fit on train, transform holdout)
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Network Security — LDA Scatter — Holdout (Zero-Day Generalization)',
                     fontsize=14, fontweight='bold')
        for idx, layer in enumerate(LAYERS):
            ax = axes[idx // 2, idx % 2]
            X_tr = train['hidden_states'][layer].numpy()
            X_ho = holdout['hidden_states'][layer].numpy()
            make_lda_scatter(ax, X_ho, y_holdout, f'Layer {layer} (LDA fit on train)',
                           fit_X=X_tr, fit_y=y_train)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # ════════════════════════════════════════════════
        # SECTION 3: Harmful/Harmless - Non-generalization (sanity_v2)
        # ════════════════════════════════════════════════
        print("Adding Harmful/Harmless - Non-generalization plots...")

        # LDA at 4 layers (existing figures: layers 0, 12, 16, 31)
        hh_lda_layers = [0, 12, 16, 31]
        hh_lda_files = [FIGS / f'sanity_v2_lda_layer{l}.png' for l in hh_lda_layers]

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Harmful/Harmless — LDA Scatter — All Data (No Generalization Split)',
                     fontsize=14, fontweight='bold')
        for idx, (layer, fpath) in enumerate(zip(hh_lda_layers, hh_lda_files)):
            ax = axes[idx // 2, idx % 2]
            if fpath.exists():
                img = imread(str(fpath))
                ax.imshow(img)
                ax.set_title(f'Layer {layer}', fontsize=11)
            else:
                ax.text(0.5, 0.5, f'Layer {layer}\n(not available)', ha='center', va='center')
            ax.axis('off')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Direction histogram (only layer 12 available)
        hh_dir_path = FIGS / 'sanity_v2_direction_layer12.png'
        if hh_dir_path.exists():
            add_existing_image(pdf, hh_dir_path,
                             'Harmful/Harmless — Mean Diff Direction — Layer 12 (No Generalization Split)')

        # ════════════════════════════════════════════════
        # SECTION 4: Harmful/Harmless - Generalization
        # ════════════════════════════════════════════════
        print("Adding Harmful/Harmless - Generalization plots...")

        # Direction histograms: train vs holdout (layer 12)
        hg_dir_path = FIGS / 'harmful_generalization_directions_layer12.png'
        if hg_dir_path.exists():
            add_existing_image(pdf, hg_dir_path,
                             'Harmful/Harmless — Mean Diff + LDA Direction — Generalization (Layer 12)')

        # LDA scatter: train vs holdout (layer 12)
        hg_lda_path = FIGS / 'harmful_generalization_lda_scatter_layer12.png'
        if hg_lda_path.exists():
            add_existing_image(pdf, hg_lda_path,
                             'Harmful/Harmless — LDA Scatter — Generalization (Layer 12)')

        # ════════════════════════════════════════════════
        # SECTION 5: Layer accuracy curves
        # ════════════════════════════════════════════════
        print("Adding layer accuracy curves...")

        acc_path = FIGS / f'layer_accuracy_{MODEL}.png'
        if acc_path.exists():
            add_existing_image(pdf, acc_path, 'Network Security — Probe Accuracy Across Layers')

        auroc_path = FIGS / f'layer_auroc_{MODEL}.png'
        if auroc_path.exists():
            add_existing_image(pdf, auroc_path, 'Network Security — Probe AUROC Across Layers')

        hh_acc_path = FIGS / 'sanity_v2_layer_accuracy.png'
        if hh_acc_path.exists():
            add_existing_image(pdf, hh_acc_path, 'Harmful/Harmless — Probe Accuracy Across Layers')

    print(f"\nPDF saved to: {OUT_PDF}")


if __name__ == '__main__':
    main()
