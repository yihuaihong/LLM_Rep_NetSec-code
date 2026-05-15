"""
Generate a 2x2 generalization test figure for UNSW-NB15,
matching the style of the Harmful/Harmless Generalization Test figure.

Layout:
  Top-left:  Mean Diff Direction - Train (known attacks)
  Top-right: LDA Direction - Train (known attacks)
  Bottom-left:  Mean Diff Direction - Holdout (unseen attacks)
  Bottom-right: LDA Direction - Holdout (unseen attacks)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from utils.probe_utils import extract_separation_direction, project_onto_direction

# ── Config ──
MODEL_NAME = "Meta-Llama-3-8B-Instruct"
DATASET_NAME = "unsw_nb15"
BEST_LAYER = 31  # best layer from metrics

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPS_DIR = os.path.join(BASE, "results", "representations", DATASET_NAME, MODEL_NAME)
OUT_PATH = os.path.join(BASE, "results", "figures", "unsw_nb15_generalization_test_layer31.png")

# ── Load data ──
print("Loading representations...")
train = torch.load(os.path.join(REPS_DIR, "train.pt"), weights_only=False)
holdout = torch.load(os.path.join(REPS_DIR, "test_holdout.pt"), weights_only=False)

y_train = train["labels"].numpy()
y_holdout = holdout["labels"].numpy()

X_train = train["hidden_states"][BEST_LAYER]
X_holdout = holdout["hidden_states"][BEST_LAYER]

# ── Extract directions ──
print("Computing directions...")
# Mean diff direction
direction = extract_separation_direction(X_train, y_train)

# Train projections
proj_train_normal = project_onto_direction(X_train[y_train == 0], direction)
proj_train_attack = project_onto_direction(X_train[y_train == 1], direction)

# Holdout projections
proj_holdout_normal = project_onto_direction(X_holdout[y_holdout == 0], direction)
proj_holdout_attack = project_onto_direction(X_holdout[y_holdout == 1], direction)

# Threshold & accuracy for holdout
threshold_md = (proj_train_normal.mean() + proj_train_attack.mean()) / 2
proj_all_holdout_md = project_onto_direction(X_holdout, direction)
pred_md = (proj_all_holdout_md > threshold_md).astype(int)
acc_md = (pred_md == y_holdout).mean()

# LDA direction
X_train_np = X_train.numpy() if hasattr(X_train, 'numpy') else np.array(X_train)
X_holdout_np = X_holdout.numpy() if hasattr(X_holdout, 'numpy') else np.array(X_holdout)

lda = LinearDiscriminantAnalysis(n_components=1)
lda.fit(X_train_np, y_train)

lda_train = lda.transform(X_train_np).ravel()
lda_holdout = lda.transform(X_holdout_np).ravel()

# LDA threshold & accuracy for holdout
threshold_lda = (lda_train[y_train == 0].mean() + lda_train[y_train == 1].mean()) / 2
pred_lda = (lda_holdout > threshold_lda).astype(int)
acc_lda = (pred_lda == y_holdout).mean()

# ── Plot ──
print(f"Plotting... (Mean Diff holdout acc: {acc_md:.3f}, LDA holdout acc: {acc_lda:.3f})")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f"UNSW-NB15 Generalization Test (Layer {BEST_LAYER})", fontsize=16, fontweight='bold')

color_normal = '#6BAED6'   # steelblue-ish
color_attack = '#FD8D3C'   # orange-ish

def hist_panel(ax, proj0, proj1, title, label0="Normal", label1="Attack", acc=None):
    ax.hist(proj0, bins=50, alpha=0.6, label=label0, density=True, color=color_normal)
    ax.hist(proj1, bins=50, alpha=0.6, label=label1, density=True, color=color_attack)
    if acc is not None:
        ax.text(0.02, 0.95, f"Acc: {acc:.3f}", transform=ax.transAxes,
                fontsize=11, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.set_xlabel("Projection Value", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)

# Top-left: Mean Diff Direction - Train
hist_panel(axes[0, 0], proj_train_normal, proj_train_attack,
           "Mean Diff Direction - Train (known attacks)",
           label0="Normal", label1="Attack (known)")

# Top-right: LDA Direction - Train
hist_panel(axes[0, 1], lda_train[y_train == 0], lda_train[y_train == 1],
           "LDA Direction - Train (known attacks)",
           label0="Normal", label1="Attack (known)")

# Bottom-left: Mean Diff Direction - Holdout
hist_panel(axes[1, 0], proj_holdout_normal, proj_holdout_attack,
           "Mean Diff Direction - Holdout (unseen attacks)",
           label0="Normal", label1="Attack (holdout)", acc=acc_md)

# Bottom-right: LDA Direction - Holdout
hist_panel(axes[1, 1], lda_holdout[y_holdout == 0], lda_holdout[y_holdout == 1],
           "LDA Direction - Holdout (unseen attacks)",
           label0="Normal", label1="Attack (holdout)", acc=acc_lda)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f"Saved to: {OUT_PATH}")
