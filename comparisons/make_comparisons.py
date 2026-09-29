"""Generate the cross-model comparison figures in comparisons/.

Reads only existing outputs (no training, no GPU):
  - task5_cam/cam_summary.json
  - task5_cam/<model>/cam_stats.csv
  - task2_*/outputs/<model>_history.json
  - the pairwise CAM correlations printed by task5_cam/GradCAM_comparison.ipynb

Run from the repo root:  python comparisons/make_comparisons.py
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "comparisons"

MODELS = ["ResNet50", "VGG16", "DenseNet121"]
# Same colours as roc_comparison.png so a model keeps its colour across every figure.
COLORS = {"ResNet50": "#2a78d6", "VGG16": "#eb6834", "DenseNet121": "#1baf7a"}
INK, MUTED, GRID = "#222222", "#666666", "#e6e6e6"

HISTORY = {
    "ResNet50": ROOT / "task2_resnet_baseline/outputs/resnet50_history.json",
    "VGG16": ROOT / "task2_vgg16/outputs/vgg16_history.json",
    "DenseNet121": ROOT / "task2_densenet121/outputs/densenet121_history.json",
}
CAM_DIR = {"ResNet50": "resnet50", "VGG16": "vgg16", "DenseNet121": "densenet121"}

# Mean per-image Pearson correlation between two models' CAMs
# (printed by task5_cam/GradCAM_comparison.ipynb; rounded to 0.43/0.50/0.40 in CAM_comparison.md).
PAIR_CORR = {
    ("ResNet50", "VGG16"): 0.432,
    ("ResNet50", "DenseNet121"): 0.498,
    ("VGG16", "DenseNet121"): 0.396,
}
BORDER_CHANCE = 0.354  # share of the image in the outer 10% band
EPOCH_BUDGET = 20  # Task 2 recipe: 20 epochs, early stopping patience 8

plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "axes.axisbelow": True,
})

summary = json.loads((ROOT / "task5_cam/cam_summary.json").read_text())
overall = {r["model"]: r for r in summary["overall"]}
by_correct = {(r["model"], r["correct"]): r for r in summary["by_correct"]}


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


def bar_labels(ax, bars, fmt="{:.3f}"):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(b.get_height()),
                ha="center", va="bottom", fontsize=8.5, color=INK)


# 1. CAM localisation metrics, one panel per metric -----------------------------------
def fig_cam_metrics():
    lesion_chance = overall["ResNet50"]["lesion_area"]
    panels = [
        ("lesion_energy", "lesion_energy\n(higher = better)", lesion_chance),
        ("peak_in_lesion", "peak_in_lesion\n(higher = better)", lesion_chance),
        ("border_energy", "border_energy\n(lower = better)", BORDER_CHANCE),
        ("hot_area", "hot_area\n(share of image with CAM > 0.5)", None),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    x = np.arange(len(MODELS))
    for ax, (key, title, chance) in zip(axes, panels):
        vals = [overall[m][key] for m in MODELS]
        bars = ax.bar(x, vals, width=0.6, color=[COLORS[m] for m in MODELS],
                      edgecolor="white", linewidth=2)
        bar_labels(ax, bars)
        if chance is not None:
            ax.axhline(chance, color=MUTED, ls="--", lw=1.2)
            ax.set_xlabel(f"dashed line: chance = {chance:.3f}", fontsize=8.5, color=MUTED)
        top = max(vals + [chance or 0])
        ax.set_ylim(0, top * 1.22)
        ax.set_xticks(x, MODELS)
        ax.set_title(title, fontsize=10.5)
    fig.suptitle("Grad-CAM localisation metrics on the 130 test images "
                 "(lesion masks used only as a ruler)", fontsize=12, color=INK, y=1.03)
    save(fig, "cam_metrics_comparison.png")


# 2. Correct vs wrong predictions ---------------------------------------------------------
def fig_correct_vs_wrong():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    x = np.arange(len(MODELS))
    w = 0.36
    for ax, key, title in [
        (axes[0], "lesion_energy", "lesion_energy (higher = better)"),
        (axes[1], "border_energy", "border_energy (lower = better)"),
    ]:
        top = 0
        for j, (correct, off) in enumerate([(True, -w / 2), (False, w / 2)]):
            rows = [by_correct[(m, correct)] for m in MODELS]
            vals = [r[key] for r in rows]
            bars = ax.bar(x + off, vals, width=w, color=[COLORS[m] for m in MODELS],
                          alpha=1.0 if correct else 0.45, hatch=None if correct else "//",
                          edgecolor="white", linewidth=2)
            bar_labels(ax, bars)
            for b, r in zip(bars, rows):
                ax.text(b.get_x() + b.get_width() / 2, 0.004, f"n={r['n']}", ha="center",
                        va="bottom", fontsize=7.5, color="white" if correct else INK)
                if key == "lesion_energy":
                    # Chance level for lesion_energy is the group's own mean lesion area,
                    # which differs between correct and wrong groups.
                    ax.hlines(r["lesion_area"], b.get_x(), b.get_x() + b.get_width(),
                              colors=INK, lw=1.4, linestyles=":")
            top = max(top, max(vals))
        if key == "border_energy":
            ax.axhline(BORDER_CHANCE, color=MUTED, ls="--", lw=1.2)
            ax.set_xlabel(f"dashed line: chance = {BORDER_CHANCE}", fontsize=8.5, color=MUTED)
            top = max(top, BORDER_CHANCE)
        ax.set_ylim(0, top * 1.2)
        ax.set_xticks(x, MODELS)
        ax.set_title(title, fontsize=10.5)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [Patch(facecolor="#888888", label="correct predictions"),
               Patch(facecolor="#888888", alpha=0.45, hatch="//", label="wrong predictions"),
               Line2D([], [], color=INK, ls=":", lw=1.4,
                      label="chance for that group (its mean lesion area)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Does the CAM look different when the model is wrong? "
                 "(wrong groups are small: n = 13 / 18 / 13)", fontsize=12, color=INK, y=1.02)
    save(fig, "cam_correct_vs_wrong.png")


# 3. Loss curves --------------------------------------------------------------------------
def fig_loss_curves():
    hist = {m: json.loads(HISTORY[m].read_text()) for m in MODELS}
    ymax = max(max(h["train_loss"] + h["val_loss"]) for h in hist.values())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, m in zip(axes, MODELS):
        h = hist[m]
        ep = np.arange(1, len(h["train_loss"]) + 1)
        ax.plot(ep, h["train_loss"], color=COLORS[m], lw=2, label="train loss")
        ax.plot(ep, h["val_loss"], color=COLORS[m], lw=2, ls="--", label="val loss")
        best = h["best_epoch"]
        ax.axvline(best, color=MUTED, lw=1, ls=":")
        ax.plot(best, h["val_loss"][best - 1], "o", ms=8, color=COLORS[m],
                markeredgecolor="white", markeredgewidth=2)
        ax.text(best, ymax * 1.02, f" best epoch {best}", fontsize=8.5, color=MUTED,
                va="bottom", ha="left")
        stop = "early-stopped" if len(ep) < EPOCH_BUDGET else "full budget"
        ax.set_title(f"{m} ({len(ep)}/{EPOCH_BUDGET} epochs, {stop})", fontsize=10.5)
        ax.set_xlabel("epoch")
        ax.grid(axis="x", color=GRID)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axes[0].set_ylabel("cross-entropy loss")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=MUTED, lw=2, label="train loss"),
                        Line2D([], [], color=MUTED, lw=2, ls="--", label="val loss")],
               loc="lower center", ncol=2, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.1))
    axes[0].set_ylim(0, ymax * 1.12)
    fig.suptitle("Task 2 training curves under the same recipe "
                 "(dot = checkpoint kept for testing)", fontsize=12, color=INK, y=1.03)
    save(fig, "loss_curves_three_models.png")


# 4. Missed malignant cases ---------------------------------------------------------------
def fig_missed_cases():
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.axhspan(0.3, 0.5, color="#f2f2f2", zorder=0)
    ax.axhspan(0.0, 0.1, color="#f2f2f2", zorder=0)
    ax.axhline(0.5, color=INK, lw=1.4, ls="--")
    ax.text(2.45, 0.5, "decision threshold 0.5", ha="right", va="bottom", fontsize=8.5,
            color=INK)
    ax.text(-0.45, 0.49, "near threshold (0.3 to 0.5)", va="top", fontsize=8.5, color=MUTED)
    ax.text(-0.45, 0.09, "confidently wrong (< 0.1)", va="top", fontsize=8.5, color=MUTED)
    for i, m in enumerate(MODELS):
        d = pd.read_csv(ROOT / "task5_cam" / CAM_DIR[m] / "cam_stats.csv")
        p = d[(d.true == "malignant") & (d.pred == "benign")].prob_malignant.to_numpy()
        jitter = rng.uniform(-0.12, 0.12, len(p))
        ax.scatter(i + jitter, p, s=70, color=COLORS[m], edgecolor="white", linewidth=2,
                   zorder=3)
        n_near = int(((p >= 0.3) & (p < 0.5)).sum())
        n_conf = int((p < 0.1).sum())
        ax.text(i, 0.62, f"{len(p)} missed\n{n_near} near threshold\n{n_conf} confidently wrong",
                ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(range(len(MODELS)), MODELS)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.02, 0.85)
    ax.set_ylabel("predicted P(malignant)")
    ax.set_title("Missed malignant cases (true = malignant, predicted = benign): "
                 "how close were they?", fontsize=11)
    save(fig, "missed_case_probability_distribution.png")


# 5. CAM agreement matrix -----------------------------------------------------------------
def fig_agreement():
    n = len(MODELS)
    mat = np.eye(n)
    for (a, b), v in PAIR_CORR.items():
        i, j = MODELS.index(a), MODELS.index(b)
        mat[i, j] = mat[j, i] = v
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.grid(False)
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=12,
                    color="white" if mat[i, j] > 0.6 else INK)
    ax.set_xticks(range(n), MODELS)
    ax.set_yticks(range(n), MODELS)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("mean per-image Pearson r")
    ax.set_title("How similar are the three models' Grad-CAMs?\n"
                 "(mean over the 130 test images)", fontsize=11)
    save(fig, "cam_model_agreement_matrix.png")


if __name__ == "__main__":
    fig_cam_metrics()
    fig_correct_vs_wrong()
    fig_loss_curves()
    fig_missed_cases()
    fig_agreement()
