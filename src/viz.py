"""
All plotting — three categories:
  1. Paper figures (clean, labeled, publication-quality)
  2. Understanding figures (build intuition about the science)
  3. Diagnostic figures (surface bugs; run these first when something looks off)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless / Kaggle safe
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from src.utils import ensure_dir

FIGURES_DIR = "results/figures"
DPI = 150
PAPER_DPI = 300

# Consistent color palette
COLORS = {
    "firstorder": "#e07b54",
    "curvature_analytic": "#5b8db8",
    "curvature_readout": "#3a7d44",
    "loo": "#333333",
    "ols": "#9b59b6",
}
METHOD_LABELS = {
    "firstorder": "First-order (gradient)",
    "curvature_analytic": "Analytic leverage",
    "curvature_readout": "Curvature readout (ours)",
    "loo": "LOO (ground truth)",
}


def _savefig(fig: plt.Figure, name: str, paper: bool = False, fig_dir: str = FIGURES_DIR) -> Path:
    ensure_dir(fig_dir)
    dpi = PAPER_DPI if paper else DPI
    path = Path(fig_dir) / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    svg_path = Path(fig_dir) / f"{name}.svg"
    if paper:
        fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved {path}")
    return path


# ===========================================================================
# DIAGNOSTIC figures
# ===========================================================================

def plot_training_curve(history: list[dict], fig_dir: str = FIGURES_DIR) -> Path:
    """Loss curve + OLS correlation over training steps."""
    steps = [h["step"] for h in history]
    losses = [h["train_loss"] for h in history]
    corrs = [h.get("ols_corr_mean", None) for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(steps, losses, color="#e07b54", lw=1.5)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Training Loss")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)

    if any(c is not None for c in corrs):
        axes[1].plot(steps, corrs, color="#3a7d44", lw=1.5)
        axes[1].axhline(1.0, color="gray", ls="--", lw=0.8)
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Correlation with OLS")
        axes[1].set_title("Model–OLS Correlation (diagnostic)")
        axes[1].set_ylim(-0.1, 1.05)
        axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return _savefig(fig, "diag_training_curve", fig_dir=fig_dir)


def plot_pred_vs_true(preds: np.ndarray, targets: np.ndarray,
                       ols_preds: np.ndarray | None = None,
                       fig_dir: str = FIGURES_DIR) -> Path:
    """Predicted vs. true y_q scatter; optionally compare with OLS."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(targets, preds, alpha=0.3, s=8, color="#e07b54", label="Model")
    if ols_preds is not None:
        ax.scatter(targets, ols_preds, alpha=0.3, s=8, color="#5b8db8", label="OLS (closed-form)")
    lo, hi = targets.min(), targets.max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="y=x")
    ax.set_xlabel("True y_q")
    ax.set_ylabel("Predicted y_q")
    ax.set_title("Diagnostic: Predicted vs. True")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return _savefig(fig, "diag_pred_vs_true", fig_dir=fig_dir)


def plot_score_distributions(scores: dict[str, np.ndarray],
                              fig_dir: str = FIGURES_DIR) -> Path:
    """Distribution of importance scores per method (degenerate if all-equal)."""
    n = len(scores)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (method, vals) in zip(axes, scores.items()):
        flat = vals.flatten()
        ax.hist(flat, bins=40, color=COLORS.get(method, "gray"), alpha=0.7, edgecolor="none")
        ax.set_title(METHOD_LABELS.get(method, method), fontsize=9)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Diagnostic: Importance Score Distributions", fontsize=11)
    fig.tight_layout()
    return _savefig(fig, "diag_score_distributions", fig_dir=fig_dir)


def plot_score_correlation_heatmap(scores: dict[str, np.ndarray],
                                    fig_dir: str = FIGURES_DIR) -> Path:
    """Pairwise Spearman correlations between methods (flattened across all tasks)."""
    from scipy.stats import spearmanr
    methods = list(scores.keys())
    n = len(methods)
    mat = np.zeros((n, n))
    flat = {m: scores[m].flatten() for m in methods}
    for i, mi in enumerate(methods):
        for j, mj in enumerate(methods):
            mat[i, j] = spearmanr(flat[mi], flat[mj]).statistic

    fig, ax = plt.subplots(figsize=(6, 5))
    labels = [METHOD_LABELS.get(m, m) for m in methods]
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(mat[i,j]) > 0.6 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Diagnostic: Pairwise Spearman Correlation of Importance Scores")
    fig.tight_layout()
    return _savefig(fig, "diag_score_correlation_heatmap", fig_dir=fig_dir)


# ===========================================================================
# UNDERSTANDING figures
# ===========================================================================

def plot_leverage_vs_context_index(lev_analytic: np.ndarray, lev_readout: np.ndarray,
                                    fig_dir: str = FIGURES_DIR) -> Path:
    """Mean leverage score per context position (are some positions systematically high?)."""
    K = lev_analytic.shape[1]
    mean_a = lev_analytic.mean(0)
    std_a = lev_analytic.std(0)
    mean_r = lev_readout.mean(0)
    std_r = lev_readout.std(0)

    fig, ax = plt.subplots(figsize=(8, 4))
    idx = np.arange(K)
    ax.plot(idx, mean_a, "o-", color=COLORS["curvature_analytic"],
            label="Analytic leverage", lw=1.5, ms=4)
    ax.fill_between(idx, mean_a - std_a, mean_a + std_a,
                    color=COLORS["curvature_analytic"], alpha=0.2)
    ax.plot(idx, mean_r, "s-", color=COLORS["curvature_readout"],
            label="Model readout", lw=1.5, ms=4)
    ax.fill_between(idx, mean_r - std_r, mean_r + std_r,
                    color=COLORS["curvature_readout"], alpha=0.2)
    ax.set_xlabel("Context position i")
    ax.set_ylabel("Leverage score")
    ax.set_title("Leverage by Context Position")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    return _savefig(fig, "understand_leverage_by_position", fig_dir=fig_dir)


def plot_curvature_spectrum(eigs_data: np.ndarray, eigs_model: np.ndarray | None = None,
                             fig_dir: str = FIGURES_DIR) -> Path:
    """Spectrum of X^T X (and optionally the model Gram) — understand the implicit objective."""
    fig, ax = plt.subplots(figsize=(7, 4))
    D = eigs_data.shape[1]
    mean_d = eigs_data.mean(0)
    ax.bar(range(D), mean_d, color=COLORS["curvature_analytic"], alpha=0.7, label="Data Gram X^TX")
    if eigs_model is not None:
        d_model = eigs_model.shape[1]
        mean_m = eigs_model.mean(0)[:D]
        ax.plot(range(len(mean_m)), mean_m / (mean_m.max() + 1e-8) * mean_d.max(),
                "o-", color=COLORS["curvature_readout"], ms=3, lw=1.5,
                label="Model Gram (scaled)")
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue magnitude")
    ax.set_title("Curvature Spectrum of Implicit Objective")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    return _savefig(fig, "understand_curvature_spectrum", fig_dir=fig_dir)


def plot_adversarial_example(lev_analytic: np.ndarray, fo_scores: np.ndarray,
                              loo_scores: np.ndarray, fig_dir: str = FIGURES_DIR) -> Path:
    """For a single adversarial context, show all three scores side by side."""
    K = lev_analytic.shape[0]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    pairs = [(fo_scores, "First-order", COLORS["firstorder"]),
             (lev_analytic, "Analytic leverage", COLORS["curvature_analytic"]),
             (loo_scores, "LOO (ground truth)", COLORS["loo"])]
    for ax, (scores, label, color) in zip(axes, pairs):
        ax.bar(range(K), scores, color=color, alpha=0.8)
        # Mark collinear pairs
        for start in range(0, K - 1, 2):
            ax.axvspan(start - 0.4, start + 1.4, alpha=0.08, color="red")
        ax.set_xlabel("Context example i")
        ax.set_ylabel("Score")
        ax.set_title(label, fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Example Adversarial Context (red bands = collinear pairs)", fontsize=11)
    fig.tight_layout()
    return _savefig(fig, "understand_adversarial_example", fig_dir=fig_dir)


# ===========================================================================
# PAPER figures
# ===========================================================================

def plot_rank_agreement(summary: dict, fig_dir: str = FIGURES_DIR) -> Path:
    """
    Headline paper figure: rank agreement (Spearman) of each method vs. LOO,
    on random and adversarial contexts, with error bars.
    """
    methods = ["firstorder", "curvature_analytic", "curvature_readout"]
    context_types = ["random", "adversarial"]
    n_ctx = len(context_types)
    n_m = len(methods)
    x = np.arange(n_ctx)
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, method in enumerate(methods):
        means = [summary[ctx][method]["spearman_mean"]["mean"] for ctx in context_types]
        stds = [summary[ctx][method]["spearman_mean"]["std"] for ctx in context_types]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, means, width, yerr=stds,
                       color=COLORS[method], alpha=0.85,
                       label=METHOD_LABELS[method], capsize=4, error_kw={"lw": 1.5})

    ax.set_xticks(x)
    ax.set_xticklabels(["Random contexts", "Adversarial (collinear) contexts"], fontsize=11)
    ax.set_ylabel("Spearman ρ with LOO ground truth", fontsize=11)
    ax.set_title("Rank Agreement vs. LOO Importance\n(error bars = std over seeds)", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, 1.05)
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return _savefig(fig, "paper_rank_agreement", paper=True, fig_dir=fig_dir)


def plot_topk_recall(summary: dict, fig_dir: str = FIGURES_DIR) -> Path:
    """Top-k recall variant of the headline figure."""
    methods = ["firstorder", "curvature_analytic", "curvature_readout"]
    context_types = ["random", "adversarial"]
    x = np.arange(len(context_types))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, method in enumerate(methods):
        means = [summary[ctx][method]["topk_mean"]["mean"] for ctx in context_types]
        stds = [summary[ctx][method]["topk_mean"]["std"] for ctx in context_types]
        ax.bar(x + (i - 1) * width, means, width, yerr=stds,
               color=COLORS[method], alpha=0.85,
               label=METHOD_LABELS[method], capsize=4, error_kw={"lw": 1.5})

    ax.set_xticks(x)
    ax.set_xticklabels(["Random contexts", "Adversarial (collinear) contexts"], fontsize=11)
    ax.set_ylabel("Top-k Recall", fontsize=11)
    ax.set_title("Top-k Recall vs. LOO Ground Truth", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return _savefig(fig, "paper_topk_recall", paper=True, fig_dir=fig_dir)


def plot_readout_vs_analytic_corr(lev_readout: np.ndarray, lev_analytic: np.ndarray,
                                   fig_dir: str = FIGURES_DIR) -> Path:
    """Scatter: model readout vs. analytic leverage — do they agree?"""
    flat_r = lev_readout.flatten()
    flat_a = lev_analytic.flatten()
    from scipy.stats import spearmanr
    rho = spearmanr(flat_r, flat_a).statistic

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(flat_a, flat_r, alpha=0.15, s=4, color=COLORS["curvature_readout"])
    ax.set_xlabel("Analytic leverage $h_i$", fontsize=11)
    ax.set_ylabel("Model readout leverage", fontsize=11)
    ax.set_title(f"Readout vs. Analytic Leverage\n(Spearman ρ = {rho:.3f})", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _savefig(fig, "paper_readout_vs_analytic", paper=True, fig_dir=fig_dir)
