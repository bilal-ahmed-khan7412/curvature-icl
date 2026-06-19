"""Phase A visualisation functions."""

from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils import ensure_dir

FIGURES_DIR = "results/figures"
DPI = 150
PAPER_DPI = 300

COLORS = {
    "firstorder": "#e07b54",
    "curvature_analytic": "#5b8db8",
    "curvature_readout": "#3a7d44",
}
METHOD_LABELS = {
    "firstorder": "First-order",
    "curvature_analytic": "Analytic leverage",
    "curvature_readout": "Curvature readout",
}


def _savefig(fig, name, paper=False, fig_dir=FIGURES_DIR):
    ensure_dir(fig_dir)
    path = Path(fig_dir) / f"{name}.png"
    fig.savefig(path, dpi=PAPER_DPI if paper else DPI, bbox_inches="tight")
    if paper:
        fig.savefig(Path(fig_dir) / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved {path}")
    return path


# ---------------------------------------------------------------------------
# Regime sweep: importance spread vs K/D
# ---------------------------------------------------------------------------

def plot_regime_sweep(sweep: dict, fig_dir: str = FIGURES_DIR) -> Path:
    """
    Diagnostic + paper figure: mean LOO importance and within-task spread
    as a function of K/D. Shows regime collapse in over-determined setting.
    """
    ks = sorted(sweep.keys())
    kd = [sweep[k]["kd_ratio"] for k in ks]
    mean_imp = [sweep[k]["mean_importance"] for k in ks]
    within_std = [sweep[k]["within_task_std_mean"] for k in ks]
    within_range = [sweep[k]["within_task_range_mean"] for k in ks]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: mean importance vs K/D
    ax = axes[0]
    ax.plot(kd, mean_imp, "o-", color="#e07b54", lw=2, ms=7, label="Mean |Δŷ_q|")
    ax.fill_between(kd,
                    [m - s for m, s in zip(mean_imp, within_std)],
                    [m + s for m, s in zip(mean_imp, within_std)],
                    color="#e07b54", alpha=0.2)
    ax.axvline(1.0, color="gray", ls="--", lw=1, label="K = D (critical)")
    ax.axvline(2.0, color="#333", ls=":", lw=1, label="Original (K/D=2)")
    ax.set_xlabel("K / D (context size / feature dim)", fontsize=11)
    ax.set_ylabel("Mean LOO importance |Δŷ_q|", fontsize=11)
    ax.set_title("LOO Importance vs. K/D Regime", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: within-task range (how much examples differ in importance)
    ax = axes[1]
    ax.bar([str(k) for k in ks], within_range, color="#5b8db8", alpha=0.8)
    ax.axvline(ks.index(8) - 0.5 + 1, color="gray", ls="--", lw=1)
    ax.set_xlabel("Context size K (D=8)", fontsize=11)
    ax.set_ylabel("Mean within-task importance range", fontsize=11)
    ax.set_title("Importance Spread: How Much Examples Differ", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate original K
    orig_idx = ks.index(16) if 16 in ks else None
    if orig_idx is not None:
        ax.patches[orig_idx].set_facecolor("#e07b54")
        ax.patches[orig_idx].set_alpha(1.0)
        ax.annotate("Original\nK=16", xy=(orig_idx, within_range[orig_idx]),
                    xytext=(orig_idx + 0.5, within_range[orig_idx] * 1.1),
                    fontsize=8, color="#e07b54")

    fig.suptitle("Phase A: K/D Regime Sweep — Finding Where Importance Is Meaningful",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    return _savefig(fig, "phaseA_regime_sweep", paper=True, fig_dir=fig_dir)


def plot_importance_distribution(regime_result: dict,
                                  fig_dir: str = FIGURES_DIR) -> Path:
    """Histogram of LOO scores for a single K — diagnostic."""
    k = regime_result["k"]
    scores = np.array(regime_result["scores_sample"]).flatten()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(scores, bins=30, color="#5b8db8", alpha=0.8, edgecolor="none")
    ax.axvline(np.mean(scores), color="#e07b54", lw=2, label=f"Mean={np.mean(scores):.3f}")
    ax.set_xlabel("|Δŷ_q| (LOO importance)")
    ax.set_ylabel("Count")
    ax.set_title(f"LOO Distribution: K={k}, K/D={regime_result['kd_ratio']:.2f}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _savefig(fig, f"phaseA_loo_dist_K{k}", fig_dir=fig_dir)


# ---------------------------------------------------------------------------
# Stability audit
# ---------------------------------------------------------------------------

def plot_stability_audit(stability: dict, sweep: dict,
                          fig_dir: str = FIGURES_DIR) -> Path:
    """Bar chart of CV values across noise seeds, query points, context resamples."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: CV breakdown
    ax = axes[0]
    labels = ["Noise-seed CV", "Query CV", "Context-resample CV"]
    vals = [stability["noise_seed_cv"], stability["query_cv"],
            stability["context_resample_cv"]]
    colors = ["#e07b54", "#5b8db8", "#3a7d44"]
    bars = ax.bar(labels, vals, color=colors, alpha=0.85)
    ax.axhline(0.5, color="black", ls="--", lw=1.2, label="Stability threshold (0.5)")
    ax.set_ylabel("Coefficient of Variation")
    ax.set_title(f"LOO Stability at K={stability['k']} (K/D={stability['kd_ratio']:.2f})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=9)

    # Right: noise CV across all K values (for context)
    ax = axes[1]
    # Re-use sweep to show mean importance as reference
    ks = sorted(sweep.keys())
    kd = [sweep[k]["kd_ratio"] for k in ks]
    mean_imp = [sweep[k]["mean_importance"] for k in ks]
    ax.plot(kd, mean_imp, "o-", color="#5b8db8", lw=2, ms=6)
    ax.axvline(stability["kd_ratio"], color="#e07b54", lw=2,
               label=f"Stability tested K={stability['k']}")
    ax.set_xlabel("K / D")
    ax.set_ylabel("Mean LOO importance")
    ax.set_title("Tested Regime in Context of Sweep")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    stable_str = "STABLE" if stability["is_stable"] else "UNSTABLE → use Shapley"
    fig.suptitle(f"Phase A: Stability Audit — LOO is {stable_str}", fontsize=12, y=1.01)
    fig.tight_layout()
    return _savefig(fig, "phaseA_stability_audit", paper=True, fig_dir=fig_dir)


# ---------------------------------------------------------------------------
# Re-test comparison
# ---------------------------------------------------------------------------

def plot_retest_comparison(retest: dict, fig_dir: str = FIGURES_DIR) -> Path:
    """
    Paper figure: re-test of 3 original methods in the good regime.
    Bar chart of Spearman correlation with ground truth.
    """
    methods = list(retest["summary"].keys())
    means = [retest["summary"][m]["spearman_mean"] for m in methods]
    stds = [retest["summary"][m]["spearman_std"] for m in methods]
    labels = [METHOD_LABELS.get(m, m) for m in methods]
    colors = [COLORS.get(m, "gray") for m in methods]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, means, color=colors, alpha=0.85,
                  yerr=stds, capsize=5, error_kw={"lw": 1.5})
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel(f"Spearman ρ with {retest['ground_truth'].upper()} ground truth", fontsize=11)
    ax.set_title(
        f"Phase A Re-test: Methods at K={retest['k']} (K/D={retest['kd_ratio']:.2f})\n"
        f"Ground truth: {retest['ground_truth']}  |  n={retest['n_tasks']} tasks",
        fontsize=11
    )
    ax.set_ylim(-0.1, max(0.6, max(means) + 0.15))
    ax.grid(True, alpha=0.3, axis="y")
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                max(m, 0) + s + 0.015, f"{m:.3f}",
                ha="center", fontsize=9)
    fig.tight_layout()
    return _savefig(fig, "phaseA_retest_comparison", paper=True, fig_dir=fig_dir)


# ---------------------------------------------------------------------------
# Combined summary figure for GATE A
# ---------------------------------------------------------------------------

def plot_gate_a_summary(sweep: dict, stability: dict, retest: dict,
                         verdict: dict, fig_dir: str = FIGURES_DIR) -> Path:
    """One-page summary of all Phase A findings for the paper."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1: Regime sweep
    ks = sorted(sweep.keys())
    kd = [sweep[k]["kd_ratio"] for k in ks]
    within_range = [sweep[k]["within_task_range_mean"] for k in ks]
    axes[0].bar([f"{k}" for k in ks], within_range, color="#5b8db8", alpha=0.8)
    best_k = verdict["best_k"]
    if best_k in ks:
        bi = ks.index(best_k)
        axes[0].patches[bi].set_facecolor("#3a7d44")
    axes[0].set_xlabel("K"); axes[0].set_ylabel("Within-task importance range")
    axes[0].set_title("A.1 Regime Sweep")
    axes[0].grid(True, alpha=0.3, axis="y")

    # 2: Stability
    labels = ["Noise", "Query", "Resample"]
    vals = [stability["noise_seed_cv"], stability["query_cv"],
            stability["context_resample_cv"]]
    axes[1].bar(labels, vals, color=["#e07b54", "#5b8db8", "#3a7d44"], alpha=0.85)
    axes[1].axhline(0.5, color="black", ls="--", lw=1.2)
    axes[1].set_ylabel("CV"); axes[1].set_title(f"A.2 Stability at K={best_k}")
    axes[1].grid(True, alpha=0.3, axis="y")

    # 3: Re-test
    methods = list(retest["summary"].keys())
    means = [retest["summary"][m]["spearman_mean"] for m in methods]
    stds = [retest["summary"][m]["spearman_std"] for m in methods]
    axes[2].bar([METHOD_LABELS.get(m, m) for m in methods], means,
                color=[COLORS.get(m, "gray") for m in methods],
                yerr=stds, capsize=4, alpha=0.85)
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].set_ylabel("Spearman ρ"); axes[2].set_title("A.3 Re-test (good regime)")
    axes[2].grid(True, alpha=0.3, axis="y")

    gate = verdict["gate"]
    fig.suptitle(f"Phase A Summary — GATE: {gate}  |  Best K={best_k} (K/D={verdict['best_kd_ratio']:.2f})  |  GT={verdict['ground_truth']}",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    return _savefig(fig, "phaseA_gate_summary", paper=True, fig_dir=fig_dir)
