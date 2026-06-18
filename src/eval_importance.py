"""
Phase 1 kill-test evaluation.

Measures how well each importance method's ranking matches LOO ground truth
on two context types:
  - Random contexts (i.i.d. x_i)
  - Adversarial contexts (collinear / jointly-important pairs)

Reports Spearman correlation, Kendall's τ, and top-k recall.
The GATE decision is printed at the end.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr
from tqdm import tqdm

from src.curvature import analytic_leverage, model_curvature_readout
from src.importance import firstorder_importance, loo_importance
from src.toy_icl import build_token_sequence, ICLTransformer, sample_tasks, TaskBatch
from src.utils import ensure_dir, save_metrics, set_seed


# ---------------------------------------------------------------------------
# Adversarial context generation
# ---------------------------------------------------------------------------

def sample_adversarial_tasks(
    batch_size: int,
    context_len: int,
    input_dim: int,
    noise_std: float,
    collinearity: float = 0.95,
    rng: np.random.Generator | None = None,
    device: str = "cpu",
) -> TaskBatch:
    """
    Construct contexts where pairs of examples are nearly collinear.
    For each pair (2i, 2i+1): x_{2i+1} = collinearity * x_{2i} + sqrt(1-c^2) * ε.
    These examples are individually low LOO importance but jointly high.
    """
    if rng is None:
        rng = np.random.default_rng()

    w = rng.standard_normal((batch_size, input_dim)).astype(np.float32)

    xs = np.zeros((batch_size, context_len, input_dim), dtype=np.float32)
    for pair_start in range(0, context_len - 1, 2):
        base = rng.standard_normal((batch_size, input_dim)).astype(np.float32)
        base /= (np.linalg.norm(base, axis=-1, keepdims=True) + 1e-8)
        perturb = rng.standard_normal((batch_size, input_dim)).astype(np.float32)
        perturb -= (perturb * base).sum(-1, keepdims=True) * base   # orthogonalize
        perturb /= (np.linalg.norm(perturb, axis=-1, keepdims=True) + 1e-8)
        xs[:, pair_start] = base
        xs[:, pair_start + 1] = collinearity * base + np.sqrt(1 - collinearity**2) * perturb
    # If context_len is odd, last example is random
    if context_len % 2 == 1:
        xs[:, -1] = rng.standard_normal((batch_size, input_dim)).astype(np.float32)

    noise = rng.standard_normal((batch_size, context_len)).astype(np.float32) * noise_std
    ys = (xs * w[:, None, :]).sum(-1) + noise

    x_query = rng.standard_normal((batch_size, input_dim)).astype(np.float32)
    y_query = (x_query * w).sum(-1).astype(np.float32)

    return TaskBatch(
        xs=torch.from_numpy(xs).to(device),
        ys=torch.from_numpy(ys).to(device),
        x_query=torch.from_numpy(x_query).to(device),
        y_query=torch.from_numpy(y_query).to(device),
        w=torch.from_numpy(w).to(device),
    )


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------

def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(spearmanr(a, b).statistic)


def kendall(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(kendalltau(a, b).statistic)


def topk_recall(scores_pred: np.ndarray, scores_true: np.ndarray, k: int) -> float:
    """Fraction of true top-k that appears in predicted top-k."""
    true_topk = set(np.argsort(scores_true)[-k:])
    pred_topk = set(np.argsort(scores_pred)[-k:])
    return len(true_topk & pred_topk) / k


# ---------------------------------------------------------------------------
# Evaluate one batch
# ---------------------------------------------------------------------------

def _eval_batch(model: ICLTransformer, batch: TaskBatch, k: int,
                device: str) -> dict[str, dict[str, list[float]]]:
    """Compute all scores and ranking metrics for a single batch."""
    loo = loo_importance(model, batch, device).cpu().numpy()      # (B, K)
    fo = firstorder_importance(model, batch, device).cpu().numpy()
    lev_analytic = analytic_leverage(batch).cpu().numpy()
    lev_readout = model_curvature_readout(model, batch, device).cpu().numpy()

    methods = {
        "firstorder": fo,
        "curvature_analytic": lev_analytic,
        "curvature_readout": lev_readout,
    }

    results: dict[str, dict[str, list[float]]] = {m: {"spearman": [], "kendall": [], "topk": []}
                                                    for m in methods}

    for b in range(loo.shape[0]):
        for m_name, m_scores in methods.items():
            results[m_name]["spearman"].append(spearman(m_scores[b], loo[b]))
            results[m_name]["kendall"].append(kendall(m_scores[b], loo[b]))
            results[m_name]["topk"].append(topk_recall(m_scores[b], loo[b], k))

    return results


def _aggregate(collected: dict[str, dict[str, list[float]]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for m, metrics in collected.items():
        out[m] = {}
        for metric, vals in metrics.items():
            arr = np.array(vals)
            out[m][f"{metric}_mean"] = float(arr.mean())
            out[m][f"{metric}_std"] = float(arr.std())
    return out


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_eval(
    model: ICLTransformer,
    cfg: dict,
    seed: int,
    device: str = "cpu",
) -> dict:
    task_cfg = cfg["task"]
    eval_cfg = cfg["eval"]
    k = eval_cfg["top_k"]
    rng_rand = np.random.default_rng(seed)
    rng_adv = np.random.default_rng(seed + 1000)
    batch_size = 64

    # --- Random contexts ---
    n_rand = task_cfg["n_eval_tasks"]
    rand_collected: dict[str, dict[str, list[float]]] = {
        m: {"spearman": [], "kendall": [], "topk": []}
        for m in ("firstorder", "curvature_analytic", "curvature_readout")
    }
    for _ in tqdm(range(0, n_rand, batch_size), desc=f"[eval seed={seed}] random"):
        bs = min(batch_size, n_rand)
        batch = sample_tasks(bs, task_cfg["context_len"], task_cfg["input_dim"],
                              task_cfg["noise_std"], rng_rand, device)
        res = _eval_batch(model, batch, k, device)
        for m in rand_collected:
            for metric in rand_collected[m]:
                rand_collected[m][metric].extend(res[m][metric])

    # --- Adversarial contexts ---
    n_adv = task_cfg["n_adv_tasks"]
    adv_collected: dict[str, dict[str, list[float]]] = {
        m: {"spearman": [], "kendall": [], "topk": []}
        for m in ("firstorder", "curvature_analytic", "curvature_readout")
    }
    for _ in tqdm(range(0, n_adv, batch_size), desc=f"[eval seed={seed}] adversarial"):
        bs = min(batch_size, n_adv)
        batch = sample_adversarial_tasks(
            bs, task_cfg["context_len"], task_cfg["input_dim"], task_cfg["noise_std"],
            task_cfg["adv_collinearity"], rng_adv, device,
        )
        res = _eval_batch(model, batch, k, device)
        for m in adv_collected:
            for metric in adv_collected[m]:
                adv_collected[m][metric].extend(res[m][metric])

    return {
        "seed": seed,
        "random": _aggregate(rand_collected),
        "adversarial": _aggregate(adv_collected),
    }


# ---------------------------------------------------------------------------
# Multi-seed evaluation + gate decision
# ---------------------------------------------------------------------------

def run_killtest(
    model: ICLTransformer,
    cfg: dict,
    device: str = "cpu",
) -> dict:
    """Run eval over all eval_seeds and aggregate. Print the gate verdict."""
    all_results = []
    for seed in cfg["eval_seeds"]:
        set_seed(seed)
        r = run_eval(model, cfg, seed=seed, device=device)
        all_results.append(r)

    # Aggregate across seeds
    methods = ("firstorder", "curvature_analytic", "curvature_readout")
    contexts = ("random", "adversarial")
    metrics = ("spearman_mean", "kendall_mean", "topk_mean")

    summary: dict[str, dict[str, dict[str, dict]]] = {ctx: {m: {} for m in methods}
                                                        for ctx in contexts}
    for ctx in contexts:
        for m in methods:
            for metric in metrics:
                vals = [r[ctx][m][metric] for r in all_results]
                summary[ctx][m][metric] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                }

    # GATE: curvature_readout must beat firstorder on adversarial Spearman
    gap = (summary["adversarial"]["curvature_readout"]["spearman_mean"]["mean"]
           - summary["adversarial"]["firstorder"]["spearman_mean"]["mean"])
    threshold = cfg["eval"]["min_spearman_improvement"]
    gate = "GO" if gap >= threshold else "NO-GO"

    verdict = {
        "gate": gate,
        "adv_spearman_gap_curvature_vs_firstorder": gap,
        "threshold": threshold,
        "summary": summary,
        "per_seed": all_results,
        "config": cfg,
    }

    # Print gate result
    print("\n" + "=" * 60)
    print(f"GATE VERDICT: {gate}")
    print(f"  Adversarial Spearman — curvature_readout: "
          f"{summary['adversarial']['curvature_readout']['spearman_mean']['mean']:.4f} "
          f"± {summary['adversarial']['curvature_readout']['spearman_mean']['std']:.4f}")
    print(f"  Adversarial Spearman — firstorder:        "
          f"{summary['adversarial']['firstorder']['spearman_mean']['mean']:.4f} "
          f"± {summary['adversarial']['firstorder']['spearman_mean']['std']:.4f}")
    print(f"  Gap: {gap:.4f}  (threshold: {threshold})")
    print("=" * 60 + "\n")

    return verdict
