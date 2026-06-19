"""
Phase A — Audit the Ground Truth.

Three sub-tasks:
  A.1  K/D regime sweep: LOO importance distribution vs K/D ratio.
  A.2  Stability audit: variance of LOO scores across noise seeds, query
       points, context resamples within the good regime.
       If LOO is unstable → fall back to Shapley attribution.
  A.3  Re-test the three original methods (firstorder, curvature_analytic,
       curvature_readout) in the good regime — the fair re-test.

All results saved to results/metrics/ and results/figures/.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from tqdm import tqdm

from src.curvature import analytic_leverage, model_curvature_readout
from src.importance import firstorder_importance, loo_importance
from src.toy_icl import (
    ICLTransformer, TaskBatch, build_token_sequence,
    ols_predict, sample_tasks,
)
from src.utils import ensure_dir, save_metrics, set_seed, sync_to_kaggle_output


# ---------------------------------------------------------------------------
# Helpers: sample a batch with a specific K (override cfg context_len)
# ---------------------------------------------------------------------------

def _sample_k(k: int, cfg: dict, n: int, rng: np.random.Generator,
               device: str) -> TaskBatch:
    t = cfg["task"]
    return sample_tasks(n, k, t["input_dim"], t["noise_std"], rng, device)


def _loo_batch(model: ICLTransformer, batch: TaskBatch,
               device: str) -> torch.Tensor:
    """Returns (B, K) LOO importance scores."""
    return loo_importance(model, batch, device)


# ---------------------------------------------------------------------------
# A.1  K/D regime sweep
# ---------------------------------------------------------------------------

K_VALUES = [4, 6, 8, 10, 12, 16, 24, 32]
N_SWEEP_TASKS = 500


def sweep_kd_regimes(
    model: ICLTransformer,
    cfg: dict,
    device: str = "cpu",
    n_tasks: int = N_SWEEP_TASKS,
    seed: int = 42,
) -> dict:
    """
    For each K in K_VALUES, sample n_tasks contexts and compute the full
    LOO importance distribution. Return mean, std, and per-example
    distributions so we can plot importance spread vs K/D.
    """
    D = cfg["task"]["input_dim"]
    model.eval()
    results = {}

    for k in tqdm(K_VALUES, desc="[phase_a] K/D sweep"):
        rng = np.random.default_rng(seed)
        all_scores = []

        for start in range(0, n_tasks, 64):
            bs = min(64, n_tasks - start)
            batch = _sample_k(k, cfg, bs, rng, device)
            scores = _loo_batch(model, batch, device).cpu().numpy()  # (bs, k)
            all_scores.append(scores)

        scores_arr = np.concatenate(all_scores, axis=0)  # (n_tasks, k)
        flat = scores_arr.flatten()

        results[k] = {
            "k": k,
            "d": D,
            "kd_ratio": k / D,
            "mean_importance": float(flat.mean()),
            "std_importance": float(flat.std()),
            "median_importance": float(np.median(flat)),
            "p90_importance": float(np.percentile(flat, 90)),
            "p10_importance": float(np.percentile(flat, 10)),
            # Spread = how much importance varies across examples within a task
            "within_task_std_mean": float(scores_arr.std(axis=1).mean()),
            "within_task_range_mean": float(
                (scores_arr.max(axis=1) - scores_arr.min(axis=1)).mean()
            ),
            "scores_sample": scores_arr[:20].tolist(),  # save a small sample
        }

    return results


# ---------------------------------------------------------------------------
# A.2  LOO stability audit (within a chosen K)
# ---------------------------------------------------------------------------

N_STABILITY_TASKS = 200
N_NOISE_SEEDS = 10


def loo_stability_audit(
    model: ICLTransformer,
    cfg: dict,
    k: int,
    device: str = "cpu",
    n_tasks: int = N_STABILITY_TASKS,
    n_noise_seeds: int = N_NOISE_SEEDS,
    base_seed: int = 42,
) -> dict:
    """
    For n_tasks fixed contexts (fixed xs, w), compute LOO scores under:
      (a) different label-noise seeds
      (b) different query points
      (c) context resamples (entirely new context for the same w)

    Returns stability statistics: for each task, the std of each example's
    LOO score across repetitions relative to the mean.
    """
    D = cfg["task"]["input_dim"]
    model.eval()

    rng_ctx = np.random.default_rng(base_seed)

    # Draw fixed task parameters (w vectors)
    w_batch = rng_ctx.standard_normal((n_tasks, D)).astype(np.float32)

    # (a) Noise-seed stability: same xs, w, x_q — only noise changes
    noise_scores = []  # list of (n_tasks, k) arrays, one per noise seed
    xs_fixed = rng_ctx.standard_normal((n_tasks, k, D)).astype(np.float32)
    xq_fixed = rng_ctx.standard_normal((n_tasks, D)).astype(np.float32)

    for ns in range(n_noise_seeds):
        rng_n = np.random.default_rng(base_seed + ns + 1000)
        noise = rng_n.standard_normal((n_tasks, k)).astype(np.float32) * cfg["task"]["noise_std"]
        ys = (xs_fixed * w_batch[:, None, :]).sum(-1) + noise

        batch = TaskBatch(
            xs=torch.from_numpy(xs_fixed).to(device),
            ys=torch.from_numpy(ys).to(device),
            x_query=torch.from_numpy(xq_fixed).to(device),
            y_query=torch.from_numpy((xq_fixed * w_batch).sum(-1).astype(np.float32)).to(device),
            w=torch.from_numpy(w_batch).to(device),
        )
        sc = _loo_batch(model, batch, device).cpu().numpy()
        noise_scores.append(sc)

    noise_scores = np.stack(noise_scores, axis=0)  # (n_noise_seeds, n_tasks, k)
    noise_std = noise_scores.std(axis=0)    # (n_tasks, k)
    noise_mean = noise_scores.mean(axis=0)  # (n_tasks, k)
    # CV = std / (mean + eps) — relative instability
    noise_cv = (noise_std / (np.abs(noise_mean) + 1e-6)).mean()

    # (b) Query stability: same xs, w, noise seed 0 — different x_q
    n_queries = 10
    query_scores = []
    rng_noise_base = np.random.default_rng(base_seed + 2000)
    noise0 = rng_noise_base.standard_normal((n_tasks, k)).astype(np.float32) * cfg["task"]["noise_std"]
    ys0 = (xs_fixed * w_batch[:, None, :]).sum(-1) + noise0

    for qi in range(n_queries):
        rng_q = np.random.default_rng(base_seed + qi + 3000)
        xq = rng_q.standard_normal((n_tasks, D)).astype(np.float32)
        batch = TaskBatch(
            xs=torch.from_numpy(xs_fixed).to(device),
            ys=torch.from_numpy(ys0).to(device),
            x_query=torch.from_numpy(xq).to(device),
            y_query=torch.from_numpy((xq * w_batch).sum(-1).astype(np.float32)).to(device),
            w=torch.from_numpy(w_batch).to(device),
        )
        sc = _loo_batch(model, batch, device).cpu().numpy()
        query_scores.append(sc)

    query_scores = np.stack(query_scores, axis=0)   # (n_queries, n_tasks, k)
    query_cv = (query_scores.std(axis=0) / (np.abs(query_scores.mean(axis=0)) + 1e-6)).mean()

    # (c) Context resample stability: same w, different xs and x_q each time
    resample_scores = []
    for ri in range(n_noise_seeds):
        rng_r = np.random.default_rng(base_seed + ri + 4000)
        xs_r = rng_r.standard_normal((n_tasks, k, D)).astype(np.float32)
        noise_r = rng_r.standard_normal((n_tasks, k)).astype(np.float32) * cfg["task"]["noise_std"]
        ys_r = (xs_r * w_batch[:, None, :]).sum(-1) + noise_r
        xq_r = rng_r.standard_normal((n_tasks, D)).astype(np.float32)
        batch = TaskBatch(
            xs=torch.from_numpy(xs_r).to(device),
            ys=torch.from_numpy(ys_r).to(device),
            x_query=torch.from_numpy(xq_r).to(device),
            y_query=torch.from_numpy((xq_r * w_batch).sum(-1).astype(np.float32)).to(device),
            w=torch.from_numpy(w_batch).to(device),
        )
        sc = _loo_batch(model, batch, device).cpu().numpy()
        resample_scores.append(sc)

    resample_scores = np.stack(resample_scores, axis=0)
    resample_cv = (resample_scores.std(axis=0) / (np.abs(resample_scores.mean(axis=0)) + 1e-6)).mean()

    # Stability verdict: noise CV < 0.5 means signal dominates noise
    is_stable = float(noise_cv) < 0.5

    return {
        "k": k,
        "kd_ratio": k / D,
        "noise_seed_cv": float(noise_cv),
        "query_cv": float(query_cv),
        "context_resample_cv": float(resample_cv),
        "is_stable": is_stable,
        "noise_mean_importance": float(np.abs(noise_mean).mean()),
    }


# ---------------------------------------------------------------------------
# A.2b  Shapley attribution (principled ground truth for joint importance)
# ---------------------------------------------------------------------------

def shapley_importance(
    model: ICLTransformer,
    batch: TaskBatch,
    device: str = "cpu",
    n_permutations: int = 2000,
) -> torch.Tensor:
    """
    Monte-Carlo Shapley values: φ_i = E_π [ v(S_π^i ∪ {i}) - v(S_π^i) ]
    where S_π^i is the set of examples before i in permutation π,
    and v(S) = -MSE of the model predicting y_q from context S.

    For K ≤ 12: exact Shapley over all 2^K subsets (more accurate).
    For K > 12: MC with n_permutations random permutations.

    Returns: (B, K) Shapley values.
    """
    model.eval()
    B, K, D = batch.xs.shape

    if K <= 12:
        return _exact_shapley(model, batch, device)
    else:
        return _mc_shapley(model, batch, device, n_permutations)


def _model_predict_subset(
    model: ICLTransformer,
    batch: TaskBatch,
    subset_mask: torch.Tensor,  # (B, K) bool
    device: str,
) -> torch.Tensor:
    """
    Predict y_q using only the examples indicated by subset_mask.
    For empty subsets: use the zero-context prediction (model sees only x_q).
    Returns (B,) predictions.
    """
    B, K, D = batch.xs.shape

    # Find max subset size to build padded sequences
    subset_sizes = subset_mask.sum(dim=1)  # (B,)
    max_size = int(subset_sizes.max().item())

    if max_size == 0:
        # Empty context: build sequence with just x_q
        seq = batch.x_query.unsqueeze(1)  # (B, 1, D)
        with torch.no_grad():
            h = model.pe(model.token_proj(seq))
            h = model.transformer(h)
            return model.head(h[:, -1]).squeeze(-1)

    # Build variable-length padded contexts
    # We handle each batch element separately for simplicity
    preds = torch.zeros(B, device=device)

    # Group by subset size for efficiency
    for b in range(B):
        idx = subset_mask[b].nonzero(as_tuple=True)[0]  # selected indices
        sz = len(idx)
        if sz == 0:
            xs_s = torch.zeros(1, 0, D, device=device)
            ys_s = torch.zeros(1, 0, device=device)
        else:
            xs_s = batch.xs[b:b+1, idx, :]   # (1, sz, D)
            ys_s = batch.ys[b:b+1, idx]       # (1, sz)
        xq_s = batch.x_query[b:b+1]           # (1, D)

        if sz == 0:
            seq = xq_s.unsqueeze(1)            # (1, 1, D)
        else:
            seq = build_token_sequence(xs_s, ys_s, xq_s)  # (1, 2*sz+1, D)

        with torch.no_grad():
            preds[b] = model(seq)

    return preds


def _exact_shapley(model: ICLTransformer, batch: TaskBatch,
                   device: str) -> torch.Tensor:
    """Exact Shapley via all 2^K subsets. Only feasible for K ≤ 12."""
    B, K, D = batch.xs.shape
    shap = torch.zeros(B, K, device=device)

    # Precompute v(S) for all 2^K subsets
    n_subsets = 2 ** K
    v = torch.zeros(B, n_subsets, device=device)

    for s_int in tqdm(range(n_subsets), desc="  [shapley] subsets", leave=False):
        mask = torch.zeros(B, K, dtype=torch.bool, device=device)
        for j in range(K):
            if s_int & (1 << j):
                mask[:, j] = True
        preds = _model_predict_subset(model, batch, mask, device)
        mse = (preds - batch.y_query) ** 2
        v[:, s_int] = -mse  # higher = better

    # Compute Shapley values
    for i in range(K):
        for s_int in range(n_subsets):
            if s_int & (1 << i):
                continue  # i is already in S; skip
            s_with_i = s_int | (1 << i)
            s_size = bin(s_int).count("1")
            weight = (math.factorial(s_size) * math.factorial(K - s_size - 1)
                      / math.factorial(K))
            shap[:, i] += weight * (v[:, s_with_i] - v[:, s_int])

    return shap


def _mc_shapley(model: ICLTransformer, batch: TaskBatch,
                device: str, n_permutations: int) -> torch.Tensor:
    """Monte-Carlo Shapley via random permutations."""
    B, K, D = batch.xs.shape
    shap = torch.zeros(B, K, device=device)
    rng = np.random.default_rng(0)

    for _ in tqdm(range(n_permutations), desc="  [shapley] MC perms", leave=False):
        perm = rng.permutation(K)
        v_prev = None

        for pos, i in enumerate(perm):
            mask = torch.zeros(B, K, dtype=torch.bool, device=device)
            for j in perm[:pos + 1]:
                mask[:, j] = True
            preds = _model_predict_subset(model, batch, mask, device)
            v_curr = -(preds - batch.y_query) ** 2

            if v_prev is not None:
                shap[:, i] += v_curr - v_prev
            else:
                # First in permutation: marginal from empty set
                empty_mask = torch.zeros(B, K, dtype=torch.bool, device=device)
                p_empty = _model_predict_subset(model, batch, empty_mask, device)
                v_empty = -(p_empty - batch.y_query) ** 2
                shap[:, i] += v_curr - v_empty

            v_prev = v_curr

    return shap / n_permutations


# ---------------------------------------------------------------------------
# A.3  Re-test original methods in the good regime
# ---------------------------------------------------------------------------

def retest_methods(
    model: ICLTransformer,
    cfg: dict,
    k: int,
    ground_truth: str,  # "loo" or "shapley"
    device: str = "cpu",
    n_tasks: int = 500,
    n_shapley_perm: int = 2000,
    seed: int = 42,
) -> dict:
    """
    Re-run firstorder, curvature_analytic, curvature_readout vs. the
    chosen ground truth at a specific K (the good regime).
    Returns Spearman/Kendall/top-k for each method.
    """
    from scipy.stats import kendalltau
    from src.eval_importance import topk_recall

    model.eval()
    rng = np.random.default_rng(seed)
    batch_size = min(32 if ground_truth == "shapley" and k <= 12 else 16, 64)
    # Shapley is expensive; use smaller batches

    all_results: dict[str, dict[str, list]] = {
        m: {"spearman": [], "kendall": [], "topk": []}
        for m in ("firstorder", "curvature_analytic", "curvature_readout")
    }
    K_TOP = min(3, k)

    for start in tqdm(range(0, n_tasks, batch_size), desc=f"  [retest K={k}]"):
        bs = min(batch_size, n_tasks - start)
        batch = _sample_k(k, cfg, bs, rng, device)

        if ground_truth == "loo":
            gt = loo_importance(model, batch, device).cpu().numpy()
        else:
            gt = shapley_importance(model, batch, device, n_shapley_perm).cpu().numpy()
            # Shapley can be negative; take abs for ranking (higher abs = more important)
            gt = np.abs(gt)

        fo = firstorder_importance(model, batch, device).cpu().numpy()
        lev = analytic_leverage(batch).cpu().numpy()
        rdout = model_curvature_readout(model, batch, device).cpu().numpy()

        methods = {"firstorder": fo, "curvature_analytic": lev, "curvature_readout": rdout}

        for b in range(bs):
            for m_name, m_scores in methods.items():
                gt_b = gt[b]
                sc_b = m_scores[b]
                if np.std(gt_b) < 1e-9 or np.std(sc_b) < 1e-9:
                    all_results[m_name]["spearman"].append(0.0)
                    all_results[m_name]["kendall"].append(0.0)
                else:
                    all_results[m_name]["spearman"].append(
                        float(spearmanr(sc_b, gt_b).statistic))
                    all_results[m_name]["kendall"].append(
                        float(kendalltau(sc_b, gt_b).statistic))
                all_results[m_name]["topk"].append(topk_recall(sc_b, gt_b, K_TOP))

    summary: dict[str, dict] = {}
    for m, metrics in all_results.items():
        summary[m] = {
            f"{met}_mean": float(np.mean(vals))
            for met, vals in metrics.items()
        }
        summary[m].update({
            f"{met}_std": float(np.std(vals))
            for met, vals in metrics.items()
        })

    return {
        "k": k,
        "kd_ratio": k / cfg["task"]["input_dim"],
        "ground_truth": ground_truth,
        "n_tasks": n_tasks,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Gate A: pick good regime, check stability, decide ground truth
# ---------------------------------------------------------------------------

def gate_a_verdict(
    sweep_results: dict,
    stability_results: dict,
    stability_threshold: float = 0.5,
) -> dict:
    """
    Determine:
    1. The best K (regime where importance is most non-degenerate).
    2. Whether LOO is stable in that regime (or whether to use Shapley).
    3. GO / NO-GO.
    """
    # Find K with highest within_task_range_mean (most spread = most signal)
    best_k = max(sweep_results.keys(),
                 key=lambda k: sweep_results[k]["within_task_range_mean"])

    noise_cv = stability_results.get("noise_seed_cv", 1.0)
    is_stable = noise_cv < stability_threshold
    ground_truth = "loo" if is_stable else "shapley"

    mean_imp = sweep_results[best_k]["mean_importance"]
    go = mean_imp > 0.01  # importance is non-negligible

    return {
        "gate": "GO" if go else "NO-GO",
        "best_k": best_k,
        "best_kd_ratio": best_k / 8,  # D=8
        "ground_truth": ground_truth,
        "noise_cv_at_best_k": noise_cv,
        "mean_importance_at_best_k": mean_imp,
        "reasoning": (
            f"K={best_k} (K/D={best_k/8:.2f}) has highest within-task importance spread. "
            f"Noise CV={noise_cv:.3f} → ground truth = {ground_truth}."
        ),
    }


# ---------------------------------------------------------------------------
# Master Phase A runner
# ---------------------------------------------------------------------------

def run_phase_a(
    model: ICLTransformer,
    cfg: dict,
    device: str = "cpu",
) -> dict:
    from src.viz_phase_a import (
        plot_regime_sweep,
        plot_stability_audit,
        plot_retest_comparison,
        plot_importance_distribution,
    )

    fig_dir = cfg["paths"]["figures_dir"]
    metrics_dir = cfg["paths"]["metrics_dir"]
    ensure_dir(fig_dir); ensure_dir(metrics_dir)

    print("\n[phase_a] === A.1 K/D Regime Sweep ===")
    sweep = sweep_kd_regimes(model, cfg, device, seed=cfg["seed"])
    p = plot_regime_sweep(sweep, fig_dir=fig_dir)
    sync_to_kaggle_output(p, "figures")

    # Plot LOO distribution for a few representative K values
    for k_plot in [8, 16, 32]:
        if k_plot in sweep:
            p = plot_importance_distribution(sweep[k_plot], fig_dir=fig_dir)
            sync_to_kaggle_output(p, "figures")

    # Find the candidate good regime (highest within-task spread)
    best_k_sweep = max(sweep.keys(), key=lambda k: sweep[k]["within_task_range_mean"])
    print(f"[phase_a] Candidate best K from sweep: {best_k_sweep} "
          f"(K/D={best_k_sweep/cfg['task']['input_dim']:.2f})")

    print(f"\n[phase_a] === A.2 Stability Audit at K={best_k_sweep} ===")
    stability = loo_stability_audit(model, cfg, k=best_k_sweep, device=device,
                                    seed=cfg["seed"])
    p = plot_stability_audit(stability, sweep, fig_dir=fig_dir)
    sync_to_kaggle_output(p, "figures")

    print(f"[phase_a] Noise CV: {stability['noise_seed_cv']:.3f}  "
          f"Query CV: {stability['query_cv']:.3f}  "
          f"Resample CV: {stability['context_resample_cv']:.3f}")

    verdict = gate_a_verdict(sweep, stability)
    best_k = verdict["best_k"]
    gt_def = verdict["ground_truth"]
    print(f"\n[phase_a] GATE A preliminary: {verdict['gate']}  "
          f"best_k={best_k}  ground_truth={gt_def}")

    print(f"\n[phase_a] === A.3 Re-test Methods at K={best_k} (gt={gt_def}) ===")
    retest = retest_methods(model, cfg, k=best_k, ground_truth=gt_def,
                             device=device, seed=cfg["seed"])
    p = plot_retest_comparison(retest, fig_dir=fig_dir)
    sync_to_kaggle_output(p, "figures")

    # Compile full Phase A output
    output = {
        "sweep": sweep,
        "stability": stability,
        "verdict": verdict,
        "retest": retest,
        "config": cfg,
    }

    m_path = save_metrics(output, "phase_a", metrics_dir)
    sync_to_kaggle_output(m_path, "metrics")

    # Print GATE A summary
    print("\n" + "=" * 60)
    print(f"GATE A VERDICT: {verdict['gate']}")
    print(f"  Best regime: K={best_k} (K/D={verdict['best_kd_ratio']:.2f})")
    print(f"  Ground truth: {gt_def}  (noise CV={stability['noise_seed_cv']:.3f})")
    print(f"  Mean importance at best K: {verdict['mean_importance_at_best_k']:.4f}")
    print("  Re-test results at best K:")
    for m, s in retest["summary"].items():
        print(f"    {m:<25} Spearman={s['spearman_mean']:.4f} ± {s['spearman_std']:.4f}")
    print("=" * 60 + "\n")

    return output
