"""
Phase 1 entrypoint: train + evaluate + kill-test.

Usage (from repo root):
    python scripts/run_phase1.py [--config configs/default.yaml] [--device cuda]

Steps:
    1. Train the toy ICL transformer
    2. Generate diagnostic plots (training curve, pred-vs-true, score distributions)
    3. Run the kill-test evaluation on random + adversarial contexts
    4. Produce paper + understanding figures
    5. Print the GATE verdict and save all results
"""

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing as a package
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from src.curvature import analytic_leverage, curvature_spectrum, model_gram_spectrum
from src.eval_importance import run_killtest
from src.importance import firstorder_importance, loo_importance, score_all_methods
from src.toy_icl import build_model, build_token_sequence, ols_predict, sample_tasks
from src.train_toy import train
from src.utils import (
    ensure_dir, load_config, load_latest_checkpoint, save_metrics,
    set_seed, sync_to_kaggle_output,
)
from src.viz import (
    plot_adversarial_example, plot_curvature_spectrum, plot_leverage_vs_context_index,
    plot_pred_vs_true, plot_rank_agreement, plot_readout_vs_analytic_corr,
    plot_score_correlation_heatmap, plot_score_distributions, plot_topk_recall,
    plot_training_curve,
)
from src.eval_importance import sample_adversarial_tasks


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--device", default=None, help="cuda or cpu (auto-detected if omitted)")
    p.add_argument("--skip-train", action="store_true",
                   help="Skip training; load latest checkpoint instead")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[phase1] Device: {device}")

    fig_dir = cfg["paths"]["figures_dir"]
    metrics_dir = cfg["paths"]["metrics_dir"]
    ensure_dir(fig_dir); ensure_dir(metrics_dir)

    # -----------------------------------------------------------------------
    # 1. Train (or load)
    # -----------------------------------------------------------------------
    if args.skip_train:
        print("[phase1] Loading latest checkpoint …")
        model = build_model(cfg, device)
        step = load_latest_checkpoint(cfg["paths"]["checkpoint_dir"], model, device=device)
        if step == 0:
            raise FileNotFoundError("No checkpoint found — run without --skip-train first.")
        history = []   # no history available
    else:
        model = train(cfg=cfg, device=device)
        # Load history from metrics if available (train() returns model but also saves JSON)
        history = []

    model.eval()

    # -----------------------------------------------------------------------
    # 2. Diagnostic: pred vs. true + OLS correlation
    # -----------------------------------------------------------------------
    print("[phase1] Generating diagnostic plots …")
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    task_cfg = cfg["task"]
    batch = sample_tasks(256, task_cfg["context_len"], task_cfg["input_dim"],
                          task_cfg["noise_std"], rng, device)

    with torch.no_grad():
        seq = build_token_sequence(batch.xs, batch.ys, batch.x_query)
        preds = model(seq).cpu().numpy()
    targets = batch.y_query.cpu().numpy()
    ols_preds = ols_predict(batch.xs, batch.ys, batch.x_query).cpu().numpy()

    p = plot_pred_vs_true(preds, targets, ols_preds, fig_dir=fig_dir)
    sync_to_kaggle_output(p, "figures")

    # Check OLS correlation — soft gate
    from scipy.stats import pearsonr
    ols_corr, _ = pearsonr(preds, ols_preds)
    print(f"[phase1] Model–OLS Pearson correlation: {ols_corr:.4f}")
    if ols_corr < 0.8:
        print("[phase1] WARNING: Model–OLS correlation is low. "
              "ICL may not have formed. Check training and extend if needed.")
    else:
        print("[phase1] Soft gate PASSED: model approximates OLS well.")

    # -----------------------------------------------------------------------
    # 3. Score distributions and correlation heatmap (diagnostic)
    # -----------------------------------------------------------------------
    scores = score_all_methods(model, batch, device)
    scores_np = {k: v.cpu().numpy() for k, v in scores.items() if k != "loo"}
    scores_np["loo"] = scores["loo"].cpu().numpy()

    p = plot_score_distributions(scores_np, fig_dir=fig_dir)
    sync_to_kaggle_output(p, "figures")
    p = plot_score_correlation_heatmap(scores_np, fig_dir=fig_dir)
    sync_to_kaggle_output(p, "figures")

    # -----------------------------------------------------------------------
    # 4. Understanding: curvature spectrum, leverage by position
    # -----------------------------------------------------------------------
    from src.curvature import curvature_spectrum, model_gram_spectrum
    eigs_data = curvature_spectrum(batch).cpu().numpy()
    eigs_model = model_gram_spectrum(model, batch, device).cpu().numpy()
    p = plot_curvature_spectrum(eigs_data, eigs_model, fig_dir=fig_dir)
    sync_to_kaggle_output(p, "figures")

    lev_a = scores_np["curvature_analytic"]
    lev_r = scores_np["curvature_readout"]
    p = plot_leverage_vs_context_index(lev_a, lev_r, fig_dir=fig_dir)
    sync_to_kaggle_output(p, "figures")

    p = plot_readout_vs_analytic_corr(lev_r, lev_a, fig_dir=fig_dir)
    sync_to_kaggle_output(p, "figures")

    # Understanding: adversarial example
    rng_adv = np.random.default_rng(cfg["seed"] + 5)
    adv_batch = sample_adversarial_tasks(
        1, task_cfg["context_len"], task_cfg["input_dim"], task_cfg["noise_std"],
        task_cfg["adv_collinearity"], rng_adv, device,
    )
    adv_loo = loo_importance(model, adv_batch, device).cpu().numpy()[0]
    adv_fo = firstorder_importance(model, adv_batch, device).cpu().numpy()[0]
    adv_lev = analytic_leverage(adv_batch).cpu().numpy()[0]
    p = plot_adversarial_example(adv_lev, adv_fo, adv_loo, fig_dir=fig_dir)
    sync_to_kaggle_output(p, "figures")

    # -----------------------------------------------------------------------
    # 5. Kill-test
    # -----------------------------------------------------------------------
    print("[phase1] Running kill-test …")
    verdict = run_killtest(model, cfg, device=device)

    m_path = save_metrics(verdict, "killtest_phase1", metrics_dir)
    sync_to_kaggle_output(m_path, "metrics")

    # Paper figures
    p = plot_rank_agreement(verdict["summary"], fig_dir=fig_dir)
    sync_to_kaggle_output(p, "figures")
    p = plot_topk_recall(verdict["summary"], fig_dir=fig_dir)
    sync_to_kaggle_output(p, "figures")

    # -----------------------------------------------------------------------
    # 6. Summary
    # -----------------------------------------------------------------------
    gate = verdict["gate"]
    gap = verdict["adv_spearman_gap_curvature_vs_firstorder"]
    print(f"\n[phase1] GATE: {gate}  (adversarial Spearman gap = {gap:.4f})")
    if gate == "GO":
        print("[phase1] Proceeding to Phase 2 is warranted.")
    else:
        print("[phase1] NO-GO: write up as a negative result. Do not rescue.")

    return verdict


if __name__ == "__main__":
    main()
