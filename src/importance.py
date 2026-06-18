"""
Importance scoring methods for in-context examples.

Three methods:
  1. LOO (leave-one-out) — ground-truth importance via context ablation.
  2. FirstOrder — gradient attribution baseline (∂ŷ_q / ∂y_i evaluated at context).
  3. CurvatureImportance — wrapper that calls curvature.py; kept here for unified API.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.toy_icl import build_token_sequence, ICLTransformer, TaskBatch


# ---------------------------------------------------------------------------
# 1. LOO ground-truth
# ---------------------------------------------------------------------------

def loo_importance(
    model: ICLTransformer,
    batch: TaskBatch,
    device: str = "cpu",
) -> torch.Tensor:
    """
    For each context example i in [0, K), compute the absolute change in
    the model's query prediction when example i is dropped from the context.

    Returns: (B, K) tensor of |Δŷ_q| for each (task, example).
    """
    model.eval()
    B, K, D = batch.xs.shape

    # Full-context prediction
    with torch.no_grad():
        seq_full = build_token_sequence(batch.xs, batch.ys, batch.x_query)
        preds_full = model(seq_full)   # (B,)

    scores = torch.zeros(B, K, device=device)

    with torch.no_grad():
        for i in range(K):
            # Drop example i
            xs_drop = torch.cat([batch.xs[:, :i], batch.xs[:, i+1:]], dim=1)
            ys_drop = torch.cat([batch.ys[:, :i], batch.ys[:, i+1:]], dim=1)
            seq_drop = build_token_sequence(xs_drop, ys_drop, batch.x_query)
            preds_drop = model(seq_drop)   # (B,)
            scores[:, i] = (preds_full - preds_drop).abs()

    return scores   # (B, K)


# ---------------------------------------------------------------------------
# 2. First-order gradient attribution baseline
# ---------------------------------------------------------------------------

def firstorder_importance(
    model: ICLTransformer,
    batch: TaskBatch,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Gradient of the query prediction w.r.t. each y_i in the context.
    Score for example i = |∂ŷ_q / ∂y_i|.

    This is the standard first-order / influence-style baseline.
    Returns: (B, K)
    """
    model.eval()
    B, K, D = batch.xs.shape

    # We need gradients w.r.t. the y tokens in the sequence.
    # Build a differentiable version of the sequence.
    ys_param = batch.ys.detach().clone().requires_grad_(True)   # (B, K)
    seq = build_token_sequence(batch.xs, ys_param, batch.x_query)

    preds = model(seq)   # (B,)
    # Sum over batch to allow a single backward (each pred only touches its own ys)
    preds.sum().backward()

    scores = ys_param.grad.abs()   # (B, K)
    return scores.detach()


# ---------------------------------------------------------------------------
# 3. Unified scoring interface
# ---------------------------------------------------------------------------

METHODS = ("loo", "firstorder", "curvature_analytic", "curvature_readout")


def score_all_methods(
    model: ICLTransformer,
    batch: TaskBatch,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """
    Returns dict method_name -> (B, K) importance scores.
    Curvature methods are imported lazily to avoid circular imports.
    """
    from src.curvature import analytic_leverage, model_curvature_readout

    results: dict[str, torch.Tensor] = {}
    results["loo"] = loo_importance(model, batch, device)
    results["firstorder"] = firstorder_importance(model, batch, device)
    results["curvature_analytic"] = analytic_leverage(batch)
    results["curvature_readout"] = model_curvature_readout(model, batch, device)
    return results
