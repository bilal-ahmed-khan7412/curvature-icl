"""
Curvature-based importance scoring.

Two variants:
  (a) analytic_leverage  — closed-form hat-matrix diagonal h_i = x_i^T (X^T X)^{-1} x_i.
      This is the reference. It's well-understood statistics; the novelty is NOT here.

  (b) model_curvature_readout — extract a second-order signal from the trained
      transformer's own hidden states. The hypothesis: the attention Gram matrix
      in the last layer approximates the implicit Hessian H = Σ x_i x_i^T,
      so the diagonal of the attention-weighted Gram encodes leverage.
      This is the novel contribution — checking whether the model exposes
      second-order structure that you can read out without knowing the closed form.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.toy_icl import build_token_sequence, ICLTransformer, TaskBatch


# ---------------------------------------------------------------------------
# (a) Analytic leverage (reference)
# ---------------------------------------------------------------------------

def analytic_leverage(batch: TaskBatch, ridge: float = 1e-6) -> torch.Tensor:
    """
    h_i = x_i^T (X^T X + ridge*I)^{-1} x_i  for each context example i.

    Returns: (B, K) leverage scores.
    """
    xs = batch.xs   # (B, K, D)
    B, K, D = xs.shape

    XtX = torch.bmm(xs.transpose(1, 2), xs)                           # (B, D, D)
    reg = ridge * torch.eye(D, device=xs.device).unsqueeze(0)         # (1, D, D)
    A = XtX + reg                                                      # (B, D, D)

    # Solve A^{-1} for each batch element
    A_inv = torch.linalg.inv(A)                                        # (B, D, D)

    # h_i = x_i^T A^{-1} x_i  via einsum
    # xs: (B, K, D), A_inv: (B, D, D) -> (B, K, D) -> (B, K)
    Ainv_x = torch.bmm(xs, A_inv.transpose(1, 2))                     # (B, K, D)
    h = (Ainv_x * xs).sum(-1)                                          # (B, K)
    return h


# ---------------------------------------------------------------------------
# (b) Model-internal curvature readout
# ---------------------------------------------------------------------------

def model_curvature_readout(
    model: ICLTransformer,
    batch: TaskBatch,
    device: str = "cpu",
    ridge: float = 1e-6,
) -> torch.Tensor:
    """
    Read second-order structure from the transformer's hidden states.

    Strategy: extract the hidden representations at the x-token positions
    in the last transformer layer. These h_i vectors play the role of the
    'implicit features' the model has learned. Compute the Gram matrix
    G = Σ h_i h_i^T (analogous to X^T X) and return leverage scores
    lev_i = h_i^T (G + ridge*I)^{-1} h_i.

    This lets us ask: does the transformer's internal geometry track the
    analytic leverage of the original x's? If yes, the model encodes the
    implicit Hessian in its representations — the core claim.

    Returns: (B, K)
    """
    model.eval()
    B, K, D = batch.xs.shape

    seq = build_token_sequence(batch.xs, batch.ys, batch.x_query)   # (B, 2K+1, D_in)

    with torch.no_grad():
        _, hidden = model.forward_with_hidden(seq)   # hidden: (B, 2K+1, d_model)

    # x tokens are at even positions 0, 2, 4, ..., 2(K-1) in the sequence
    x_positions = torch.arange(0, 2 * K, 2)
    h_x = hidden[:, x_positions, :]   # (B, K, d_model)

    # Gram matrix in hidden space
    G = torch.bmm(h_x.transpose(1, 2), h_x)                           # (B, d_model, d_model)
    d = h_x.shape[-1]
    reg = ridge * torch.eye(d, device=device).unsqueeze(0)
    G_inv = torch.linalg.inv(G + reg)                                  # (B, d_model, d_model)

    # Leverage in hidden space
    Ginv_h = torch.bmm(h_x, G_inv.transpose(1, 2))                    # (B, K, d_model)
    lev = (Ginv_h * h_x).sum(-1)                                       # (B, K)
    return lev


# ---------------------------------------------------------------------------
# Curvature spectrum (for understanding / paper figures)
# ---------------------------------------------------------------------------

def curvature_spectrum(batch: TaskBatch, ridge: float = 1e-6) -> torch.Tensor:
    """
    Eigenvalues of the context Gram X^T X per task.
    Returns (B, D) sorted descending.
    Useful for understanding the implicit objective's curvature landscape.
    """
    xs = batch.xs   # (B, K, D)
    XtX = torch.bmm(xs.transpose(1, 2), xs)
    eigs = torch.linalg.eigvalsh(XtX)   # (B, D) ascending
    return eigs.flip(-1)                 # descending


def model_gram_spectrum(
    model: ICLTransformer,
    batch: TaskBatch,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Eigenvalues of the hidden-space Gram H^T H, where H are the model's
    x-token representations. Returns (B, d_model) sorted descending.
    """
    model.eval()
    B, K, _ = batch.xs.shape
    seq = build_token_sequence(batch.xs, batch.ys, batch.x_query)

    with torch.no_grad():
        _, hidden = model.forward_with_hidden(seq)

    x_positions = torch.arange(0, 2 * K, 2)
    h_x = hidden[:, x_positions, :]   # (B, K, d_model)
    G = torch.bmm(h_x.transpose(1, 2), h_x)
    eigs = torch.linalg.eigvalsh(G)
    return eigs.flip(-1)
