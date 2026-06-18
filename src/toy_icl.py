"""
Synthetic in-context linear regression + small decoder transformer.

Data format (Garg et al. 2022 style):
  A context of K (x_i, y_i) pairs, followed by a query x_q.
  Tokens: [x_0, y_0, x_1, y_1, ..., x_{K-1}, y_{K-1}, x_q]
  Each x token is D-dim; each y token is 1-dim scalar (padded to D).
  Model predicts y_q from the final token's output.
"""

import math
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Task / data
# ---------------------------------------------------------------------------

@dataclass
class TaskBatch:
    xs: torch.Tensor      # (B, K, D) in-context x's
    ys: torch.Tensor      # (B, K)    in-context y's
    x_query: torch.Tensor # (B, D)
    y_query: torch.Tensor # (B,)
    w: torch.Tensor       # (B, D)    ground-truth weight (for analysis)


def sample_tasks(
    batch_size: int,
    context_len: int,
    input_dim: int,
    noise_std: float = 0.1,
    rng: np.random.Generator | None = None,
    device: str = "cpu",
) -> TaskBatch:
    if rng is None:
        rng = np.random.default_rng()

    w = rng.standard_normal((batch_size, input_dim)).astype(np.float32)
    xs = rng.standard_normal((batch_size, context_len, input_dim)).astype(np.float32)
    noise = rng.standard_normal((batch_size, context_len)).astype(np.float32) * noise_std
    ys = (xs * w[:, None, :]).sum(-1) + noise   # (B, K)

    x_query = rng.standard_normal((batch_size, input_dim)).astype(np.float32)
    y_query = (x_query * w).sum(-1).astype(np.float32)   # noiseless query target

    return TaskBatch(
        xs=torch.from_numpy(xs).to(device),
        ys=torch.from_numpy(ys).to(device),
        x_query=torch.from_numpy(x_query).to(device),
        y_query=torch.from_numpy(y_query).to(device),
        w=torch.from_numpy(w).to(device),
    )


def task_iterator(
    n_steps: int,
    batch_size: int,
    context_len: int,
    input_dim: int,
    noise_std: float,
    seed: int,
    device: str = "cpu",
) -> Iterator[TaskBatch]:
    rng = np.random.default_rng(seed)
    for _ in range(n_steps):
        yield sample_tasks(batch_size, context_len, input_dim, noise_std, rng, device)


def build_token_sequence(xs: torch.Tensor, ys: torch.Tensor,
                          x_query: torch.Tensor) -> torch.Tensor:
    """
    Interleave context (x, y) pairs then append x_query.
    Returns shape (B, 2K+1, D) where y tokens are zero-padded to D dims
    with y in the first position.
    """
    B, K, D = xs.shape
    # y tokens: scalar in dim-0, rest zeros
    y_tokens = torch.zeros(B, K, D, device=xs.device, dtype=xs.dtype)
    y_tokens[:, :, 0] = ys   # (B, K)

    # Interleave: x_0, y_0, x_1, y_1, ...
    seq = torch.stack([xs, y_tokens], dim=2)   # (B, K, 2, D)
    seq = seq.view(B, 2 * K, D)                # (B, 2K, D)

    # Append query (no y token)
    query_tok = x_query.unsqueeze(1)           # (B, 1, D)
    seq = torch.cat([seq, query_tok], dim=1)   # (B, 2K+1, D)
    return seq


# ---------------------------------------------------------------------------
# Closed-form OLS / ridge reference (for diagnostic comparison)
# ---------------------------------------------------------------------------

def ols_predict(xs: torch.Tensor, ys: torch.Tensor,
                x_query: torch.Tensor, ridge: float = 1e-6) -> torch.Tensor:
    """
    Ridge regression closed form: w = (X^T X + ridge*I)^{-1} X^T y.
    xs: (B, K, D), ys: (B, K), x_query: (B, D) -> predictions (B,)
    """
    B, K, D = xs.shape
    XtX = torch.bmm(xs.transpose(1, 2), xs)                       # (B, D, D)
    reg = ridge * torch.eye(D, device=xs.device).unsqueeze(0)
    A = XtX + reg                                                  # (B, D, D)
    Xty = torch.bmm(xs.transpose(1, 2), ys.unsqueeze(-1))         # (B, D, 1)
    w_hat = torch.linalg.solve(A, Xty).squeeze(-1)                # (B, D)
    return (w_hat * x_query).sum(-1)                               # (B,)


# ---------------------------------------------------------------------------
# Transformer model
# ---------------------------------------------------------------------------

class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class ICLTransformer(nn.Module):
    """
    Small causal decoder transformer for in-context linear regression.
    Input: token sequence (B, T, D); output: scalar prediction at last token.
    """
    def __init__(self, input_dim: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int, dropout: float = 0.0,
                 max_context_len: int = 64):
        super().__init__()
        max_seq = 2 * max_context_len + 1

        self.token_proj = nn.Linear(input_dim, d_model)
        self.pe = SinusoidalPE(d_model, max_seq)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (B, T, D) -> predictions (B,) from last token."""
        B, T, _ = seq.shape
        h = self.pe(self.token_proj(seq))
        mask = self._causal_mask(T, seq.device)
        h = self.transformer(h, mask=mask, is_causal=True)
        return self.head(h[:, -1]).squeeze(-1)   # scalar per batch item

    def forward_with_hidden(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (predictions, final-layer hidden states (B, T, d_model))."""
        B, T, _ = seq.shape
        h = self.pe(self.token_proj(seq))
        mask = self._causal_mask(T, seq.device)
        h = self.transformer(h, mask=mask, is_causal=True)
        preds = self.head(h[:, -1]).squeeze(-1)
        return preds, h


def build_model(cfg: dict, device: str = "cpu") -> ICLTransformer:
    m = cfg["model"]
    t = cfg["task"]
    model = ICLTransformer(
        input_dim=t["input_dim"],
        d_model=m["d_model"],
        n_heads=m["n_heads"],
        n_layers=m["n_layers"],
        d_ff=m["d_ff"],
        dropout=m["dropout"],
        max_context_len=t["context_len"],
    )
    return model.to(device)
