"""Train the tiny ICL transformer on synthetic linear regression tasks."""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.toy_icl import build_model, build_token_sequence, ols_predict, sample_tasks, TaskBatch
from src.utils import (
    ensure_dir, load_config, save_checkpoint, save_metrics,
    set_seed, StepLogger, sync_to_kaggle_output,
)


def compute_loss(model: torch.nn.Module, batch: TaskBatch) -> torch.Tensor:
    seq = build_token_sequence(batch.xs, batch.ys, batch.x_query)
    preds = model(seq)
    return F.mse_loss(preds, batch.y_query)


def evaluate(model: torch.nn.Module, cfg: dict, n_tasks: int,
             seed: int, device: str) -> dict:
    """Compute MSE + correlation with OLS on held-out tasks."""
    model.eval()
    task_cfg = cfg["task"]
    rng = np.random.default_rng(seed + 9999)
    all_mse, all_corr = [], []

    with torch.no_grad():
        for _ in range(0, n_tasks, 64):
            bs = min(64, n_tasks)
            batch = sample_tasks(bs, task_cfg["context_len"], task_cfg["input_dim"],
                                  task_cfg["noise_std"], rng, device)
            seq = build_token_sequence(batch.xs, batch.ys, batch.x_query)
            preds = model(seq)
            ols_preds = ols_predict(batch.xs, batch.ys, batch.x_query)

            mse = F.mse_loss(preds, batch.y_query).item()
            # Correlation between model preds and OLS preds (diagnostic)
            corr = torch.corrcoef(torch.stack([preds, ols_preds]))[0, 1].item()
            all_mse.append(mse)
            all_corr.append(corr)

    model.train()
    return {
        "mse_mean": float(np.mean(all_mse)),
        "mse_std": float(np.std(all_mse)),
        "ols_corr_mean": float(np.mean(all_corr)),
        "ols_corr_std": float(np.std(all_corr)),
    }


def train(cfg: dict | None = None, config_path: str = "configs/default.yaml",
          device: str | None = None) -> torch.nn.Module:
    if cfg is None:
        cfg = load_config(config_path)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] Device: {device}")

    set_seed(cfg["seed"])
    model = build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] Model params: {n_params:,}")

    train_cfg = cfg["train"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    def lr_schedule(step: int) -> float:
        # Linear warmup then cosine decay
        if step < train_cfg["warmup_steps"]:
            return step / max(1, train_cfg["warmup_steps"])
        progress = (step - train_cfg["warmup_steps"]) / max(
            1, train_cfg["n_steps"] - train_cfg["warmup_steps"])
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    ensure_dir(cfg["paths"]["checkpoint_dir"])
    logger = StepLogger(Path(cfg["paths"]["metrics_dir"]) / "train_log.jsonl")

    task_cfg = cfg["task"]
    rng = np.random.default_rng(cfg["seed"])

    history: list[dict] = []
    model.train()

    for step in tqdm(range(1, train_cfg["n_steps"] + 1), desc="Training"):
        batch = sample_tasks(
            train_cfg["batch_size"],
            task_cfg["context_len"],
            task_cfg["input_dim"],
            task_cfg["noise_std"],
            rng,
            device,
        )

        optimizer.zero_grad()
        loss = compute_loss(model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
        optimizer.step()
        scheduler.step()

        if step % train_cfg["eval_every"] == 0 or step == 1:
            eval_metrics = evaluate(model, cfg, n_tasks=256, seed=cfg["seed"], device=device)
            lr_now = optimizer.param_groups[0]["lr"]
            logger.log(step, train_loss=loss.item(), lr=lr_now, **eval_metrics)
            history.append({"step": step, "train_loss": loss.item(), **eval_metrics})

        if step % train_cfg["checkpoint_every"] == 0:
            ckpt_path = save_checkpoint(model, optimizer, step, cfg,
                                        cfg["paths"]["checkpoint_dir"])
            sync_to_kaggle_output(ckpt_path, "checkpoints")

    # Save final checkpoint and training history
    ckpt_path = save_checkpoint(model, optimizer, train_cfg["n_steps"], cfg,
                                cfg["paths"]["checkpoint_dir"])
    sync_to_kaggle_output(ckpt_path, "checkpoints")

    metrics = {
        "config": cfg,
        "seed": cfg["seed"],
        "n_params": n_params,
        "history": history,
        "final_eval": evaluate(model, cfg, n_tasks=512, seed=0, device=device),
    }
    m_path = save_metrics(metrics, "train_phase1", cfg["paths"]["metrics_dir"])
    sync_to_kaggle_output(m_path, "metrics")

    print(f"[train] Done. Final eval: {metrics['final_eval']}")
    return model


if __name__ == "__main__":
    train()
