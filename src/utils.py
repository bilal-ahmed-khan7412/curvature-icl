"""Shared utilities: seeds, config loading, IO, logging, Kaggle persistence."""

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_metrics(metrics: dict, name: str, metrics_dir: str = "results/metrics") -> Path:
    ensure_dir(metrics_dir)
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = Path(metrics_dir) / f"{name}_{ts}.json"
    # Never silently overwrite; timestamp guarantees uniqueness
    with open(fname, "w") as f:
        json.dump(metrics, f, indent=2, default=_json_default)
    print(f"[utils] Metrics saved → {fname}")
    return fname


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Not serializable: {type(obj)}")


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    step: int, cfg: dict, ckpt_dir: str = "results/checkpoints") -> Path:
    ensure_dir(ckpt_dir)
    fname = Path(ckpt_dir) / f"ckpt_step{step:07d}.pt"
    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": cfg,
    }, fname)
    print(f"[utils] Checkpoint saved → {fname}")
    return fname


def load_checkpoint(path: str | Path, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer | None = None,
                    device: str = "cpu") -> int:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    step = ckpt.get("step", 0)
    print(f"[utils] Checkpoint loaded from {path} (step={step})")
    return step


def load_latest_checkpoint(ckpt_dir: str, model: torch.nn.Module,
                            optimizer: torch.optim.Optimizer | None = None,
                            device: str = "cpu") -> int:
    ckpt_dir = Path(ckpt_dir)
    ckpts = sorted(ckpt_dir.glob("ckpt_step*.pt"))
    if not ckpts:
        return 0
    return load_checkpoint(ckpts[-1], model, optimizer, device)


# ---------------------------------------------------------------------------
# Kaggle persistence helpers
# ---------------------------------------------------------------------------

def is_kaggle() -> bool:
    return os.path.exists("/kaggle")


def kaggle_output_dir() -> Path:
    """On Kaggle, /kaggle/working is the output dir that can be saved as a Dataset."""
    if is_kaggle():
        return Path("/kaggle/working")
    return Path("results")


def sync_to_kaggle_output(src: str | Path, subdir: str = "") -> None:
    """Copy a file to /kaggle/working/<subdir>/ so it persists as a Dataset output."""
    if not is_kaggle():
        return
    dest = kaggle_output_dir() / subdir
    dest.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(src, dest / Path(src).name)
    print(f"[utils] Synced {src} → {dest}")


# ---------------------------------------------------------------------------
# Simple step logger
# ---------------------------------------------------------------------------

class StepLogger:
    def __init__(self, log_path: str | Path | None = None):
        self.entries: list[dict] = []
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            ensure_dir(self.log_path.parent)

    def log(self, step: int, **kwargs) -> None:
        entry = {"step": step, "time": time.strftime("%H:%M:%S"), **kwargs}
        self.entries.append(entry)
        parts = [f"step={step}"] + [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                     for k, v in kwargs.items()]
        print("[log] " + "  ".join(parts))
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry, default=_json_default) + "\n")
