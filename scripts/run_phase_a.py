"""
Phase A entrypoint: K/D regime audit + stability check + method re-test.

Usage (from repo root):
    python scripts/run_phase_a.py [--config configs/default.yaml] [--device cuda]

Loads the trained checkpoint from Phase 1, runs the full Phase A pipeline,
prints the GATE A verdict, and saves all figures + metrics.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.phase_a import run_phase_a
from src.toy_icl import build_model
from src.utils import ensure_dir, load_config, load_latest_checkpoint, set_seed
from src.viz_phase_a import plot_gate_a_summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--device", default=None)
    p.add_argument("--skip-train", action="store_true", default=True,
                   help="Always true for Phase A — loads existing checkpoint")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[phase_a] Device: {device}")

    ensure_dir(cfg["paths"]["figures_dir"])
    ensure_dir(cfg["paths"]["metrics_dir"])

    set_seed(cfg["seed"])
    model = build_model(cfg, device)
    step = load_latest_checkpoint(cfg["paths"]["checkpoint_dir"], model, device=device)
    if step == 0:
        raise FileNotFoundError(
            "No checkpoint found. Run scripts/run_phase1.py --device cuda first.")
    print(f"[phase_a] Loaded checkpoint at step {step}")
    model.eval()

    output = run_phase_a(model, cfg, device=device)

    # Gate A summary figure
    from src.utils import sync_to_kaggle_output
    p = plot_gate_a_summary(
        output["sweep"], output["stability"], output["retest"],
        output["verdict"], fig_dir=cfg["paths"]["figures_dir"]
    )
    sync_to_kaggle_output(p, "figures")

    verdict = output["verdict"]
    print(f"\n[phase_a] GATE A: {verdict['gate']}")
    print(f"  Use K={verdict['best_k']} with ground truth='{verdict['ground_truth']}'")
    if verdict["gate"] == "NO-GO":
        print("[phase_a] LOO is degenerate in all regimes. Report before proceeding.")
    else:
        print("[phase_a] Proceed to Phase B with the above regime + ground truth.")

    return output


if __name__ == "__main__":
    main()
