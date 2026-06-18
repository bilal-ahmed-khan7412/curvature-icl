# CLAUDE.md — Curvature of In-Context Learning (Living Project Memory)

> Update this file at the end of every session. It is the project's memory across resets.

---

## 1. Goal

**Hypothesis:** The second-order (curvature) structure of a transformer's implicit in-context learning objective predicts which in-context examples are load-bearing — and does so better than first-order methods (influence functions / gradient attribution) in the adversarial case where examples are individually low-importance but jointly important (collinear / redundant / high-leverage).

**Kill criterion (Section 4.4):** If the curvature/leverage readout does NOT beat the first-order baseline on adversarial (collinear-pair) contexts by a margin ≥ 0.05 Spearman ρ (surviving error bars over 5 seeds), the project stops and becomes a negative-result paper (target: ICBINB / TMLR).

---

## 2. Current Status

**Phase:** 1 — scaffolding complete, not yet run on Kaggle.

**Done:**
- Full repo structure created locally.
- All Phase 1 source modules written: `toy_icl.py`, `train_toy.py`, `importance.py`, `curvature.py`, `eval_importance.py`, `viz.py`, `utils.py`.
- Config (`configs/default.yaml`), entrypoint (`scripts/run_phase1.py`), Kaggle runner notebook (`notebooks/kaggle_runner.ipynb`).
- `requirements.txt`, `CLAUDE.md`, `results/SUMMARY.md`.

**In progress:** None — awaiting first Kaggle run.

**Not yet done:**
- Train on Kaggle T4.
- Verify soft gate (model ≈ OLS).
- Run kill-test; get gate verdict.
- `src/__init__.py` (not strictly needed; scripts use sys.path).

---

## 3. How to Run on Kaggle

### Fresh session setup (≤ 2 min)

1. **Enable GPU**: Notebook Settings → Accelerator → T4 GPU.
2. **Clone / update repo** (Cell 1 of `notebooks/kaggle_runner.ipynb`):
   ```bash
   git clone https://github.com/YOUR_USERNAME/curvature-icl.git /kaggle/working/curvature-icl
   cd /kaggle/working/curvature-icl
   pip install -q -r requirements.txt
   ```
3. **Config override**: Cell 2 redirects all output paths to `/kaggle/working/results/` so they survive as Kaggle Dataset output.
4. **If resuming**: attach the prior run's output Dataset; set `SKIP_TRAIN = True` in Cell 3.

### Phase 1 runtime estimates (T4)
| Step | Estimated time |
|------|---------------|
| Training (100k steps, batch 64) | ~40–60 min |
| Diagnostic plots | < 2 min |
| Kill-test eval (1500 tasks) | ~20–40 min |
| **Total** | ~1–2 hr |

### Entrypoint commands (alternative to notebook)
```bash
# From repo root
python scripts/run_phase1.py --device cuda

# Skip training (load checkpoint)
python scripts/run_phase1.py --device cuda --skip-train
```

---

## 4. Key Decisions & Rationale

| Decision | Choice | Why |
|---|---|---|
| Task type | Linear regression ICL (Garg et al. 2022) | Well-studied; closed-form OLS lets us verify ICL has formed. |
| Model | 4-layer, 4-head, d_model=64 transformer | Small enough to train in <1 hr on T4; big enough to learn ICL. |
| Input dim D | 8 | K=16 context > D=8 so OLS is overdetermined; leverage scores are non-trivial. |
| Context len K | 16 | Enough examples for leverage to vary meaningfully. |
| Noise std | 0.1 | Low noise so LOO signal is clean; model can learn OLS well. |
| Curvature readout | Leverage from hidden-space Gram at x-token positions | Cleanest analog to analytic leverage; avoids needing attention weights. |
| Adversarial construction | Collinear pairs (ρ=0.95) with Gram–Schmidt orthogonalization | Creates genuinely jointly-important examples with first-order scores suppressed. |
| Kill threshold | Spearman gap ≥ 0.05 over 5 seeds | Conservative; requires consistent improvement, not a single lucky seed. |
| Seed list | [42, 43, 44, 45, 46] | Fixed before seeing results; not cherry-picked. |

---

## 5. Results Log

*(No results yet — first Kaggle run pending.)*

---

## 6. Open Problems / Known Issues / Suspected Bugs

- `model_curvature_readout` extracts leverage from the **final-layer** hidden states at x-token positions. This is a design choice, not validated — the most informative layer may differ. If the readout underperforms, try averaging over all layers or using the penultimate layer.
- The `forward_with_hidden` method returns the full sequence hidden states. On long contexts the attention mask is causal, so x-token representations do not "see" later y-tokens — this is intentional but worth verifying.
- No `__init__.py` in `src/` — scripts use `sys.path.insert`. This is fine for Kaggle but fragile if the package is ever installed. Add `__init__.py` if needed.
- The OLS soft-gate threshold (Pearson > 0.8) is a judgment call. If the model converges to a different in-context algorithm (e.g., gradient-descent-like), this threshold may need adjustment.

---

## 7. Next Steps

1. **Push repo to GitHub** (update `REPO_URL` in Cell 1 of the Kaggle notebook).
2. **Run Phase 1 on Kaggle** (Cells 1–6): train → soft gate → kill-test → gate verdict.
3. **Update this file** after the run with: training curve stats, OLS correlation, kill-test Spearman numbers, gate verdict, figure paths.
