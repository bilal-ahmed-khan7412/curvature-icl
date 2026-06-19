# Results Summary — Curvature of In-Context Learning

> Updated after each phase. Headline numbers + money figures only.

---

## Status: PIVOT — Phase A in progress

**Phase 1 gate: NO-GO** (2026-06-19) → **Reframed as inconclusive** (K/D=2 regime likely degenerate)

**Phase A: awaiting Kaggle run**

---

## Phase 1 Results

| Metric | First-order | Curvature readout | 
|--------|------------|-------------------|
| Spearman ρ (adversarial) | **0.1457 ± 0.0069** | -0.0068 ± 0.0092 |

**Gap (curvature readout vs. first-order, adversarial Spearman):** -0.1526  (needed ≥ +0.05)

**Headline finding:** The curvature readout performs at chance on adversarial (collinear) contexts. First-order gradient attribution actually works on those same contexts — the premise that first-order fails on collinear pairs did not hold in this toy setting. Both findings are clean and honest.

**Metrics file:** `results/metrics/killtest_phase1_20260618_220321.json`

---

## Money Figures

*(Paths to be filled after run.)*

- `results/figures/paper_rank_agreement.png` — headline bar chart
- `results/figures/paper_topk_recall.png`
- `results/figures/paper_readout_vs_analytic.png`
- `results/figures/diag_training_curve.png`
- `results/figures/diag_pred_vs_true.png`
- `results/figures/understand_adversarial_example.png`
