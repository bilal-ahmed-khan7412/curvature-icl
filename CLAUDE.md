# CLAUDE.md — Curvature / Importance of In-Context Learning (Living Project Memory)

> Update this file at the end of every session. It is the project's memory across resets.

---

## 1. Goal (UPDATED 2026-06-19 — PIVOT)

**Original hypothesis (Phase 1):** Second-order curvature/leverage predicts load-bearing ICL examples better than first-order methods.

**Phase 1 result:** All methods near-zero: first-order Spearman 0.146 (adversarial) / 0.066 (random); analytic leverage 0.015; curvature readout -0.007. Crucially, even the analytic closed-form leverage failed — which means the failure is in the *signal choice*, not the readout implementation.

**Why this is inconclusive, not a negative result:** The original kill-test used K=16, D=8 → K/D=2 (over-determined). In an over-determined regime, any single dropped example barely shifts the implicit fit — LOO importance collapses toward zero for *all* examples, so no method can correlate with it. The hypothesis was never tested on a meaningful target.

**Pivoted goal:** Find the regime where in-context example importance is a well-defined, stable quantity (Phase A), then build a method that actually predicts it (Phase B), then scale to real models (Phase C).

**New headline hypothesis (Phase B):** Importance is *query-conditioned*, not geometric. The right signal is `s_i = x_q^T (X^TX)^{-1} x_i` (how aligned example i is with the query through the Gram), not `h_i = x_i^T (X^TX)^{-1} x_i` (leverage). First-order gradient attribution captures a noisy version of this; a clean closed-form + model-internal readout should do better.

**Kill criterion (Phase B):** Method must achieve Spearman ≥ 0.4 and clearly beat first-order, with error bars over seeds.

---

## 2. Current Status

**Phase:** A — audit the ground truth. Code written, awaiting Kaggle run.

**Done (Phase 1):**
- Full repo trained on Kaggle T4; all diagnostic + paper figures generated.
- Kill-test confirmed: all methods near-zero in K=16, D=8 regime.
- Key diagnostic: analytic leverage also fails → leverage is the wrong signal, not the readout.

**Done (Phase A — local):**
- `src/phase_a.py`: K/D regime sweep, LOO stability audit, exact Shapley (K≤12) / MC-Shapley (K>12), re-test of 3 original methods in good regime.
- `scripts/run_phase_a.py`: CLI entrypoint for Phase A.
- Updated `notebooks/kaggle_runner.ipynb` with Phase A cells.

**Not yet done:**
- Run Phase A on Kaggle T4.
- GATE A verdict.
- Phase B / C (conditional on GATE A).

---

## 3. How to Run on Kaggle

### Fresh session setup (≤ 2 min)
1. Enable GPU: T4 x1.
2. Cell 1: clone/pull `https://github.com/bilal-ahmed-khan7412/curvature-icl.git` (username: bilalkhan8068 on API).
3. Cell 2: override output paths to `/kaggle/working/results/`.
4. **Phase A**: run Phase A cells (load checkpoint → sweep → stability → GATE A).
5. **Phase B** (only on GATE A GO): Phase B cells.

### Phase A runtime estimates (T4)
| Step | Estimated time |
|------|---------------|
| Load checkpoint | < 1 min |
| K/D regime sweep (8 regimes × 500 tasks) | ~15–25 min |
| LOO stability audit | ~5–10 min |
| Shapley (K≤12, 500 tasks) | ~10–20 min |
| Re-test 3 methods in good regime | ~5 min |
| **Total Phase A** | ~40–60 min |

### Entrypoint
```bash
python scripts/run_phase_a.py --device cuda --skip-train
```

---

## 4. Key Decisions & Rationale

| Decision | Choice | Why |
|---|---|---|
| K/D sweep range | K=4,6,8,10,12,16,24,32 with D=8 | Covers under-determined (K/D<1) through very over-determined (K/D=4) |
| Shapley implementation | Exact for K≤12 (2^K subsets), MC for K>12 (~2000 permutations) | Toy model is fast; exact is preferable where feasible |
| Shapley value function | v(S) = -MSE of model predicting y_q with context S | Directly measures contribution to query prediction |
| Good-regime hypothesis | K ≈ D (K/D ~1–1.5) | Where fit is determined but not massively over-determined |
| Stability criterion | LOO std across noise seeds < 0.5 × LOO mean | Example importance must be signal, not noise |
| Phase 1 K/D | K=16, D=8 → K/D=2 | Over-determined; LOO importance was near-zero; result inconclusive |
| Phase B lead signal | `s_i = x_q^T (X^TX)^{-1} x_i` | Query-conditioned relevance; first-order approximates this; should be cleanly predictive |

---

## 5. Results Log

- **2026-06-19 — Phase 1 kill-test (Kaggle T4, K=16, D=8, K/D=2)**
  - Model–OLS Pearson: 0.979 (ICL formed ✓).
  - Adversarial Spearman — curvature_readout: -0.007; firstorder: 0.146; curvature_analytic: 0.015.
  - Gate: NO-GO (all methods near-zero).
  - **Interpretation (revised):** K/D=2 is over-determined. LOO importance is noise-dominated. Result is *inconclusive*, not a negative result on the curvature hypothesis. Re-testing in good regime (Phase A) required.

*(Phase A results pending Kaggle run.)*

---

## 6. Open Problems / Known Issues

- The original kill-test's adversarial construction (collinear pairs) was correct, but the regime (K/D=2) may have suppressed the very effect being tested. Need Phase A to confirm.
- Shapley is expensive for K>12; MC sampling introduces noise — use enough permutations (≥2000) and check convergence.
- The model-internal readout in Phase B may need to probe multiple layers, not just the final layer. Keep this as a tunable.

---

## 7. Next Steps

1. **Push to GitHub** and run Phase A on Kaggle (Cells 8–12 in the notebook).
2. **STOP at GATE A**: report the regime, ground-truth stability, and re-test results here before proceeding.
3. Update this file and `SUMMARY.md` after GATE A.
