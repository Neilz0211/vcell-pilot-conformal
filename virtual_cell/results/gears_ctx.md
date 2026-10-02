# GEARS in the transfer matrix (15 epochs, 3 seeds, α = 0.1) — 2026-10-02

Code: `gears/gears_ctx.py` (isolated env, local RTX 3090); numbers in `results/gears_ctx_metrics.json`.
- GEARS is trained on pseudobulk responses: one "cell" per perturbation, zero-vector controls, model selection on
  overall validation MSE.
- Mean and ridge10 use identical splits.

Cells give the source radius qA (lower = more efficient in the source line) / coverage in the target line of the
source-calibrated set.

| Pair | mean | ridge10 | GEARS |
|---|---|---|---|
| K562 → RPE1 | 1.513 / 0.852 | 1.373 / 0.791 | 1.562 / 0.826 |
| RPE1 → K562 | 1.409 / 0.863 | 1.253 / 0.793 | 1.613 / 0.912 |
| HepG2 → Jurkat | 1.565 / 0.928 | 1.412 / 0.898 | 1.596 / 0.915 |
| Jurkat → HepG2 | 1.445 / 0.863 | 1.364 / 0.854 | 1.515 / 0.866 |

## Reading
1. **GEARS, trained this way, is less efficient than predict-the-mean in the source line** in all four directions
   (qA 2–14 % larger).
   - This is consistent with Ahlmann-Eltze et al. 2025 (deep perturbation models do not beat linear baselines).
   - It may also partly reflect our pseudobulk adaptation of GEARS: one cell per condition, ~700 training
     perturbations, default hyperparameters. It is **not** a tuned GEARS.
2. **GEARS sits on the F2 curve rather than breaking it.**
   - It is the least source-efficient model in each direction, and its transferred coverage is not below the mean's,
     except K562 → RPE1 (0.826 vs 0.852).
   - RPE1 → K562 is the clearest case: GEARS has the widest source set and is the only model near nominal (0.912).
   - So the inversion "tighter source set → worse transferred coverage" holds across mean / ridge / GEARS. But GEARS
     does not test the regime "a *better* deep model is more overconfident".
3. **Pilot-anchored sets are valid for GEARS too** (0.88–0.93 with 3 seeds; noisy). The exact adapted ball tightens
   GEARS by 3–8 %.
4. **Still to do for reviewer hygiene:** a deep model that actually beats ridge in the source line.
   - Options: STATE (licence and size to check) or scGPT fine-tuning (GPU-heavy).
   - Alternatively, tune GEARS (more epochs, hidden size) and train it on single cells. That needs single-cell data:
     on the pod, or subsampled locally.

## Tuned GEARS (MSE loss, 60 epochs, LR step 20, lr 3e-3; 3 seeds) — `results/gears_tuned_metrics.json`

| Pair | mean qA / covB / qB* | ridge10 qA / covB / qB* | **GEARS** qA / covB / qB* |
|---|---|---|---|
| K562 → RPE1 | 1.513 / 0.852 / 1.646 | 1.373 / 0.791 / 1.608 | 1.391 / **0.766** / 1.683 |
| RPE1 → K562 | 1.409 / 0.863 / 1.571 | 1.253 / 0.793 / 1.573 | 1.303 / **0.774** / 1.641 |
| HepG2 → Jurkat | 1.565 / 0.928 / 1.473 | 1.412 / 0.898 / 1.451 | 1.384 / **0.852** / 1.551 |
| Jurkat → HepG2 | 1.445 / 0.863 / 1.643 | 1.364 / 0.854 / 1.574 | 1.379 / **0.853** / 1.611 |

qB* = oracle radius in the target line (full-screen calibration), i.e. true target-line efficiency.

### Reading
- **Tuned GEARS is on par with ridge in the source line**: within ±4 %, better in HepG2 → Jurkat, and 26–60 %
  closer to ridge than to the mean.
- **It has the lowest transferred coverage in every direction** (0.766 / 0.774 / 0.852 / 0.853).
- **The deep model is more overconfident than its source efficiency alone predicts.**
  - Its target-line efficiency is the worst of the three in 3 of 4 directions; in Jurkat → HepG2 it falls between
    ridge and the mean.
  - Its oracle target radius exceeds even predict-the-mean in 3 of 4 directions (1.683 vs 1.646; 1.641 vs 1.571;
    1.551 vs 1.473).
  - So more of what GEARS learns in the source line is context-specific: mechanism M2 (non-transferring signal) is
    larger for the deep model.
- **Pilot-anchored and exact adapted sets stay valid** (0.87–0.92). Three seeds is noisy; the 0.867–0.873 cells are
  within split noise of pilot-only exact CP, which has a guarantee.
- This closes the "where are the deep models?" gap for GEARS. STATE (licence: non-commercial; public checkpoint
  trained on these four lines, so it would leak the target) stays optional.
