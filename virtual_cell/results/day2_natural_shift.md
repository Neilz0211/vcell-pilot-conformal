# Day 2 — natural library shift (K562 genome-wide pseudobulk) — 2026-10-01

**Result: a natural library shift breaks perturbation-level CP badly, and weighting fixes it.** An
essential-enriched screening library calibrated on the natural genome-wide composition drops nominal 90 % to
**69 %**. With no overlap it drops to **54 %**. Weights on the known library design (essentiality) restore 0.90–0.91.
They also shrink over-conservative sets by about 33 % when the shift goes the other way.

**The hidden-shift failure.** Classifier-estimated weights (on GO memberships + z) do not see the support violation:
ESS 28, 5 % abstention, coverage 0.746 where the oracle abstains on 100 %. A feature-based ESS / overlap diagnostic
gives false assurance when the variable that drives the shift is only partly encoded in z. Estimated weights are
also noisy: ESS 172 at zero shift, out of n = 1,500.

Setup: 10,315 perturbations with GO terms, 20 % essential-library targets. Model: ridge on z (GO SVD 64-d + target
baseline expression), trained on 30 %. n_cal = 1,500, n_test = 600. α = 0.1. 30 seeds. Code: `src/day2.py`.

| Calibration → screening (essential share) | Unweighted cov. (radius) | Oracle cov. (radius, ESS) | Estimated cov. (ESS, abstention) |
|---|---|---|---|
| 0.20 → 0.20 | 0.901 (1.38) | 0.901 (1.38, 1500) | 0.899 (172, 0 %) |
| 0.20 → 0.50 | 0.826 | 0.909 (1.84, 968) | 0.903 (118, 0.1 %) |
| 0.20 → 0.75 | 0.759 | 0.905 (2.07, 524) | 0.909 (102, 0.1 %) |
| 0.20 → 1.00 | **0.694** | 0.908 (2.26, 303) | 0.882 (54, 0.2 %) |
| 0.75 → 0.20 | 0.966 (2.04, over-conservative) | 0.901 (1.37, 577) | 0.898 (97) |
| 0.00 → 1.00 (disjoint) | **0.541** | abstains on 100 % ✓ | **0.746 (28, 5 %)** ✗ |

Why: the median error on essential targets is 0.98, against 0.63 on non-essential ones.

**Evaluation lens (genome-wide, radius at 90 %):** mean 1.415, ridge 1.366 (−3.5 %), kNN 1.400 (−1.1 %). On
the essential library the gains were −13 % and −16 %. Most genome-wide perturbations have tiny effects, so the
models are almost no better than the mean.
