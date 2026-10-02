# Day 1 go / no-go (virtual cell, Replogle K562 essential pseudobulk) — 2026-10-01

**Verdict: conditional GO.**
- Q1 holds.
- Q2 shows real, visible under-coverage (−4 to −6 points) on categories where model errors genuinely differ. Weighting
  with a classifier trained on the GO memberships repairs it back to nominal while ESS ≥ 30.
- The effect is moderate, not dramatic. Q3's cheap uncertainty signals are weak.
- The stronger finding in this data is evaluation-lens: at the same coverage, the models' sets are only 13–16 %
  smaller than predict-the-mean.

Setup: 2,057 perturbations with GO annotations (of 2,176). Readout = the 1,000 most variable genes,
standardised per gene. Score = RMSE between predicted and observed pseudobulk response. The set is a ball of radius
q̂. z = a 64-d SVD of the target gene's GO memberships. 50 seeds, α = 0.1. Code: `src/day1.py`; numbers in
`results/day1_metrics.json`.

## Q1: perturbation-level split CP, random hold-out
| Model | Coverage (mean ± sd) | Set radius q̂ (lower is better) |
|---|---|---|
| predict-the-mean | 0.902 ± 0.016 | 1.535 |
| ridge(z) | 0.898 ± 0.017 | 1.340 (−13 %) |
| kNN(z) | 0.894 ± 0.020 | 1.289 (−16 %) |

Coverage is at the 0.90 nominal and the sets are non-trivial, so Q1 holds. At equal coverage the models barely beat
the mean, consistent with the literature showing deep models are no better than simple baselines.

## Q2: library shift along GO categories (ridge model)
- **Construction.** The calibration library down-weights category C (odds e^{−β/2}) and the screening library
  up-weights it (e^{+β/2}). The oracle density ratio is e^{±β}.
- **Estimators.** Two classifiers estimate the weights on an independent draw of both libraries: logistic regression
  on z ("est-z") and on the raw GO memberships ("est-GO").

GO:0003723 (RNA binding, n = 628; median error inside the category 0.86, outside 0.59):

| β | Unweighted | Oracle-weighted (ESS) | est-GO (ESS, abstention) | est-z (ESS) |
|---|---|---|---|---|
| 0 | 0.910 | 0.910 (524) | 0.918 (175, 0 %) | 0.909 (502) |
| 1 | 0.882 | 0.903 (225) | 0.907 (99, 0.1 %) | 0.893 (361) |
| 2 | **0.856** | 0.909 (62) | **0.904 (41, 1.1 %)** | 0.882 (165) |
| 3 | **0.846** | 0.924 (20) | 0.897 (18, 6.4 %) | 0.879 (120) |

- **Under-coverage is real but moderate.** At strong shift it is 4.4–5.4 points. Categories with similar errors
  (nucleolus, extracellular exosome, membrane) lose only 1–3 points.
- **Categories with smaller errors over-cover** (mitochondrion, plasma membrane: 0.92 unweighted). Weighting pulls
  them back to nominal.
- **est-GO repairs coverage; est-z does not.** The est-z weights are too flat: their ESS is far above the oracle's.
  The weight model has to see the features that actually define the libraries. This matches the Prop 4 / ESS story.
- **Small ESS makes the oracle over-cover** (0.93–0.95 at ESS ≈ 10–20). The sets get conservative and fluctuate,
  which fits the √(C_lib/n) fluctuation in Thm 3. The abstention rate rises with shift (est-GO: 6–7 % at β = 3).

## Q3: can an uncertainty proxy predict the top-quartile errors (AUC, 20 seeds)?
| Proxy | AUC |
|---|---|
| Distance in z to the 5 nearest training perturbations | 0.509 ± 0.018 (useless) |
| Bootstrap ridge spread | 0.622 ± 0.018 (just above 0.6) |

## Against the go criterion
- Q1 holds: ✓.
- Q2 needs visible under-coverage repaired by weighting at ESS ≥ 30. ✓ for RNA binding, β = 2: −5.4 points
  (0.856), est-GO back to 0.904 at ESS 41. The size of the effect is borderline: it relies on categories whose
  errors really differ.
- Q3 needs AUC > 0.6: only bootstrap spread just clears it.

## Not done on day 1
- Cell-level vs perturbation-level CP (E1c): needs the single-cell file (10 GB).
- GEARS and other deep models: needs torch_geometric; not installed.
- Combinatorial perturbations (Norman).
- Unseen contexts (Prop 5).

## Next steps (proposed)
1. **A natural, unconstructed library shift.**
   - Calibrate on the essential library and screen the genome-wide library (`K562_gwps_normalized_bulk_01.h5ad` is
     already downloaded, ~10k perturbations). That shift is defined by the experiment design itself.
   - Also test RPE1 → K562 as a context shift with K = 2, for Prop 5.
2. **Add a real deep model.** Either GEARS (with its logvar head) or STATE inference (licence to check). Repeat Q1
   and Q3 to see whether deep models know better when they are wrong (E3).
3. **Report the evaluation-lens result systematically.** Set size at equal coverage, model vs mean, across datasets.
   It may be the more striking headline.
4. Write up Thm 1–3 (adapted from the OCP paper's proofs) and run the needle simulation.
