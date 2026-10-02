# Perturbation-specific readout (20 seeds, ridge10, α = 0.1, 12 directed cross-line pairs) — 2026-10-02

Code: `src/specific.py`; numbers in `results/specific_metrics.json`.

**Question.** Does perturbation p respond differently from the typical perturbation in the target line?
- The typical response μ_B is estimated by the pilot mean.
- p is *certified* when its 90 % ball excludes μ̂_B.
- Truth (evaluation only): the true perturbation-specific effect ‖Y_B(p) − μ_B‖ is in the top 25 % of test
  perturbations.
- K = readout space: 0 = all 1,000 genes; K = top-K source-line principal components.

## Results

Averaged over the 12 pairs. Columns per method: certified per 1,000 perturbations, false certifications per 1,000,
recall of true responders.

| K | m | Pilot-only: certified / false / recall | Adapted (Prop A ball): certified / false / recall | Mean coverage pilot / adapted |
|---|---|---|---|---|
| all genes | 50 | 0 / 0 / 0.000 | 0 / 0 / 0.000 | 0.898 / 0.906 |
| 1 PC | 20 | 36 / 13 / 0.088 | 19 / 6 / 0.053 | 0.905 / 0.919 |
| 1 PC | 50 | 39 / 14 / 0.097 | 23 / 8 / 0.064 | 0.901 / 0.902 |
| 5 PCs | 50 | 18 / 3 / 0.054 | 5 / 1 / 0.017 | 0.899 / 0.907 |
| 20 PCs | 50 | 8 / 1 / 0.023 | 2 / 0 / 0.005 | 0.897 / 0.903 |

## Reading

1. **Neither method certifies many perturbation-specific effects in a new cell line.**
   - At most about 4 % of perturbations are certified, on a one-dimensional readout.
   - In the full 1,000-gene space, none are.
   - Recall of the top-25 % true responders is ≤ 10 %.
   - This is a limit of the source model, not of calibration. Ridge's predicted specific effects are small relative to
     the cross-line residual noise, consistent with Systema and Ahlmann-Eltze.
2. **Wording correction (GPT, 2026-10-02).** The common shift cancels from the adapted ball's centre (centre − μ̂ = f(z_p) − mean pilot prediction). The pilot-only ball keeps the common bias in its centre, but that does not invalidate it: its coverage holds and its calls are not "false by construction". Both compare against an *estimated* μ̂ whose error is not propagated, so a certification is relative to the pilot mean.
3. **The adapted sets are more conservative.**
   - They make about 40–70 % fewer certifications and about 40–100 % fewer false certifications.
   - They also find fewer true responders.
   - There is no free lunch: the offset adaptation removes calls driven by the shared shift, but it does not add calls.
4. **In RPE1-source pairs, certification precision is poor for both methods.**
   - At 1 PC, m = 50: pilot-only 0.28–0.60, adapted 0.15–0.37.
   - The specific signal a model learns in RPE1 does not transfer, which is mechanism M2.
   - Here the adapted ball's precision is lower, not higher, so a "safety" claim is not supported in those pairs.
5. **Coverage holds throughout** (mean 0.897–0.926; minimum 0.88 for pilot-only, 0.89 for adapted).

## Implication for the paper

Report this as a limitation and a diagnostic:
- valid sets make visible that current models cannot certify perturbation-specific effects in a new cell line;
- tighter whole-transcriptome balls do not translate into more decisions.

Do **not** claim that adaptation improves decision-making.

## Audit (2026-10-02, `src/audit_specific.py`, 10 seeds, m = 50)

1. **No code bug.** A positive control certifies 83–100 % of perturbations: the predictor is the true target
   response plus Gaussian noise, sd 0.3–1.0 per gene. The readout works when predictions are informative.
2. **The negative result is real and not caused by transfer.** Ridge trained and tested within one line (no transfer)
   certifies 0–0.6 %. Its median predicted specific effect is about 1/6 of the set radius (K = 1: 1.6 vs 9.3 for
   Jurkat in-source; 2.7 vs 12.1 for HepG2 → Jurkat).
3. **Metric correction.** "Truth = top 25 % of true specific effects" does not match the certification question,
   which asks whether a perturbation differs from the typical one; almost every perturbation does.
   - The oracle control certifies almost everything at precision 0.25, exactly the base rate.
   - So precision must be read as *enrichment of the strongest quartile over the 0.25 base rate*, not as
     "1 − false certifications".
   - The earlier "false certifications per 1,000" and "adapted precision lower, so not safer" statements are
     **withdrawn**.
   - Correct reading: ridge certifications are enriched for strong responders in most pairs (precision 0.6–0.9
     vs 0.25), but not in RPE1-source pairs (0.15–0.37).
