# Second context pair: Nadig 2025 HepG2 ↔ Jurkat (20 seeds, α = 0.1) — 2026-10-02

## Pipeline
- Raw single cell → `pod/pseudobulk.py`: median-library normalisation, per-gem-group z-score against non-targeting
  controls, mean per perturbation.
- Run on a RunPod CPU pod; analysis by `src/offset_pairs.py` on the pod.
- Numbers in `results/nadig_pairs_metrics.json`.

## Results

Ridge10 on GO z; "exact" = the (m+1)/m × LOO-radius adapted ball. The last column is the exact ball's radius change
vs pilot-only.

| Pair | Pipeline | Δ² | ridge src cov | + oracle offset | mean src cov | qA → qB (oracle) | pilot / exact radius, m = 50 (exact cov) | Change |
|---|---|---|---|---|---|---|---|---|
| HepG2 even → odd (same line, same pipeline) | same | **0.006** | 0.895 | 0.896 | 0.899 | 1.35 → 1.37 | 1.38 / 1.42 (0.905) | +3 % |
| HepG2 → Jurkat | same | **0.083** | 0.885 | 0.902 | 0.923 | 1.39 → 1.44 | 1.45 / 1.44 (0.903) | −1 % |
| Jurkat → HepG2 | same | **0.083** | **0.846** | 0.872 | 0.858 | 1.35 → **1.59** | 1.58 / 1.54 (0.906) | −3 % |
| K562-ess → HepG2 | different | 0.076 | 0.841 | 0.871 | 0.866 | 1.34 → 1.55 | 1.63 / 1.58 (0.918) | −3 % |
| K562-ess → Jurkat | different | 0.044 | 0.891 | 0.904 | 0.912 | 1.36 → 1.39 | 1.44 / 1.43 (0.914) | −1 % |
| RPE1 → HepG2 | different | 0.133 | 0.849 | 0.863 | 0.874 | 1.31 → 1.51 | 1.50 / 1.50 (0.901) | 0 % |
| (reference) K562-ess → RPE1 | Replogle | 0.234 | 0.789 | 0.899 | 0.851 | 1.35 → 1.61 | 1.67 / 1.46 (0.917) | −13 % |

## Reading (honest)
1. **The same-pipeline same-line control is clean** (Δ² = 0.006, coverage 0.90). The pseudobulk pipeline creates no
   offset of its own.
2. **Source calibration fails again in a new pair, asymmetrically.**
   - Jurkat → HepG2: 0.846 for ridge.
   - HepG2 → Jurkat: mild, 0.885.
   - Into HepG2 from any source: 0.84–0.85.
3. **F2 holds directionally in every cross-line pair: ridge covers less than the mean.**
   - HepG2 → Jurkat 0.885 vs 0.923; Jurkat → HepG2 0.846 vs 0.858.
   - ess → HepG2 0.841 vs 0.866; ess → Jurkat 0.891 vs 0.912; RPE1 → HepG2 0.849 vs 0.874.
4. **F3 is pair-dependent.**
   - The HepG2/Jurkat offset is ~3× smaller than K562/RPE1 (0.08 vs 0.23).
   - Removing it recovers only a fraction of the gap (Jurkat → HepG2: 0.846 → 0.872).
   - The rest is a **scale/tail shift**: the target-line 90 % radius is larger (1.35 → 1.59), because HepG2 responses
     are heavier-tailed.
   - So "the gap is an offset" is true for K562 ↔ RPE1 and only partly true elsewhere. Write it as **gap = shared
     offset + scale / tail shift**, with the share varying by pair.
5. **Prop C predicts the adaptation value across pairs.**
   - Here ρ ≈ 0.085 (m* ≈ 12). The predicted RMS ratio at m = 50 is √(1.02 / 1.085) = 0.97. With the (m+1)/m
     inflation the exact ball comes out at about −1 %; observed −1 % to −3 %.
   - For K562 → RPE1 (ρ ≈ 0.24) the prediction is −11 %; observed −13 %.
   - Same-line (ρ ≈ 0.006): predicted cost; observed +3 %.
   - This supports the "estimate ρ from the pilots, then decide whether to adapt" story.
6. **RPE1 looks like the outlier line.** Its offset is the largest (0.13–0.23) in every pair. That fits biology: RPE1 is
   a non-cancer, TP53-wild-type line; the others are cancer lines. This is not tested.
7. **Pilot-only calibration and the exact adapted ball are valid in every pair** (0.90–0.92).

## Implications for the paper
- Headline mechanism: "source-calibrated sets fail across cell lines; the failure decomposes into a shared additive
  offset (dominant for K562 ↔ RPE1) and a scale/tail shift (dominant for HepG2 ↔ Jurkat)".
- In both cases, the m target pilots certify validity.
- The offset adaptation's value is predicted in advance by ρ (Prop C / m*).
- Consider a **scale-adaptive** variant: offset plus a per-pair scale from the pilots via LOO, which is still symmetric
  and so valid by Prop B. This would capture the HepG2 tail shift.
