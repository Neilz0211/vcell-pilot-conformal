# Four-line transfer matrix (K562, RPE1, HepG2, Jurkat) + controls — 2026-10-02

## Sources
- K562/RPE1: Replogle 2022 pseudobulk. HepG2/Jurkat: Nadig 2025 raw counts → `pod/pseudobulk.py`.
- **Pipeline equivalence checked.** K562-essential raw single cell through our pipeline vs Replogle's own pseudobulk:
  Δ² = 0.000, per-target correlation 0.979. So mixing the two pipelines adds no offset.
- 20 seeds per pair; ridge10 on GO z; α = 0.1.
- Combined from `offset_pairs_metrics.json`, `nadig_pairs_metrics.json` and `matrix_pairs_metrics.json`.
- Columns:
  - **ρ** = Δ² / residual energy in the target line.
  - **"exact gain"** = radius of the finite-sample-valid adapted ball vs pilot-only at m = 50.
  - **"Prop C pred"** = √((1 + 1/m)/(1 + ρ)) · (m + 1)/m − 1.

| Pair | Δ² | ρ | ridge src cov | mean src cov | ridge + oracle offset | exact gain | Prop C pred | exact cov |
|---|---|---|---|---|---|---|---|---|
| RPE1 → Jurkat | 0.242 | 0.240 | 0.796 | 0.876 | 0.867 | −5.6 % | −7.5 % | 0.896 |
| Jurkat → RPE1 | 0.242 | 0.249 | 0.791 | 0.821 | 0.903 | −11.4 % | −7.8 % | 0.908 |
| K562 → RPE1 | 0.234 | 0.241 | 0.789 | 0.851 | 0.899 | −12.3 % | −7.5 % | 0.917 |
| K562-gwps → RPE1 | 0.216 | 0.216 | 0.811 | 0.847 | 0.913 | −11.6 % | −6.6 % | 0.914 |
| RPE1 → HepG2 | 0.133 | 0.141 | 0.849 | 0.874 | 0.863 | 0.0 % | −3.5 % | 0.901 |
| HepG2 → RPE1 | 0.133 | 0.141 | 0.860 | 0.894 | 0.916 | −6.8 % | −3.6 % | 0.913 |
| HepG2 → Jurkat | 0.083 | 0.085 | 0.885 | 0.923 | 0.902 | −0.8 % | −1.1 % | 0.903 |
| Jurkat → HepG2 | 0.083 | 0.085 | 0.846 | 0.858 | 0.872 | −2.7 % | −1.1 % | 0.906 |
| K562 → HepG2 | 0.076 | 0.082 | 0.842 | 0.868 | 0.871 | −2.7 % | −1.0 % | 0.918 |
| HepG2 → K562 | 0.076 | 0.082 | 0.883 | 0.905 | 0.892 | +0.9 % | −1.0 % | 0.901 |
| K562 → Jurkat | 0.044 | 0.048 | 0.892 | 0.914 | 0.905 | −0.5 % | +0.6 % | 0.914 |
| Jurkat → K562 | 0.044 | 0.049 | 0.870 | 0.871 | 0.879 | +0.6 % | +0.6 % | 0.899 |
| *same line:* K562-ess ↔ K562-gwps (two screens) | 0.011 | 0.013 | 0.89–0.91 | 0.90–0.91 | 0.90 | +1.7 / +3.7 % | +2.4 % | 0.90–0.92 |
| *same line, same pipeline:* HepG2 halves | 0.006 | 0.007 | 0.90 | 0.90 | 0.90 | +2.9 / +3.7 % | +2.6 % | 0.90 |
| *pipeline control:* K562 Replogle ↔ ours | 0.000 | 0.000 | 0.903 | 0.903 | 0.904 | +2.5 % | +3.0 % | 0.904 |

K562 → HepG2 and K562 → Jurkat each appear twice in the raw results (Replogle bulk as the source, and our
reprocessing of K562 as the source). The numbers are nearly identical, consistent with pipeline equivalence; the table
lists each once.

## Findings across 12 directed cross-line pairs (6 line pairs)
1. **F2 holds in every cross-line pair.** In 12/12 directed pairs, ridge (the better source model) covers less than
   predict-the-mean after transfer. Jurkat → K562 is nearly a tie (0.870 vs 0.871).
2. **Offset energy predicts the failure.** corr(Δ², ridge source coverage) = **−0.93** across the 12 pairs.
3. **Prop C predicts the value of adaptation.** corr(observed, predicted exact-ball gain) = **0.92** over all 18
   directed pairs including controls.
   - When ρ is large, the observed gain exceeds the energy-based prediction (−11 to −12 % vs −7.5 %); this is a
     quantile vs mean-energy effect, see `discussion_round2.md` §1.4.
   - The sign is right in the regimes that matter: costs +2 to +4 % when ρ ≈ 0, a wash around ρ ≈ 0.05–0.08, and
     gains when ρ ≥ 0.13.
   - **Practical rule: adapt only if ρ̂ > ~0.1**, i.e. m > m* with margin.
4. **RPE1 is the outlier line.** Its offset is 0.13–0.24 against every other line; the three cancer lines are
   0.04–0.08 apart from each other. That fits RPE1 being the non-transformed, TP53-wild-type line (a hypothesis, not
   tested).
5. **The oracle offset restores coverage when the offset dominates** (all RPE1-target pairs → 0.90–0.92).
   - Where ρ is small, the residual gap is a scale/tail shift; Jurkat → HepG2 is the clearest case (0.846 → 0.872).
   - Pilot-only and the exact adapted ball are valid in every pair (0.896–0.918).
6. **Wording.** Do not claim "low-dimensional" from 4 lines. Use "shared additive context offset whose size is
   predicted by ρ, plus a pair-specific scale/tail component".
