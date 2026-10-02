# E-A: is the offset a cell-line property or a screen (batch) property? (20 seeds, α = 0.1) — 2026-10-01

Code: `src/offset_pairs.py`; numbers in `results/offset_pairs_metrics.json`.
- **Δ²** = per-gene mean-shift energy between the two screens.
- **"exact"** = the (m+1)/m × LOO-radius ball for the offset-adapted model. Coverage ≥ 1 − α holds in finite samples
  (`discussion_round2.md` §1).

| Pair | Cross-screen corr | Δ² | ridge src cov | src radius + oracle offset | ridge oracle radius in B | pilot-only, m = 50 | **exact adapted, m = 50 (cov)** |
|---|---|---|---|---|---|---|---|
| K562-ess → K562-gwps (same line) | 0.46 | **0.011** | 0.893 | 0.897 | 1.42 | 1.41 | 1.44 (0.903) |
| K562-gwps → K562-ess (same line) | 0.46 | **0.011** | 0.906 | 0.905 | 1.40 | 1.40 | 1.45 (0.915) |
| K562-gwps → RPE1 | 0.25 | **0.216** | **0.811** | 0.913 | 1.62 | 1.64 | **1.45 (0.914)** |
| K562-ess → RPE1 | 0.25 | **0.234** | **0.789** | 0.899 | 1.61 | 1.67 | **1.46 (0.917)** |

## Reading
- **Falsifier F3 does not trigger.**
  - The same cell line measured in two different screens has almost no offset (Δ² 0.011, about 20× smaller), and
    source calibration stays valid (0.89–0.91).
  - Across cell lines the offset is 0.22–0.23 from either source screen, and source calibration fails (0.79–0.81).
  - **So the offset is a property of the cell context, not of the screen batch.**
- **The oracle offset explains the whole cross-line coverage gap for ridge** (0.79–0.81 → 0.90–0.91 with the source
  radius).
- **With a finite-sample guarantee, 50 pilots give radius 1.45–1.46.** That is 12–13 % below pilot-only and 10 %
  below the full-screen oracle of the unadapted model (1.61–1.62). At m = 20: 1.60–1.62 vs pilot 1.74 (−7 to −8 %).
- **Second source screen replicates the result.** gwps → RPE1 behaves the same as ess → RPE1.
- **Adaptation costs a little when there is no offset.** In the same-line pairs the exact adapted ball is 2–4 %
  wider than pilot-only, from the (m+1)/m factor plus offset-estimation noise.
- **Practical rule.** Estimate ρ̂ from the pilots and adapt only when the offset is non-trivial. This is the
  self-estimated stopping / adapt rule in E-E; the rule must be decided before seeing the test data, or both arms must
  be reported.
