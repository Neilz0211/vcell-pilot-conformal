# Context shift: calibrate in one cell line, deploy in another (K562-essential ↔ RPE1, 30 seeds, α = 0.1) — 2026-10-01

Code: `src/context.py`; numbers in `results/context_metrics.json`.

## Setup
- **Targets and readout.** 1,978 shared target genes with GO annotation; 1,000 shared readout genes, standardised
  within each cell line.
- **Responses barely agree across lines.** The mean per-target correlation between a target's K562 and RPE1
  responses is 0.25.
- **Split.** 40 % of targets for training, 30 % for calibration, 30 % for testing. The pilots are m test targets
  labelled in the new cell line.

## Results

Cells give coverage, with the set radius in parentheses.

| Direction | Model | Source-calibrated | Pilot CP, m = 20 | Pilot CP, m = 50 | Oracle (full screen in the new line) |
|---|---|---|---|---|---|
| K562 → RPE1 | ridge(z) | **0.786** (1.35) | 0.911 (1.71) | 0.909 (1.65) | 0.905 (1.60) |
| K562 → RPE1 | mean | 0.850 (1.49) | 0.909 (1.76) | 0.909 (1.70) | 0.903 (1.65) |
| RPE1 → K562 | ridge(z) | **0.799** (1.28) | 0.889 (1.61) | 0.891 (1.54) | 0.898 (1.54) |
| RPE1 → K562 | mean | 0.861 (1.42) | 0.885 (1.61) | 0.891 (1.55) | 0.900 (1.55) |
| either | measured source response ("transfer") | — | 0.896 (1.89) | 0.902 (1.85) | 0.902 (1.84) |

**The binomial test switch has weak power at this shift.**
- Rejection rate for ridge: 0.37 at m = 20 and 0.67 at m = 50 (K562 → RPE1); 0.33 and 0.40 (RPE1 → K562).
- Coverage after switching is 0.85–0.88, still short of nominal.

## Reading
1. **Source-calibrated sets under-cover by 10–11 points in a new cell line**, and they fail silently.
2. **The model that looks better in the source line is the more overconfident one in the new line.**
   - Ridge's source radius is 9–10 % smaller than the mean's (1.35 vs 1.49).
   - In the new line ridge is only 0–3 % better (1.60 vs 1.65; 1.54 vs 1.55).
   - So its coverage drops further (0.79 vs 0.85): the source-line gain is context-specific and does not transfer.
   - This matches the "virtual cells do not beat the mean" literature, now with a calibration consequence.
3. **Copying the measured response from the other line is worse than predicting the mean** (radius 1.84 vs
   1.55–1.65). The cross-line correlation is low.
4. **20–50 pilots in the new line give near-oracle sets** (coverage 0.89–0.91, radius within 3–7 % of the oracle at
   m = 50).
   - The test-then-switch rule is dominated: always using the pilot CP is both valid and cheap here.
   - Recommendation: a few dozen pilot perturbations in each new context. Source calibration should not be trusted.
