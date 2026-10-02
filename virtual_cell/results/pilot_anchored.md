# Pilot-anchored checks for hidden library shift (gwps, 30 seeds, α = 0.1) — 2026-10-01

Code: `src/pilot.py`; numbers in `results/pilot_metrics.json`; log in `results/pilot_run30.log`.

## Setup
- **Libraries.**
  - Calibration library: 1,500 perturbations.
  - Screening library: essential-only.
  - Pilots: m perturbations from the screening library, labelled. The rest of the screening library (600) is the
    test set.
- **Methods.**
  - `cal_w`: weighted CP with classifier-estimated weights.
  - `pilot`: split CP on the pilots alone. It is infinite (abstains) when m < 9.
  - `test`: a one-sided binomial test of `cal_w`'s coverage on the pilots (level 0.05). If it rejects, switch to
    `pilot`.
- **Bug fixed in this round.** `draw()` returns essential perturbations first, so the pilots were biased. They are
  now shuffled.

## Results

| Scenario | m | `cal_w` coverage | `pilot` coverage (radius) | `test` coverage | `test` reject rate |
|---|---|---|---|---|---|
| natural (0.2 → 1.0) | 10 | 0.875 | 0.917 (2.72) | 0.880 | 0.03 |
| natural | 20 | 0.868 | 0.901 (2.40) | 0.901 | 0.17 |
| natural | 50 | 0.890 | 0.902 (2.29) | 0.903 | 0.13 |
| disjoint (0 → 1) | 10 | 0.750 | 0.910 (2.69) | 0.872 | 0.40 |
| disjoint | 20 | 0.742 | 0.901 (2.41) | 0.893 | 0.53 |
| disjoint | 50 | **0.704** | 0.902 (2.29) | **0.903** | **0.77** |
| no shift (0.2 → 0.2) | 10 | 0.894 | 0.917 (1.89) | 0.898 | 0.03 |
| no shift | 20 | 0.911 | 0.914 (1.58) | 0.914 | 0.03 |
| no shift | 50 | 0.888 | 0.896 (1.41) | 0.898 | 0.03 |

Oracle weights (library design known): ~0.90 on the natural case and the no-shift control. In the disjoint case
they abstain on every test point (support violation).

## Reading
- **Feature-estimated weights fail silently under hidden shift.** In the disjoint case `cal_w` reaches 0.70–0.75,
  even though estimated ESS looks healthy (day 2). This is the core "false assurance" result.
- **About 20 pilots fix it.**
  - Pilot-only split CP is valid from m ≥ 9.
  - test-then-switch reaches nominal coverage at m ≥ 20 under the disjoint shift.
  - Its false-alarm rate is 3 % on the no-shift control.
- **Power is still limited.** Rejection is 0.53 at m = 20 and 0.77 at m = 50. Coverage is nevertheless restored on
  average, because the seeds that are not rejected are the ones where `cal_w` happened to cover better.
- **Caveat: this is a post-selection mixture.** test-then-switch has no formal coverage guarantee: it can lose up to
  roughly α + the test's type-II error. It needs a proper formal treatment, for example always using pilot CP, or a
  combined certificate.
- **The price of pilots is radius.** The pilot radius at m = 20 is 2.4, against `cal_w`'s ~2.2 when it is valid.
  The pilot-only set is wider, but it is honest.
