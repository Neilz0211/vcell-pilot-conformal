# E1: pilot-anchored conformal — src / pilot-only / P2 ratio-CP / P3 transfer-CP (30 seeds, α = 0.1) — 2026-10-01

Code: `src/pilot2.py`; numbers in `results/pilot2_metrics.json`. Cells give coverage / mean radius.

| Scenario | Method (no pilots → src) | m = 10 | m = 20 | m = 50 | m = 100 |
|---|---|---|---|---|---|
| K562 → RPE1 | src 0.788 / 1.35 | | | | |
| | pilot | 0.909 / 1.83 | 0.911 / 1.71 | 0.910 / 1.65 | 0.904 / 1.61 |
| | **P2** | 0.926 / 1.81 | **0.915 / 1.64** | **0.908 / 1.59** | 0.910 / 1.56 |
| | P3 γ = 20 | 0.863 / 1.51 | 0.881 / 1.56 | 0.891 / 1.57 | 0.898 / 1.59 |
| RPE1 → K562 | src 0.797 / 1.28 | | | | |
| | pilot | 0.890 / 1.72 | 0.889 / 1.61 | 0.891 / 1.54 | 0.896 / 1.53 |
| | **P2** | 0.903 / 1.69 | **0.890 / 1.52** | **0.890 / 1.46** | 0.895 / 1.46 |
| | P3 γ = 20 | 0.869 / 1.47 | 0.872 / 1.47 | 0.875 / 1.48 | 0.885 / 1.50 |
| disjoint library | src 0.737 / 1.64 | | | | |
| | pilot | 0.922 / 2.81 | 0.913 / 2.49 | 0.910 / 2.34 | 0.908 / 2.27 |
| | P2 | 0.931 / 2.90 | 0.906 / 2.44 | 0.905 / 2.33 | 0.907 / 2.28 |
| | P3 γ = 20 | **0.767** / 1.77 | **0.805** / 1.87 | 0.852 / 2.00 | 0.873 / 2.08 |
| natural → essential | src 0.876 / 2.18 | | | | |
| | pilot | 0.927 / 2.95 | 0.919 / 2.62 | 0.911 / 2.42 | 0.900 / 2.27 |
| | P2 | 0.923 / 2.86 | 0.925 / 2.70 | 0.912 / 2.46 | 0.904 / 2.32 |
| | P3 γ = 20 | 0.895 / 2.30 | 0.898 / 2.31 | 0.902 / 2.32 | 0.901 / 2.28 |
| no shift | src 0.896 / 1.48 | | | | |
| | pilot / P2 / P3 γ = 20 at m = 20 | | 0.915 / 1.66 · 0.906 / 1.57 · 0.909 / 1.52 | | |
| reverse | src 0.907 / 1.45 | | | | |
| | P2 / P3 γ = 20 at m = 20 | | 0.901 / 1.56 · 0.912 / 1.47 | | |

## Against the go / kill criteria in `discussion_round1.md`
- **P2 is valid everywhere** (0.89–0.93), as the theory says.
- **P2's efficiency gain is real but small.** Against the go bar (radius ≤ 90 % of pilot-only):
  - context shift, m = 20–50: radius 3.5–5.6 % below pilot-only — gain present, short of the bar;
  - library shift: no gain.
- **P2 is not ≥ 10 % wider than pilot-only anywhere**, so the kill criterion does not trigger either.
- **Reading: the z-dependent shape only partly transfers.** The kNN difficulty map learned in the source line is a
  weak predictor in the target.
- **P3 does not fail gracefully.**
  - Under hidden shift (disjoint library) it under-covers badly even with γ = 20: 0.77 at m = 10, 0.81 at m = 20.
  - In the context-shift cases it stays at 0.86–0.89.
  - It helps only where shift is mild or absent (no shift, reverse, natural → essential): about nominal coverage
    with a radius close to `src`.
- **The KS distance estimated from the pilots barely predicts the real coverage gap.** The correlation with the gap
  is 0.17–0.26 (disjoint) and about 0 elsewhere, so an adaptive γ that picks itself from the pilots is unlikely to
  work at m ≤ 50.

## What this means for the paper
- Under hidden or context shift, nothing tested here clearly beats pilot-only conformal at a given m. P2 shaves
  4–6 % in the context case.
- This supports a minimax statement:
  - when the target score law is unconstrained outside the source, the m-pilot order statistic is essentially the
    best one can do;
  - the efficiency floor is set by m (radius excess ~ √(α(1−α)/m) / density).
- So "how many pilots" is the actionable quantity. Clever transfer cannot replace pilots when the shift is hidden.
- P3-style transfer is safe only when the shift is visible or mild. Pilots cannot certify that at m ≤ 50: KS power
  is too low.
