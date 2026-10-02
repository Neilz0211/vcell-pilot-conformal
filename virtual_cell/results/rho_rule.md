# Pre-registered ρ̂ adapt rule (20 seeds, ridge10, α = 0.1) — 2026-10-02

## Setup
- Code: `src/rho_rule.py`. Numbers in `results/rho_rule_metrics.json`; log in `results/rho_rule_run20.log`.
- **Rule** (written before running): adapt iff ρ̂ > (3m+1)/(m(m−1)). This is the certified break-even from
  `discussion_round3.md` §1.3.
- ρ̂ is estimated from the m pilots only.
- 12 directed cross-line pairs plus 2 same-line controls (K562-ess ↔ K562-gwps).

## Pre-registered criteria vs outcome

| Criterion | Target | Outcome |
|---|---|---|
| Rule picks the smaller-radius arm (by mean radius over seeds) | ≥ 12/14 pairs | **11/14 at m = 20, 11/14 at m = 50, 13/14 at m = 100**: met only at m = 100 (missed by one pair at m = 20 and 50) |
| Mean rule-arm coverage | ≥ 0.89 | **0.912 / 0.904 / 0.902**: met |
| Kill: \|ρ̂ − ρ\| > 0.2 at m = 20 | — | **Not triggered.** Max \|bias\| 0.029; per-seed sd 0.015–0.14 at m = 20 |

## Reading
- **The misses are near-ties at the break-even.** Rule-arm radius vs the better fixed arm:
  - mean regret is +0.06 % (m = 20), +0.09 % (m = 50) and +0.01 % (m = 100);
  - worst single pair is +1.0 %.
  - Example misses: K562 → Jurkat at m = 50 (1.434 vs 1.432) and RPE1 → K562 at m = 20 (1.588 vs 1.600).
- **ρ̂ is essentially unbiased** (|bias| ≤ 0.03) and identifies the regime correctly:
  - the same-line controls never adapt (0 %);
  - RPE1 pairs at m = 50 always adapt (100 %);
  - pairs near the break-even (ρ ≈ 0.08) switch from mostly not adapting at m = 20 to mostly adapting at m = 50, as the
    threshold 3/m predicts.
- **Honest summary for the paper.** The pre-registered pair-count criterion was met at m = 100 and missed by one pair
  at m = 20 and 50. In all cases the rule is within 1 % of the better fixed arm, with coverage ≥ 0.90 on average.
- **Caveat.** Selecting between the two sets with the same pilots is not covered by Prop A, so coverage is measured,
  not guaranteed. Coverage is fine empirically (0.89–0.94 per pair).
