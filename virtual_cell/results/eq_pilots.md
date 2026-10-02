# Equivalent pilot count (20 seeds, ridge10, α = 0.1) — 2026-10-02

Code: `src/eq_pilots.py`; numbers in `results/eq_pilots_metrics.json`.

**Question.** At the same nominal coverage, how many pilots does pilot-only calibration of the unadapted model need to
match the radius of the adapted ball (Prop A outer ball) with m_a pilots?

**Setup.**
- The evaluation set is fixed: 280 targets.
- "Full screen" = pilot-only calibration on every remaining target (about 300–400). This is the best the unadapted
  model can do.

| Pair | Full-screen radius | Adapted radius at m_a = 20 / 50 | Pilot-only needs, for m_a = 20 | Pilot-only needs, for m_a = 50 |
|---|---|---|---|---|
| K562 → RPE1 | 1.605 | 1.589 / 1.463 | never | never (−9 % below full screen) |
| Jurkat → RPE1 | 1.606 | 1.610 / 1.460 | 150 | never (−9 %) |
| RPE1 → K562 | 1.544 | 1.600 / 1.454 | 20 | never (−6 %) |
| RPE1 → Jurkat | 1.508 | 1.607 / 1.426 | 30 | never (−5 %) |
| HepG2 → RPE1 | 1.478 | 1.545 / 1.409 | 30 | never (−5 %) |
| Jurkat → HepG2 | 1.559 | 1.675 / 1.511 | 30 | never (−3 %) |
| K562 → HepG2 | 1.581 | 1.799 / 1.559 | 20 | never (−1.4 %) |
| K562 → Jurkat | 1.415 | 1.549 / 1.432 | 20 | 100 |
| HepG2 → Jurkat | 1.451 | 1.653 / 1.447 | 20 | 100 |
| RPE1 → HepG2 | 1.487 | 1.682 / 1.479 | 10 | 50 |
| HepG2 → K562 | 1.453 | 1.616 / 1.440 | 20 | 50 |
| Jurkat → K562 | 1.424 | 1.618 / 1.446 | 20 | 50 |

## Reading

1. **50 pilots: adaptation beats any number of pilots in 7 of 12 pairs.**
   - The adapted ball beats pilot-only calibration of the unadapted model with any number of pilots, including the
     whole target screen, by 1.4–9 %.
   - These are all pairs involving RPE1 plus Jurkat → HepG2 and K562 → HepG2.
   - In the other 5 pairs, pilot-only calibration needs 50–100 pilots to match.
2. **20 pilots: adaptation buys little.**
   - Pilot-only calibration usually matches with 20–30 pilots.
   - The exception is Jurkat → RPE1, which needs 150.
3. **10 pilots: adaptation is worse.**
   - The (m+1)/m factor and the order statistic k = m make the outer ball wide.
   - Pilot-only matches with 10 everywhere except Jurkat → RPE1, which needs 20.
4. **This is not a cost saving in dollars.**
   - The unadapted model cannot be calibrated below its own target-line error, whereas the offset removes a shared
     component that no amount of calibration data can remove.
   - What it is: a pilot-budget comparison at equal coverage.
   - Grid resolution limits m_eq (10, 20, 30, 50, 75, 100, 150, 200, 300).
