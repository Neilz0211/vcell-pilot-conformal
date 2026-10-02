# Paired budget comparison, leak-free preprocessing (20 seeds, ridge10, α = 0.1) — 2026-10-02

Code: `src/paired.py`; numbers in `results/paired_{pooled,source,sourcetrain}_metrics.json`.

## Design
- **Data split.** Each pair's ~2,000 shared targets are split 40 % train / 30 % unused / 30 % target pool. The pool is
  divided into a fixed 280-target test set and a disjoint calibration pool of 310–410 targets (asserted in the code).
- **Shared pilots.** All methods use the same pilots: the first m of the calibration pool.
- **Methods compared.**
  - **full**: the unadapted model calibrated on the whole calibration pool.
  - **pilot**: pilot-only split conformal.
  - **adapted**: Prop A outer ball, reusing all m pilots.
  - **split**: same budget, offset estimated from the first m/2 pilots and split conformal on the other m/2.
- **Order statistics.** m = 50 uses the 46th; 25 uses the 24th.
- **Preprocessing.**
  - The main run, *sourcetrain*, fits genes and per-gene scale on the source training split only; its rules were
    locked before it ran.
  - *pooled* (the original) and *source* (genes and scale from all source targets) were run earlier, before the rules
    were locked. They give the same picture.
- **What the intervals mean.** 95 % t-intervals over 20 seeds describe the sampling stability of splits and pilot
  draws, not generalisation to new cell lines.

## Main run (sourcetrain)

Adapted (m = 50) compared with full-pool calibration of the unadapted model, and with split 25 + 25.

| Pair | Coverage: full / adapted / split / pilot | Adapted vs full: mean [95 % CI], wins | Adapted vs split: mean [95 % CI], wins |
|---|---|---|---|
| K562 → RPE1 | 0.909 / 0.920 / 0.923 / 0.920 | −7.9 % [−16.5, +0.7], 15/20 | −4.3 % [−10.9, +2.4], 11/20 |
| K562 → HepG2 | 0.915 / 0.926 / 0.920 / 0.918 | −2.2 % [−7.2, +2.8], 14/20 | +1.4 % [−7.6, +10.4], 9/20 |
| K562 → Jurkat | 0.910 / 0.917 / 0.921 / 0.909 | +0.4 % [−4.1, +4.8], 9/20 | −3.7 % [−11.8, +4.3], 12/20 |
| RPE1 → K562 | 0.903 / 0.897 / 0.901 / 0.889 | **−18.6 % [−22.4, −14.9], 20/20** | −0.8 % [−5.9, +4.2], 12/20 |
| RPE1 → HepG2 | 0.907 / 0.901 / 0.922 / 0.896 | −0.8 % [−7.0, +5.4], 10/20 | −4.5 % [−10.0, +1.1], 12/20 |
| RPE1 → Jurkat | 0.900 / 0.894 / 0.914 / 0.891 | **−16.4 % [−20.1, −12.7], 19/20** | **−7.3 % [−13.5, −1.1], 14/20** |
| HepG2 → K562 | 0.906 / 0.897 / 0.913 / 0.891 | **−5.9 % [−9.9, −1.8], 14/20** | −4.8 % [−10.5, +0.9], 13/20 |
| HepG2 → RPE1 | 0.911 / 0.916 / 0.929 / 0.911 | **−6.1 % [−10.9, −1.4], 15/20** | **−6.3 % [−12.6, −0.1], 13/20** |
| HepG2 → Jurkat | 0.903 / 0.913 / 0.919 / 0.908 | −1.2 % [−5.3, +2.9], 12/20 | −2.7 % [−8.9, +3.4], 13/20 |
| Jurkat → K562 | 0.899 / 0.900 / 0.917 / 0.893 | +0.9 % [−2.7, +4.5], 9/20 | **−7.9 % [−15.3, −0.5], 17/20** |
| Jurkat → RPE1 | 0.901 / 0.915 / 0.921 / 0.908 | **−9.5 % [−14.6, −4.4], 15/20** | −4.5 % [−11.1, +2.2], 13/20 |
| Jurkat → HepG2 | 0.906 / 0.900 / 0.931 / 0.891 | −4.1 % [−9.5, +1.3], 12/20 | **−9.9 % [−18.0, −1.8], 13/20** |

## Reading
1. **Removing all target information from preprocessing changes nothing material.**
   - The adapted ball with 50 pilots is narrower than full-pool calibration of the unadapted model, with the CI below
     0, in 5/12 pairs. These are RPE1 → K562, RPE1 → Jurkat, HepG2 → RPE1, Jurkat → RPE1 (four RPE1 pairs) and
     HepG2 → K562, by 6–19 %. The other 7 pairs are within sampling noise.
   - The same 5/12 holds under the pooled and source preprocessing.
   - This replaces the earlier 7-of-12 wording.
2. **The gain from reusing pilots (Prop A vs split 25 + 25) is modest.**
   - The adapted ball is narrower on average in 11/12 pairs (by 0.8–9.9 %), with the CI below 0 in 4/12.
   - The split method is more conservative (coverage 0.90–0.93 vs adapted 0.89–0.93).
   - The difference combines two things: a better offset estimate (50 vs 25 pilots) and more calibration scores
     (order statistic 46 of 50 vs 24 of 25).
   - At m = 20 the two are about tied.
3. **Coverage of all methods is near nominal** (0.889–0.931).
