# Idea 1: adapt-then-calibrate with the same pilots (30 seeds, α = 0.1) — 2026-10-01

Code: `src/adapt.py`; numbers in `results/adapt_metrics.json`.

## Setup
- **Source model.** Ridge on GO z, fit in line A.
- **Adaptation.** The m pilots in line B fit a per-gene line offset b = mean pilot residual: "offset" (affine adds
  nothing).
- **Calibration.** Leave-one-out on the same pilots, using the ⌈(m+1)(1−α)⌉-th LOO score. The test set is fixed
  across m.

Cells give coverage / radius.

| Direction | m | pilot-only, unadapted ridge | **ridge + offset (LOO)** | pilot mean (LOO) | Radius change vs pilot-only |
|---|---|---|---|---|---|
| K562 → RPE1 | 10 | 0.909 / 1.83 | 0.916 / 1.69 | 0.918 / 1.77 | −8 % |
| | 20 | 0.911 / 1.71 | 0.911 / **1.50** | 0.909 / 1.56 | −12 % |
| | 50 | 0.910 / 1.65 | 0.908 / **1.42** | 0.907 / 1.47 | −14 % |
| | 100 | 0.904 / 1.61 | 0.899 / **1.36** | 0.902 / 1.43 | −16 % |
| RPE1 → K562 | 10 | 0.890 / 1.72 | 0.888 / 1.67 | 0.894 / 1.73 | −3 % |
| | 20 | 0.889 / 1.61 | 0.887 / **1.52** | 0.884 / 1.58 | −6 % |
| | 50 | 0.891 / 1.54 | 0.887 / **1.42** | 0.893 / 1.50 | −8 % |
| | 100 | 0.896 / 1.53 | 0.892 / **1.41** | 0.895 / 1.48 | −8 % |

## Reading
1. **One set of pilots, used twice, gives sets 8–16 % tighter at about nominal coverage.**
   - Coverage is 0.887–0.916, with pilot-only exact CP at 0.889–0.911 on the same splits.
   - Leave-one-out with a mean-offset estimator is very stable, so jackknife behaves like exact CP here.
   - The formal guarantee to state is jackknife+ (≥ 1 − 2α), or exact full conformal: the offset estimator is a
     mean, so full CP is computable in closed form. **To do.**
2. **With 50 pilots, the adapted model beats the full-screen oracle of the unadapted model.**
   - Unadapted ridge with oracle full calibration in B has radius 1.60 (K562 → RPE1) and 1.54 (RPE1 → K562), from
     `context_shift.md`.
   - Ridge + offset with 50 pilots reaches 1.42 in both directions.
   - So the cell-line gap is largely a low-dimensional (per-gene offset) shift. A few dozen pilots both detect and
     remove it.
3. **The source model's z-signal partly survives once the offset is removed.**
   - Ridge + offset beats the pilot mean by 3–5 % (1.42 vs 1.47 at m = 50; 1.36 vs 1.43 at m = 100).
   - Unadapted, ridge is no better than the mean in the new line (`ladder.md`).
   - So "the source gain does not transfer" is mostly an offset artefact. This refines the headline: the gain
     transfers once a context offset is learned from pilots.

## Next
- Exact or guaranteed version: full conformal for the mean-offset estimator, or jackknife+.
- Richer low-rank adaptation:
  - offset plus a few residual PCs from the pilots;
  - a shrinkage weight between source and pilot means chosen by LOO.
- The same idea under library shift (offset within the essential library).
- A second context pair.
