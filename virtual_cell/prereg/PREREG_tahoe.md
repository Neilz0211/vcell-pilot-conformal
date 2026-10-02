# Pre-registration: calibration transfer on Tahoe-100M (chemical perturbations)

Written 2026-10-02, before any Tahoe-100M response data entered an analysis. Committed to git before the
extraction run; the commit hash and timestamp are the registration record.

Disclosure of what was seen before registering: the metadata tables (drug, cell line, sample) and, to learn the
schema, two DE shards (`train-00000`, `train-00500`, one cell line each). From those I looked only at the column
names, the block layout, the fraction of NaN log2FoldChange values (~0.5) and the fraction of rows with
baseMean >= 10 (~0.2). No response vectors were compared across lines and no coverage or radius was computed.

## Data

Tahoe-100M (`tahoebio/Tahoe-100M`, CC0), `metadata/pseudobulk_differential_expression`: DESeq2 per
plate x drug x dose x cell line against that plate's DMSO wells. 50 cancer cell lines pooled in every well
(cell village), 379 drugs, 3 doses, 14 plates. Extraction: `pod/tahoe_extract.py`.

## Protocol

- **Response**: log2FoldChange (not the Wald stat, which scales with cell numbers per line). NaN = missing.
- **Gene pool (symmetric, once)**: genes with median baseMean >= 10 over all line x unit blocks and finite LFC in
  >= 95% of blocks; top 3,000 by overall LFC variance. Per pair: a gene is kept only if finite in every kept unit
  of both lines; units with any remaining NaN in the pool genes of either line are dropped.
- **Unit**: drug x dose. Unit filter (metadata only): n_cells_trt >= 30 in both lines of the pair; kept fraction
  reported. No activity filter in the primary analysis.
- **Splits**: 40/30/30 train/calibration/test, grouped by drug (no drug in two parts). Pilots are drawn from the
  target's calibration part, as in the main paper.
- **Readout**: per pair and seed, top 1,000 genes by source-train variance; per-gene SD from the source-train split
  scales both lines (`sourcetrain`, leak-free). Pooled preprocessing is a secondary table.
- **Features**: RDKit Morgan fingerprint (radius 2, 2048 bits) from the drug SMILES, SVD to 64 dimensions over all
  drugs (unsupervised, mirrors the GO-SVD), plus log10 dose. Drugs without a parseable SMILES: one-hot of their
  target annotation, else dropped (count reported).
- **Models**: predict-the-mean, ridge (lambda in {1000, 100, 10, 1}), kNN on the features (k = 10), and
  "other-lines mean" (the unit's mean LFC over the 48 lines that are neither source nor target, in source scale).
- **Pairs**: all 50 x 49 = 2,450 directed pairs for ridge(lambda=10) and mean, 20 seeds, m in {20, 50, 100}.
  Ladder, paired comparison and the rho-hat rule on 100 pairs drawn with
  `np.random.default_rng(0).choice(2450, 100, replace=False)` over the sorted pair list, 10 seeds.
- **Uncertainty**: pooled statistics get a line-cluster bootstrap (resample cell lines, keep pairs whose two lines
  are both in the resample, 1,000 replicates). Per-target-line medians are reported alongside pooled values.
- alpha = 0.1 throughout.

## Predictions (pass/fail each, all reported)

- **P1 (law)**: Spearman(Delta^2, ridge source-calibrated target coverage) over the 2,450 pairs is negative with
  |rho_s| >= 0.5, and the line-cluster bootstrap 95% CI excludes 0.
- **P2 (quantitative)**: the model-free predicted coverage mean(s_cal^2 + Delta^2 <= q_A^2) (cross term ignored)
  has MAE <= 3 percentage points and correlation >= 0.8 with observed coverage over pairs.
- **P3 (inversion)**: ridge(lambda=10) covers less than predict-the-mean in >= 75% of pairs; along the ladder, the
  within-pair Spearman between source radius and transferred coverage has median >= 0.8 over the 100-pair subset
  (smaller radius, lower coverage); the other-lines-mean rung loses the most coverage in >= 60% of those pairs.
- **P4 (repair)**: mean coverage of the adapted ball at m = 50 lies in [0.88, 0.93]; adapted ridge is narrower than
  the adapted pilot mean in >= 60% of pairs.
- **P5 (rho-hat rule)**: on the 100-pair subset the rule picks the smaller-radius arm in >= 80% of pairs at m = 50
  and m = 100; mean coverage of the rule's choice >= 0.89; |bias(rho-hat)| <= 0.05.
- **P6 (Prop C)**: correlation between predicted and observed radius change at m = 50 >= 0.7, MAE <= 3 pp.

## Pre-specified diagnostics (no pass/fail)

- Delta^2 within each of the 14 plates versus pooled: if within-plate Delta^2 is much smaller than pooled, a
  plate x line interaction, not a shared context offset, carries the shift, and this is reported as such.
- Highest dose only (~ one unit per drug) as a robustness run for the correlated-dose caveat of Prop C.
- Secondary "source-active" panel: units whose source-line RMS LFC exceeds the source median.

A failed P1 or P3 is reported as a scope limitation of the paper's claims, not re-analysed until it passes.
