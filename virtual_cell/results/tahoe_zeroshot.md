# Tahoe-100M: zero- and few-pilot calibration by pooling other lines (exploratory, 2026-10-02)

`pod/tahoe_zeroshot.py /root/tahoe --sources 12 --seeds 5` (12 random source lines x 5 seeds; ridge lambda=10 on
fingerprint-SVD features; 44 target lines with enough units). Per-line coverage is averaged over sources and seeds.
The pod was deleted after the run; the per-row JSON was not kept, this table is the script's printed summary.

| arm | mean coverage | per-line 5th pct | per-line min | lines < 0.85 | radius / pil50 (median) |
|---|---|---|---|---|---|
| src (source-calibrated) | 0.902 | 0.770 | 0.448 | 9/44 | 1.112 |
| pool (zero pilots, raw pooled scores) | 0.878 | 0.690 | 0.249 | 12/44 | 1.024 |
| poolfree (zero pilots, scale from control-cell counts) | 0.899 | 0.768 | 0.712 | 7/44 | 1.005 |
| poolcell (zero pilots, scale from treated-cell counts; upper bound) | 0.900 | 0.791 | 0.716 | 7/44 | 1.006 |
| pil10 (10 pilots, split CP) | 0.908 | 0.873 | 0.844 | 2/44 | 1.032 |
| pil50 (50 pilots, split CP) | 0.902 | 0.880 | 0.866 | 0/44 | 1.000 |
| hyb10 (10 pilots for scale + pooled normalised quantile) | 0.899 | 0.860 | 0.854 | 0/44 | 1.003 |

Reading: zero-pilot pooling is not reliable per line even with free covariates; ten pilots already give a radius within
about 3% of fifty, and borrowing other lines' score shape mainly tightens per-line coverage at m = 10.
