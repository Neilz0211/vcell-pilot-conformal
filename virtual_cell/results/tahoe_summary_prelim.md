# Tahoe-100M calibration transfer: pre-registered predictions

Pairs analysed: 2162 of 2450 (skipped: 288); units per pair median 1107, drugs 377

## P1 law: Spearman(Delta^2, ridge source-calibrated coverage) < 0, |rho| >= 0.5, cluster CI excludes 0
Spearman -0.827, line-cluster bootstrap 95% CI [-0.876, -0.753]; Pearson -0.819. Delta^2 range 0.024-0.607 (median 0.050); ridge coverage 0.000-1.000 (median 0.949); with the oracle offset 0.953. **PASS**

## P2 quantitative: model-free predicted coverage, MAE <= 3 pp and corr >= 0.8
MAE 16.97 pp, corr 0.553, mean signed error +6.70 pp. **FAIL**

## P3 inversion: ridge < mean in >= 75% of pairs; ladder Spearman median >= 0.8; other-lines mean worst in >= 60%
ridge10 covers less than predict-the-mean in 85.4% of 2162 pairs (median gap -0.2 pp). Ladder (100-pair subset, 89 analysed): within-pair Spearman(source radius, transferred coverage) median 0.14 (IQR -0.14-0.54); other-lines mean has the lowest transferred coverage in 30%. **FAIL**

Mean over the subset, per model: source radius qA / source-calibrated target coverage / test coverage in the source line:
- mean: qA 1.270, covB_src 0.811, covA_test 0.897, qB_oracle 1.236
- ridge1000: qA 1.268, covB_src 0.810, covA_test 0.897, qB_oracle 1.236
- ridge100: qA 1.265, covB_src 0.807, covA_test 0.897, qB_oracle 1.238
- ridge10: qA 1.264, covB_src 0.805, covA_test 0.897, qB_oracle 1.242
- ridge1: qA 1.300, covB_src 0.805, covA_test 0.899, qB_oracle 1.286
- knn10: qA 1.317, covB_src 0.810, covA_test 0.898, qB_oracle 1.295
- otherlines: qA 1.175, covB_src 0.800, covA_test 0.898, qB_oracle 1.126

## P4 repair: adapted-ball coverage at m=50 in [0.88, 0.93]; adapted ridge narrower than adapted pilot mean in >= 60% of pairs
adapted coverage mean 0.913 (pair range 0.878-0.961); pilot-only 0.900. Adapted ridge narrower than adapted pilot mean in 5.8% (median +0.5%). Adapted vs pilot-only radius: median +0.4%. **FAIL**

- m=20: pilot cov 0.901, adapted cov 0.929, split cov 0.909; adapted/pilot radius median +4.3%, adapted/split -3.4%
- m=50: pilot cov 0.900, adapted cov 0.913, split cov 0.923; adapted/pilot radius median +0.4%, adapted/split -6.1%
- m=100: pilot cov 0.900, adapted cov 0.906, split cov 0.901; adapted/pilot radius median -0.9%, adapted/split -0.4%

## P5 rho-hat rule (100-pair subset, ridge10): better arm >= 80% at m=50,100; rule coverage >= 0.89; |bias| <= 0.05
- m=20: picks the smaller-radius arm in 100% of pairs; mean rule coverage 0.899; mean bias of rho-hat +0.001
- m=50: picks the smaller-radius arm in 78% of pairs; mean rule coverage 0.901; mean bias of rho-hat -0.000
- m=100: picks the smaller-radius arm in 89% of pairs; mean rule coverage 0.904; mean bias of rho-hat -0.000
**FAIL**

## P6 Prop C: corr(predicted, observed radius change at m=50) >= 0.7, MAE <= 3 pp
corr 0.918, MAE 0.94 pp, slope 0.69; rho range 0.028-0.432. **PASS**

## Diagnostics
- Plate: share of per-plate offset energy carried by the shared offset, ||d||^2 / mean_p ||d_p||^2: median 0.16 (IQR 0.14-0.18); 1 = identical offset on every plate.
- Per-target-line median ridge coverage: lowest NCI-H661 0.001, SW 1271 0.003, CHP-212 0.102; highest A498 0.983, HEC-1-A 0.984, NCI-H2347 0.985.
- Cross term 2<e_A, d>: median -0.0059 vs Delta^2 median 0.0503.

## Post-hoc (exploratory, not pre-registered)
spearman cov vs D2 -0.827, vs sB2-sA2 -0.984, vs total energy change -0.984
sB2/sA2 ratio median 0.78 IQR 0.50-1.23
spearman cov vs (qA^2-D2)/sB2 0.982
