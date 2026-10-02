## 1. Partial Spearman
Spearman(cov, Delta2) -0.827; Spearman(cov, log var ratio) -0.984; Spearman(Delta2, log var ratio) 0.817
partial(cov, Delta2 | log var ratio) -0.227; partial(cov, log var ratio | Delta2) -0.950

## 2. Variance ratio versus treated-cell numbers, per target line
over 47 target lines: Spearman(median log var ratio, median n_cells_trt) -0.837; Spearman(median coverage, median n_cells) 0.806
- NCI-H661: median coverage 0.001, median var ratio 3.52, median treated cells per unit 92
- SW 1271: median coverage 0.003, median var ratio 4.13, median treated cells per unit 148
- CHP-212: median coverage 0.102, median var ratio 2.80, median treated cells per unit 353
- AN3 CA: median coverage 0.109, median var ratio 2.52, median treated cells per unit 379
- SNU-1: median coverage 0.498, median var ratio 1.66, median treated cells per unit 738
- A498: median coverage 0.983, median var ratio 0.46, median treated cells per unit 1868
- HEC-1-A: median coverage 0.984, median var ratio 0.50, median treated cells per unit 1767
- NCI-H2347: median coverage 0.985, median var ratio 0.52, median treated cells per unit 1608
- all lines: median treated cells per unit, range 92-4210, median 1312

## 3. Inversion restricted to under-covering pairs (predict-the-mean coverage < 0.90)
787 of 2162 pairs; ridge10 < mean in 93.4% (median gap -0.89 pp); source radius ridge/mean median -0.37%
ladder subset, under-covering pairs: 37; within-pair Spearman median 0.26; other-lines worst in 70%; other-lines vs mean: source radius median -9.5%, coverage gap median -2.4 pp, lower in 81%

## 4. Over-coverage side
pairs with ridge coverage > 0.93: 55.8%; < 0.85: 30.3%; < 0.5: 13.2%. Among over-covering pairs the source radius exceeds the target's own calibrated radius by median +37.7% (IQR +21.1 to +64.5%).

## 5. Regret of the rho-hat rule (100-pair subset, ridge10)
- m=20: wrong arm in 0 pairs; radius regret median 0.00%, max 0.00%
- m=50: wrong arm in 20 pairs; radius regret median 0.44%, max 1.66%
- m=100: wrong arm in 10 pairs; radius regret median 0.17%, max 1.12%
850 HT-29->HepG2/C3A 358 D2 0.127 covsrc 0.671 mean 0.671 ad50 0.925 792s
