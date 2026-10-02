"""Inference for the CRISPRi shift-energy law without pseudo-replication.

Delta^2 is symmetric in the two lines, so the 12 directed pairs of Table 1 carry only 6 distinct Delta^2 values. Reports
the directed correlation (as in the paper), the correlation over the 6 undirected pairs (coverage averaged over the two
directions), a bootstrap over undirected pairs, the version with the three same-line controls, and leave-one-line-out.
Inputs: results/{offset,nadig,matrix}_pairs_metrics.json (and the RPE1->K562 row, results/rpe1_ess_pair_metrics.json if
present; otherwise the Table 1 value). Output: results/delta_inference.md.
usage: uv run python src/delta_inference.py"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

R = Path(__file__).resolve().parents[1] / "results"
NAME = {"ess": "k562", "k562m": "k562", "rpe1": "rpe1", "hepg2": "hepg2", "jurkat": "jurkat"}
rows = {}
for f in ("offset_pairs_metrics.json", "nadig_pairs_metrics.json", "matrix_pairs_metrics.json", "rpe1_ess_pair_metrics.json"):
    if not (R / f).exists(): continue
    for k, v in json.load(open(R / f)).items():
        a, b = k.split("->")
        if a not in NAME or b not in NAME or NAME[a] == NAME[b] or "ridge10" not in v: continue
        rows.setdefault((NAME[a], NAME[b]), (v["ridge10"]["Delta2"], v["ridge10"]["covB_src"]))
rows.setdefault(("rpe1", "k562"), (0.234, 0.801))   # Table 1 (offset_pairs.py --pairs rpe1:ess)
assert len(rows) == 12, rows.keys()
keys = sorted(rows); d = np.array([rows[k][0] for k in keys]); c = np.array([rows[k][1] for k in keys])
out = ["# Shift-energy law on CRISPRi without pseudo-replication", ""]
out.append(f"- directed, n = 12: Pearson {pearsonr(d, c)[0]:.3f}, Spearman {spearmanr(d, c)[0]:.3f}")
U = {}
for (a, b), (x, y) in rows.items():
    U.setdefault(tuple(sorted((a, b))), []).append((x, y))
ud = np.array([v[0][0] for v in U.values()]); uc = np.array([np.mean([y for _, y in v]) for v in U.values()])
out.append(f"- undirected, n = 6 (coverage averaged over directions): Pearson {pearsonr(ud, uc)[0]:.3f} (p = {pearsonr(ud, uc)[1]:.4f}), Spearman {spearmanr(ud, uc)[0]:.3f}")
rng = np.random.default_rng(0); bs = []
for _ in range(10000):
    i = rng.integers(0, 6, 6)
    if len(set(ud[i])) >= 3: bs.append(pearsonr(ud[i], uc[i])[0])
out.append(f"- bootstrap over the 6 undirected pairs: 95% interval [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]")
cd = np.r_[ud, [0.011, 0.006, 0.0]]; cc = np.r_[uc, [(0.893 + 0.906) / 2, (0.895 + 0.899) / 2, 0.903]]
out.append(f"- with the three same-line controls (K562 ess/gwps, HepG2 halves, K562 published/reprocessed), n = 9: Pearson {pearsonr(cd, cc)[0]:.3f}")
for L in ("k562", "rpe1", "hepg2", "jurkat"):
    m = np.array([L not in k for k in keys])
    out.append(f"- without {L} (directed n = {m.sum()}): Pearson {pearsonr(d[m], c[m])[0]:.3f}")
(R / "delta_inference.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
