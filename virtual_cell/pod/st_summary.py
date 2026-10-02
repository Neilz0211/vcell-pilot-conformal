"""Aggregate st_eval outputs over seeds: per directed pair, mean over seeds of source radius, transferred coverage,
oracle-offset coverage and the adapted ball (m = 50), for STATE, ridge and predict-the-mean; prints a LaTeX table body
and the counts quoted in the text.
usage: python st_summary.py st_eval_s0.json [st_eval_s12.json ...]"""
import json
import sys
from collections import defaultdict

import numpy as np

R = {}
for f in sys.argv[1:]:
    R.update(json.load(open(f)))
by = defaultdict(list)
for k, v in R.items():
    by[k.split("|")[0]].append(v)
NAME = {"hepg2": "HepG2", "jurkat": "Jurkat", "k562": "K562", "rpe1": "RPE1"}
mean = lambda vs, m, x: float(np.mean([v[m][x] for v in vs]))  # noqa: E731
rows, cnt = [], defaultdict(int); cov_st, ad_st, orc = [], [], []
for pair in sorted(by):
    vs = by[pair]; a, b = pair.split("->")
    q = {m: mean(vs, m, "qA") for m in ("state", "ridge10", "mean")}
    c = {m: mean(vs, m, "covB_src") for m in ("state", "ridge10", "mean")}
    o = mean(vs, "state", "covB_src_offset"); ac = mean(vs, "state", "ad_cov_50"); aq = {m: mean(vs, m, "ad_q_50") for m in ("state", "ridge10", "mean")}
    cnt["state_wider_than_ridge"] += q["state"] > q["ridge10"]; cnt["state_wider_than_mean"] += q["state"] > q["mean"]
    cnt["state_cov_below_mean"] += c["state"] < c["mean"]; cnt["ridge_cov_below_mean"] += c["ridge10"] < c["mean"]
    cnt["ridge_narrower_than_mean"] += q["ridge10"] < q["mean"]; cnt["state_adq_widest"] += aq["state"] > max(aq["ridge10"], aq["mean"])
    cov_st.append(c["state"]); ad_st.append(ac); orc.append(o)
    rows.append(f"{NAME[a]} $\\to$ {NAME[b]} & {len(vs)} & {vs[0]['n_eval']} & {q['state']:.2f} / {q['ridge10']:.2f} / {q['mean']:.2f} & "
                f"{c['state']:.3f} / {c['ridge10']:.3f} / {c['mean']:.3f} & {o:.3f} & {ac:.3f} & {aq['state']:.2f} / {aq['ridge10']:.2f} / {aq['mean']:.2f}\\\\")
print("\n".join(rows))
print()
for k, v in cnt.items():
    print(f"{k}: {v}/{len(by)}")
print(f"STATE transferred coverage {min(cov_st):.3f}-{max(cov_st):.3f} (median {np.median(cov_st):.3f}); oracle-offset median {np.median(orc):.3f}; "
      f"adapted m=50 {min(ad_st):.3f}-{max(ad_st):.3f}")
seeds = sorted({k.split('|')[1] for k in R})
print("seeds:", seeds)
