"""Summary of tuned GEARS on 12 directed pairs x 10 seeds (results/gears12/seed_lines.txt, per-seed lines from the runs).
Per pair: mean source radius (qA) and transferred coverage (covB) for mean / ridge / GEARS; paired difference
GEARS - ridge in covB with a 95 % t-interval over seeds; source radius ratio GEARS / ridge."""
import collections
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

R = Path(__file__).resolve().parents[1] / "results"
d = collections.defaultdict(lambda: collections.defaultdict(list))
for line in open(R / "gears12" / "seed_lines.txt", encoding="utf-8"):
    m = re.match(r"g12_(\S+) +seed \d+: mean: qA ([\d.]+) covB ([\d.]+) ridge10: qA ([\d.]+) covB ([\d.]+) gears: qA ([\d.]+) covB ([\d.]+)", line.strip())
    if m:
        for k, x in zip(["mqA", "mcov", "rqA", "rcov", "gqA", "gcov"], map(float, m.groups()[1:])):
            d[m.group(1)][k].append(x)
out = {}
for p, v in sorted(d.items()):
    v = {k: np.array(x) for k, x in v.items()}; n = len(v["gcov"]); diff = v["gcov"] - v["rcov"]
    h = stats.t.ppf(0.975, n - 1) * diff.std(ddof=1) / np.sqrt(n)
    out[p] = {"n": n, **{k: float(x.mean()) for k, x in v.items()}, "diff": float(diff.mean()), "ci": [float(diff.mean() - h), float(diff.mean() + h)],
              "qratio": float((v["gqA"] / v["rqA"]).mean())}
    o = out[p]
    print(f"{p:14s} n={n} qA mean/ridge/GEARS {o['mqA']:.3f}/{o['rqA']:.3f}/{o['gqA']:.3f} (GEARS/ridge {o['qratio']:.3f}) | covB {o['mcov']:.3f}/{o['rcov']:.3f}/{o['gcov']:.3f} | GEARS-ridge {o['diff']:+.3f} [{o['ci'][0]:+.3f},{o['ci'][1]:+.3f}]")
lower = sum(o["diff"] < 0 for o in out.values()); sig = sum(o["ci"][1] < 0 for o in out.values()); higher_sig = sum(o["ci"][0] > 0 for o in out.values())
ridge_below_mean = sum(o["rcov"] < o["mcov"] for o in out.values())
print(f"GEARS covB < ridge in {lower}/12; CI below 0 in {sig}/12; CI above 0 in {higher_sig}/12; ridge < mean in {ridge_below_mean}/12")
(R / "gears12_summary.json").write_text(json.dumps(out, indent=1))
