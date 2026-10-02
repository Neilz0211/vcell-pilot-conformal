"""Post-hoc (exploratory, NOT pre-registered) analyses of the Tahoe run, suggested in review after the P1-P6 scoring.
1 partial Spearman of coverage on Delta^2 given log variance ratio, and the reverse;
2 per-target-line variance ratio versus treated-cell numbers (biological scale vs measurement noise);
3 the inversion restricted to under-covering pairs (predict-the-mean coverage < 0.90);
4 the over-coverage side (share of pairs > 0.93, radius waste qA / qB_oracle);
5 regret of the rho-hat rule when it picks the wrong arm.
usage: python tahoe_posthoc.py tahoe_all.json tahoe_sub.json /root/tahoe/compact"""
import json
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr

ALL = json.load(open(sys.argv[1]))["pairs"]; SUB = json.load(open(sys.argv[2]))["pairs"]; root = sys.argv[3]
P = {k: v for k, v in ALL.items() if not v.get("skipped")}; keys = sorted(P)
r = lambda m, f: np.array([f(P[k][m]) for k in keys])  # noqa: E731
cov = r("ridge10", lambda v: v["covB_src"]); D2 = r("ridge10", lambda v: v["Delta2"])
lvr = np.log(r("ridge10", lambda v: v["sB2"]) / r("ridge10", lambda v: v["sA2"]))


def pspear(x, y, z):
    rx, ry, rz = (rankdata(a) for a in (x, y, z))
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)  # noqa: E731
    return np.corrcoef(res(rx), res(ry))[0, 1]


print("## 1. Partial Spearman")
print(f"Spearman(cov, Delta2) {spearmanr(cov, D2).statistic:.3f}; Spearman(cov, log var ratio) {spearmanr(cov, lvr).statistic:.3f}; "
      f"Spearman(Delta2, log var ratio) {spearmanr(D2, lvr).statistic:.3f}")
print(f"partial(cov, Delta2 | log var ratio) {pspear(cov, D2, lvr):.3f}; partial(cov, log var ratio | Delta2) {pspear(cov, lvr, D2):.3f}\n")

print("## 2. Variance ratio versus treated-cell numbers, per target line")
lines = open(root + "/lines.txt").read().split("\n"); nc = np.load(root + "/ncells.npy")
med_nc = {l: float(np.median(nc[i][nc[i] >= 0])) for i, l in enumerate(lines)}
tl = {}
for k, c, v in zip(keys, cov, lvr):
    t = k.split("->")[1]; tl.setdefault(t, []).append((c, v))
T = sorted(tl); mc = np.array([np.median([x[0] for x in tl[t]]) for t in T]); mv = np.array([np.median([x[1] for x in tl[t]]) for t in T])
n = np.array([med_nc[t] for t in T])
print(f"over {len(T)} target lines: Spearman(median log var ratio, median n_cells_trt) {spearmanr(mv, n).statistic:.3f}; "
      f"Spearman(median coverage, median n_cells) {spearmanr(mc, n).statistic:.3f}")
for t in [T[i] for i in np.argsort(mc)[:5]] + [T[i] for i in np.argsort(mc)[-3:]]:
    i = T.index(t); print(f"- {t}: median coverage {mc[i]:.3f}, median var ratio {np.exp(mv[i]):.2f}, median treated cells per unit {n[i]:.0f}")
print(f"- all lines: median treated cells per unit, range {n.min():.0f}-{n.max():.0f}, median {np.median(n):.0f}\n")

print("## 3. Inversion restricted to under-covering pairs (predict-the-mean coverage < 0.90)")
covm = r("mean", lambda v: v["covB_src"]); u = covm < 0.90
print(f"{u.sum()} of {len(keys)} pairs; ridge10 < mean in {100 * (cov[u] < covm[u]).mean():.1f}% (median gap {100 * np.median(cov[u] - covm[u]):+.2f} pp); "
      f"source radius ridge/mean median {100 * np.median(r('ridge10', lambda v: v['qA'])[u] / r('mean', lambda v: v['qA'])[u] - 1):+.2f}%")
S = {k: v for k, v in SUB.items() if not v.get("skipped") and v["mean"]["covB_src"] < 0.90}
path = ["mean", "ridge1000", "ridge100", "ridge10", "ridge1", "knn10"]; mods = path + ["otherlines"]
sp = [spearmanr([S[k][m]["qA"] for m in path], [S[k][m]["covB_src"] for m in path]).statistic for k in S]
worst = np.mean([min(mods, key=lambda m: S[k][m]["covB_src"]) == "otherlines" for k in S])
ol = np.array([S[k]["otherlines"]["covB_src"] - S[k]["mean"]["covB_src"] for k in S])
olq = np.array([S[k]["otherlines"]["qA"] / S[k]["mean"]["qA"] - 1 for k in S])
print(f"ladder subset, under-covering pairs: {len(S)}; within-pair Spearman median {np.nanmedian(sp):.2f}; other-lines worst in {100 * worst:.0f}%; "
      f"other-lines vs mean: source radius median {100 * np.median(olq):+.1f}%, coverage gap median {100 * np.median(ol):+.1f} pp, lower in {100 * (ol < 0).mean():.0f}%\n")

print("## 4. Over-coverage side")
qA = r("ridge10", lambda v: v["qA"]); qo = r("ridge10", lambda v: v["qB_oracle"]); o = cov > 0.93
print(f"pairs with ridge coverage > 0.93: {100 * o.mean():.1f}%; < 0.85: {100 * (cov < 0.85).mean():.1f}%; < 0.5: {100 * (cov < 0.5).mean():.1f}%. "
      f"Among over-covering pairs the source radius exceeds the target's own calibrated radius by median {100 * np.median(qA[o] / qo[o] - 1):+.1f}% "
      f"(IQR {100 * np.percentile(qA[o] / qo[o] - 1, 25):+.1f} to {100 * np.percentile(qA[o] / qo[o] - 1, 75):+.1f}%).\n")

print("## 5. Regret of the rho-hat rule (100-pair subset, ridge10)")
SS = {k: v for k, v in SUB.items() if not v.get("skipped")}
for m in (20, 50, 100):
    reg = []
    for k, v in SS.items():
        x = v["ridge10"]; better_ad = x[f"ad_q_{m}"] < x[f"pilot_q_{m}"]; rule_ad = x[f"rule_adapt_{m}"] > 0.5
        if better_ad != rule_ad:
            reg.append(abs(x[f"ad_q_{m}"] - x[f"pilot_q_{m}"]) / min(x[f"ad_q_{m}"], x[f"pilot_q_{m}"]))
    print(f"- m={m}: wrong arm in {len(reg)} pairs; radius regret median {100 * np.median(reg) if reg else 0:.2f}%, max {100 * max(reg) if reg else 0:.2f}%")
