"""Score the pre-registered predictions P1-P6 (PREREG_tahoe.md) from tahoe_pairs.py outputs.
usage: python tahoe_summary.py ALL.json SUBSET.json [HIGHDOSE.json] > summary.md"""
import json
import sys

import numpy as np
from scipy.stats import spearmanr

ALL = json.load(open(sys.argv[1])); SUB = json.load(open(sys.argv[2]))
HI = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None
lines = ALL["lines"]
P = {k: v for k, v in ALL["pairs"].items() if not v.get("skipped")}
print(f"# Tahoe-100M calibration transfer: pre-registered predictions\n")
print(f"Pairs analysed: {len(P)} of {len(ALL['pairs'])} (skipped: {len(ALL['pairs']) - len(P)}); "
      f"units per pair median {np.median([v['n_units'] for v in P.values()]):.0f}, drugs {np.median([v['n_drugs'] for v in P.values()]):.0f}\n")

keys = sorted(P); src = [k.split("->")[0] for k in keys]; tgt = [k.split("->")[1] for k in keys]
r = lambda f: np.array([f(P[k]) for k in keys])  # noqa: E731
D2 = r(lambda v: v["ridge10"]["Delta2"]); cov = r(lambda v: v["ridge10"]["covB_src"]); covm = r(lambda v: v["mean"]["covB_src"])
covo = r(lambda v: v["ridge10"]["covB_src_offset"]); covp = r(lambda v: v["ridge10"]["cov_pred"])


def cluster_boot(stat, B=1000, seed=0):
    rng = np.random.default_rng(seed); L = sorted(set(src) | set(tgt)); idx = {l: i for i, l in enumerate(L)}
    si = np.array([idx[s] for s in src]); ti = np.array([idx[t] for t in tgt]); out = []
    for _ in range(B):
        c = np.bincount(rng.integers(0, len(L), len(L)), minlength=len(L))
        w = c[si] * c[ti]; sel = np.repeat(np.arange(len(keys)), w)
        if len(sel) > 10: out.append(stat(sel))
    return np.percentile(out, [2.5, 97.5])


def verdict(ok):
    return "**PASS**" if ok else "**FAIL**"


rs = spearmanr(D2, cov).statistic
ci = cluster_boot(lambda s: spearmanr(D2[s], cov[s]).statistic)
print("## P1 law: Spearman(Delta^2, ridge source-calibrated coverage) < 0, |rho| >= 0.5, cluster CI excludes 0")
print(f"Spearman {rs:.3f}, line-cluster bootstrap 95% CI [{ci[0]:.3f}, {ci[1]:.3f}]; Pearson {np.corrcoef(D2, cov)[0, 1]:.3f}. "
      f"Delta^2 range {D2.min():.3f}-{D2.max():.3f} (median {np.median(D2):.3f}); ridge coverage {cov.min():.3f}-{cov.max():.3f} "
      f"(median {np.median(cov):.3f}); with the oracle offset {np.median(covo):.3f}. {verdict(rs < 0 and abs(rs) >= 0.5 and ci[1] < 0)}\n")

mae = 100 * np.abs(covp - cov).mean(); cc = np.corrcoef(covp, cov)[0, 1]
print("## P2 quantitative: model-free predicted coverage, MAE <= 3 pp and corr >= 0.8")
print(f"MAE {mae:.2f} pp, corr {cc:.3f}, mean signed error {100 * (covp - cov).mean():+.2f} pp. {verdict(mae <= 3 and cc >= 0.8)}\n")

frac = (cov < covm).mean()
S = {k: v for k, v in SUB["pairs"].items() if not v.get("skipped")}
path = ["mean", "ridge1000", "ridge100", "ridge10", "ridge1", "knn10"]
sp = [spearmanr([S[k][m]["qA"] for m in path], [S[k][m]["covB_src"] for m in path]).statistic for k in S]
mods = path + ["otherlines"]
worst = np.mean([min(mods, key=lambda m: S[k][m]["covB_src"]) == "otherlines" for k in S])
print("## P3 inversion: ridge < mean in >= 75% of pairs; ladder Spearman median >= 0.8; other-lines mean worst in >= 60%")
print(f"ridge10 covers less than predict-the-mean in {100 * frac:.1f}% of {len(keys)} pairs (median gap {100 * np.median(cov - covm):+.1f} pp). "
      f"Ladder (100-pair subset, {len(S)} analysed): within-pair Spearman(source radius, transferred coverage) median {np.nanmedian(sp):.2f} "
      f"(IQR {np.nanpercentile(sp, 25):.2f}-{np.nanpercentile(sp, 75):.2f}); other-lines mean has the lowest transferred coverage in "
      f"{100 * worst:.0f}%. {verdict(frac >= 0.75 and np.nanmedian(sp) >= 0.8 and worst >= 0.6)}\n")
print("Mean over the subset, per model: source radius qA / source-calibrated target coverage / test coverage in the source line:")
for m in mods:
    print(f"- {m}: qA {np.mean([S[k][m]['qA'] for k in S]):.3f}, covB_src {np.mean([S[k][m]['covB_src'] for k in S]):.3f}, "
          f"covA_test {np.mean([S[k][m]['covA_test'] for k in S]):.3f}, qB_oracle {np.mean([S[k][m]['qB_oracle'] for k in S]):.3f}")
print()

adc = r(lambda v: v["ridge10"]["ad_cov_50"]); adq = r(lambda v: v["ridge10"]["ad_q_50"]); mq = r(lambda v: v["ridge10"]["madapted_q_50"])
pc = r(lambda v: v["ridge10"]["pilot_cov_50"]); pq = r(lambda v: v["ridge10"]["pilot_q_50"])
print("## P4 repair: adapted-ball coverage at m=50 in [0.88, 0.93]; adapted ridge narrower than adapted pilot mean in >= 60% of pairs")
print(f"adapted coverage mean {adc.mean():.3f} (pair range {adc.min():.3f}-{adc.max():.3f}); pilot-only {pc.mean():.3f}. "
      f"Adapted ridge narrower than adapted pilot mean in {100 * (adq < mq).mean():.1f}% (median {100 * np.median(adq / mq - 1):+.1f}%). "
      f"Adapted vs pilot-only radius: median {100 * np.median(adq / pq - 1):+.1f}%. {verdict(0.88 <= adc.mean() <= 0.93 and (adq < mq).mean() >= 0.6)}\n")
for m in (20, 50, 100):
    print(f"- m={m}: pilot cov {r(lambda v: v['ridge10'][f'pilot_cov_{m}']).mean():.3f}, adapted cov {r(lambda v: v['ridge10'][f'ad_cov_{m}']).mean():.3f}, "
          f"split cov {r(lambda v: v['ridge10'][f'split_cov_{m}']).mean():.3f}; adapted/pilot radius median "
          f"{100 * np.median(r(lambda v: v['ridge10'][f'ad_q_{m}'] / v['ridge10'][f'pilot_q_{m}'] - 1)):+.1f}%, adapted/split "
          f"{100 * np.median(r(lambda v: v['ridge10'][f'ad_q_{m}'] / v['ridge10'][f'split_q_{m}'] - 1)):+.1f}%")
print()

print("## P5 rho-hat rule (100-pair subset, ridge10): better arm >= 80% at m=50,100; rule coverage >= 0.89; |bias| <= 0.05")
ok5 = True
for m in (20, 50, 100):
    hit = []; rc = []; bias = []
    for k in S:
        v = S[k]["ridge10"]; better = v[f"ad_q_{m}"] < v[f"pilot_q_{m}"]; rule = v[f"rule_adapt_{m}"] > 0.5
        hit.append(better == rule); rc.append(v[f"rule_cov_{m}"]); bias.append(v[f"rho_hat_{m}"] - v["rho_or"])
    h = np.mean(hit); print(f"- m={m}: picks the smaller-radius arm in {100 * h:.0f}% of pairs; mean rule coverage {np.mean(rc):.3f}; "
                          f"mean bias of rho-hat {np.mean(bias):+.3f}")
    if m in (50, 100): ok5 &= h >= 0.8 and np.mean(rc) >= 0.89 and abs(np.mean(bias)) <= 0.05
print(f"{verdict(ok5)}\n")

m = 50
rho = r(lambda v: v["ridge10"]["Delta2"] / v["ridge10"]["sB2"])
pred = 100 * ((m + 1) / m * np.sqrt((m / (m - 1)) / (1 + rho)) - 1); obs = 100 * (adq / pq - 1)
c6 = np.corrcoef(pred, obs)[0, 1]; mae6 = np.abs(pred - obs).mean()
print("## P6 Prop C: corr(predicted, observed radius change at m=50) >= 0.7, MAE <= 3 pp")
print(f"corr {c6:.3f}, MAE {mae6:.2f} pp, slope {np.polyfit(pred, obs, 1)[0]:.2f}; rho range {rho.min():.3f}-{rho.max():.3f}. {verdict(c6 >= 0.7 and mae6 <= 3)}\n")

print("## Diagnostics")
pl = [P[k]["_plate"] for k in keys if "_plate" in P[k]]
ratio = np.array([x["Delta2_pooled_all"] / x["Delta2_within_plate_mean"] for x in pl])
print(f"- Plate: share of per-plate offset energy carried by the shared offset, ||d||^2 / mean_p ||d_p||^2: median {np.median(ratio):.2f} "
      f"(IQR {np.percentile(ratio, 25):.2f}-{np.percentile(ratio, 75):.2f}); 1 = identical offset on every plate.")
per_t = {}
for k, c in zip(keys, cov): per_t.setdefault(k.split('->')[1], []).append(c)
med_t = sorted((np.median(v), t) for t, v in per_t.items())
print(f"- Per-target-line median ridge coverage: lowest {', '.join(f'{t} {c:.3f}' for c, t in med_t[:3])}; highest {', '.join(f'{t} {c:.3f}' for c, t in med_t[-3:])}.")
print(f"- Cross term 2<e_A, d>: median {np.median(r(lambda v: v['ridge10']['cross'])):+.4f} vs Delta^2 median {np.median(D2):.4f}.")
if HI:
    H = {k: v for k, v in HI["pairs"].items() if not v.get("skipped")}; hk = sorted(H)
    hd = np.array([H[k]["ridge10"]["Delta2"] for k in hk]); hc = np.array([H[k]["ridge10"]["covB_src"] for k in hk])
    hm = np.array([H[k]["mean"]["covB_src"] for k in hk])
    print(f"- Highest dose only ({len(hk)} pairs): Spearman(Delta^2, coverage) {spearmanr(hd, hc).statistic:.3f}; ridge < mean in "
          f"{100 * (hc < hm).mean():.1f}%; adapted coverage m=50 {np.mean([H[k]['ridge10']['ad_cov_50'] for k in hk]):.3f}.")
