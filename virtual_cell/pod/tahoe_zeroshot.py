"""Exploratory I1/T4: calibrate a NEW cell line with zero or few pilots by borrowing calibration scores from other lines.

For a source line A, ridge (lambda = 10, fingerprint-SVD features) is fitted on A's train drugs (drug-grouped split as
tahoe_pairs.py; readout genes and scales from A's train split). For every other line L, scores s_{L,i} = rmse(f(z_i),
Y_L,i) are computed on the calibration drugs and on the test drugs. For a target line B:
  src      source-calibrated radius (status quo)
  pool     zero pilots: conformal quantile of the calibration scores of all lines C != A, B, pooled (lines exchangeable)
  poolfree zero pilots: scores of each line C divided by a scale predicted from FREE covariates of C (log median
           number of control cells per plate); the pooled normalised quantile times B's predicted scale. The scale model
           is a least-squares fit, across lines C != B, of log median calibration score on the covariates.
  poolcell same with the median number of treated cells per unit as covariate (not free: an upper bound)
  pil10 / pil50   pilot-only split conformal on 10 / 50 random calibration units of B
  hyb10    10 pilots of B used only to estimate B's scale (median pilot score); pooled normalised quantile (normalised by
           each line's median calibration score) times that scale.
Coverage on B's test units; per-line coverage summarised as mean, 5th percentile, share of lines below 0.85; radius
relative to pil50. Validity of the pooled arms rests on exchangeability of lines (and is approximate for hyb10).
usage: python tahoe_zeroshot.py /root/tahoe --sources 12 --seeds 5 --out tahoe_zeroshot.json"""
import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

AL = 0.1


def q_conf(s, al=AL):
    n = len(s); k = math.ceil((1 - al) * (n + 1))
    return float("inf") if k > n else float(np.sort(s)[k - 1])


def rmse(P, Y):
    return np.sqrt(((P - Y) ** 2).mean(1))


D = {}


def init(root):
    c = root + "/compact"
    D["Y"] = np.load(c + "/Y.npy", mmap_mode="r"); D["nc"] = np.load(c + "/ncells.npy")
    D["U"] = pd.read_csv(c + "/units.tsv", sep="\t", dtype={"plate": str}); D["Z"] = np.load(c + "/Z.npy")
    lines = open(c + "/lines.txt").read().split("\n"); all_lines = open(root + "/lines.txt").read().split("\n")
    nctrl = np.load(root + "/ncells_ctrl.npy"); li = [all_lines.index(l) for l in lines]
    ctrl = []
    for i in li:
        v = nctrl[i][nctrl[i] >= 0]; ctrl.append(np.median(v) if len(v) else np.nan)
    D["cov_free"] = np.log(np.array(ctrl, float))
    D["cov_cell"] = np.log(np.array([np.median(D["nc"][k][D["nc"][k] >= 0]) for k in range(len(lines))], float))
    D["lines"] = lines


def run(args):
    a, seed = args
    Y, nc, U, Z = D["Y"], D["nc"], D["U"], D["Z"]; L = Y.shape[0]
    ok = np.isfinite(Z).all(1) & (nc[a] >= 30)
    rng = np.random.default_rng(seed); drugs = U.drug.values
    ud = rng.permutation(np.unique(drugs[ok])); nd = len(ud)
    dtr, dca = set(ud[: int(0.4 * nd)]), set(ud[int(0.4 * nd): int(0.7 * nd)])
    tr = np.where(ok & np.isin(drugs, list(dtr)))[0]; ca_all = np.where(ok & np.isin(drugs, list(dca)))[0]
    te_all = np.where(ok & ~np.isin(drugs, list(dtr | dca)))[0]
    YA = np.asarray(Y[a], np.float64)
    fin = np.isfinite(YA[tr]).all(0)
    top = np.where(fin)[0][np.argsort(np.nanvar(YA[tr][:, fin], 0))[::-1][:1000]]; sd = YA[tr][:, top].std(0) + 1e-8
    f = Ridge(alpha=10.0).fit(Z[tr], YA[tr][:, top] / sd); P = f.predict(np.nan_to_num(Z))  # rows without features are never scored
    sc = {}
    for l in range(L):
        Yl = np.asarray(Y[l], np.float64)[:, top] / sd
        good = np.isfinite(Yl).all(1) & (nc[l] >= 30)
        ca, te = ca_all[good[ca_all]], te_all[good[te_all]]
        if len(ca) < 60 or len(te) < 60: continue
        sc[l] = (rmse(P[ca], Yl[ca]), rmse(P[te], Yl[te]))
    out = {}
    if a not in sc: return a, seed, out
    qA = q_conf(sc[a][0])
    targets = [b for b in sc if b != a]
    med = {l: np.median(sc[l][0]) for l in sc}
    for b in targets:
        others = [c for c in targets if c != b]
        r = {}
        te = sc[b][1]
        r["src"] = qA
        r["pool"] = q_conf(np.concatenate([sc[c][0] for c in others]))
        for nm, cov in (("poolfree", D["cov_free"]), ("poolcell", D["cov_cell"])):
            xs = np.array([cov[c] for c in others]); ys = np.log([med[c] for c in others]); okc = np.isfinite(xs)
            if okc.sum() < 5 or not np.isfinite(cov[b]): continue
            beta = np.polyfit(xs[okc], ys[okc], 1)
            pred = lambda c: np.exp(np.polyval(beta, cov[c]))  # noqa: E731
            norm = np.concatenate([sc[c][0] / pred(c) for c in others if np.isfinite(cov[c])])
            r[nm] = q_conf(norm) * pred(b)
        pr = np.random.default_rng(seed * 1000 + b)
        cal_b = sc[b][0]; perm = pr.permutation(len(cal_b))
        r["pil10"] = q_conf(cal_b[perm[:10]]); r["pil50"] = q_conf(cal_b[perm[:50]])
        normp = np.concatenate([sc[c][0] / med[c] for c in others])
        r["hyb10"] = q_conf(normp) * np.median(cal_b[perm[:10]])
        out[b] = {k: (float((te <= v).mean()), float(v)) for k, v in r.items()}
    return a, seed, out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--sources", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--workers", type=int, default=24); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lines = open(a.root + "/compact/lines.txt").read().split("\n")
    srcs = sorted(np.random.default_rng(0).choice(len(lines), a.sources, replace=False).tolist())
    res = []
    with ProcessPoolExecutor(a.workers, initializer=init, initargs=(a.root,)) as ex:
        for s, seed, out in ex.map(run, [(s, sd) for s in srcs for sd in range(a.seeds)]):
            for b, r in out.items():
                res.append({"src": lines[s], "tgt": lines[b], "seed": seed, **{k: v for k, v in r.items()}})
            print("done source", lines[s], "seed", seed, len(out), flush=True)
    json.dump(res, open(a.out, "w"))
    arms = ["src", "pool", "poolfree", "poolcell", "pil10", "pil50", "hyb10"]
    df = pd.DataFrame([{"tgt": r["tgt"], **{f"{k}_cov": r[k][0] for k in arms if k in r}, **{f"{k}_q": r[k][1] for k in arms if k in r}} for r in res])
    per_line = df.groupby("tgt").mean(numeric_only=True)
    for k in arms:
        if f"{k}_cov" not in df: continue
        c = per_line[f"{k}_cov"]; rr = (df[f"{k}_q"] / df["pil50_q"]).replace([np.inf], np.nan)
        print(f"{k:9s} coverage mean {df[f'{k}_cov'].mean():.3f} | per-line mean: 5th pct {c.quantile(0.05):.3f}, min {c.min():.3f}, "
              f"lines < 0.85: {(c < 0.85).sum()}/{len(c)} | radius / pil50 median {rr.median():.3f}", flush=True)
    print("ZEROSHOT_DONE", flush=True)


if __name__ == "__main__":
    main()
