"""Calibration transfer on Tahoe-100M (chemical perturbations), per PREREG_tahoe.md. Pod-side, self-contained.

Per directed pair (A -> B) of cell lines and seed:
  units = drug x dose with n_cells_trt >= 30 in both lines; genes of the 3,000-gene pool finite in every kept unit of
  both lines (units with NaN in genes that are finite in >= 99% of units are dropped first);
  split 40/30/30 train/cal/test GROUPED BY DRUG; readout = top 1,000 genes by source-train variance, both lines
  divided by the source-train per-gene SD (leak-free "sourcetrain");
  models fitted in A: mean, ridge (lambda 1000/100/10/1) on the fingerprint-SVD features, kNN (Tanimoto, k=10),
  other-lines mean (mean LFC of the unit over the lines other than A and B);
  source-calibrated split CP on cal(A) -> coverage on test(B); oracle-offset version; model-free coverage prediction;
  pilots = first m of a random order of cal(B), m in {20, 50, 100}: pilot-only split CP, adapted outer ball
  ((m+1)/m x LOO quantile, Prop A), adapted pilot mean, rho-hat and the pre-registered rule.
Modes: --models core (mean, ridge10; all pairs) or ladder (all models; subset of pairs). --dose high keeps the
highest dose per drug only.
usage: python tahoe_pairs.py DIR --pairs all|subset100 [--models core|ladder] [--seeds 20] [--shard k/N] --out F
"""
import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ALPHA = 0.1
MS = (20, 50, 100)
G_OUT = 1000


def split_cp_q(scores, alpha):
    n = len(scores); k = math.ceil((1 - alpha) * (n + 1))
    return float(np.inf) if k > n else float(np.sort(scores)[k - 1])


def rmse(P, Y):
    return np.sqrt(((P - Y) ** 2).mean(1))


def loo_radius_offset(P, Y, alpha):
    m = len(Y); E = Y - P; tot = E.sum(0)
    b = (tot[None, :] - E) / (m - 1)          # offset fitted without unit i
    R = np.sqrt(((E - b) ** 2).mean(1))       # rmse(P_i + b_-i, Y_i)
    return split_cp_q(R, alpha)


def rho_hat(E):
    m, G = E.shape; eb = E.mean(0)
    s2 = ((E - eb) ** 2).sum() / ((m - 1) * G)
    return float(((eb ** 2).sum() / G - s2 / m) / s2)


D = {}


def init(root, dose):
    D["Y"] = np.load(root + "/Y.npy", mmap_mode="r")
    D["nc"] = np.load(root + "/ncells.npy")
    D["U"] = pd.read_csv(root + "/units.tsv", sep="\t", dtype={"plate": str})
    D["Z"] = np.load(root + "/Z.npy"); D["fp"] = np.load(root + "/fp.npy")
    keep = np.isfinite(D["Z"]).all(1)
    if dose == "high":
        U = D["U"]; mx = U.groupby("drug").concentration.transform("max").values
        keep &= U.concentration.values == mx
    D["unit_ok"] = keep


def tanimoto(A, B):
    A = A.astype(np.float32); B = B.astype(np.float32)
    inter = A @ B.T; return inter / (A.sum(1)[:, None] + B.sum(1)[None, :] - inter + 1e-8)


def run_pair(args):
    a, b, seeds, models = args
    Y, nc, U, Z, FP = D["Y"], D["nc"], D["U"], D["Z"], D["fp"]
    u = np.where(D["unit_ok"] & (nc[a] >= 30) & (nc[b] >= 30))[0]
    YA0 = np.asarray(Y[a][u], np.float64); YB0 = np.asarray(Y[b][u], np.float64)
    F = np.isfinite(YA0) & np.isfinite(YB0)
    good_g = F.mean(0) >= 0.99
    row_ok = F[:, good_g].all(1)
    u, YA0, YB0, F = u[row_ok], YA0[row_ok], YB0[row_ok], F[row_ok]
    col = F.all(0); YA0, YB0 = YA0[:, col], YB0[:, col]
    if col.sum() < G_OUT or len(u) < 300:
        return (a, b), {"skipped": True, "n_units": int(len(u)), "n_genes": int(col.sum())}
    drugs = U.drug.values[u]; plates = U.plate.values[u]
    others = None
    if "otherlines" in models:
        L = Y.shape[0]; acc = np.zeros_like(YA0); cnt = np.zeros_like(YA0)
        gi = np.where(col)[0]
        for l in range(L):
            if l in (a, b): continue
            x = np.asarray(Y[l][u], np.float64)[:, gi]; f = np.isfinite(x)
            acc += np.where(f, x, 0); cnt += f
        others = acc / np.maximum(cnt, 1)
    ud = np.unique(drugs)
    rec = {}
    put = lambda mname, k, v: rec.setdefault(mname, {}).setdefault(k, []).append(float(v))  # noqa: E731
    for s in range(seeds):
        rng = np.random.default_rng(s)
        dp = rng.permutation(ud); nd = len(dp)
        dtr, dca = set(dp[: int(0.4 * nd)]), set(dp[int(0.4 * nd): int(0.7 * nd)])
        tr = np.where([d in dtr for d in drugs])[0]; ca = np.where([d in dca for d in drugs])[0]
        te = np.setdiff1d(np.arange(len(u)), np.concatenate([tr, ca]))
        top = np.argsort(YA0[tr].var(0))[::-1][:G_OUT]; sd = YA0[tr][:, top].std(0) + 1e-8
        YA, YB = YA0[:, top] / sd, YB0[:, top] / sd
        pil_order = rng.permutation(ca)
        delta = YB[ca].mean(0) - YA[ca].mean(0); D2 = float((delta ** 2).mean())
        preds = {}
        mu = YA[tr].mean(0); preds["mean"] = np.tile(mu, (len(u), 1))
        for lam in ((1000, 100, 10, 1) if "ladder" in models else (10,)):
            preds[f"ridge{lam}"] = Ridge(alpha=float(lam)).fit(Z[u][tr], YA[tr]).predict(Z[u])
        if "ladder" in models:
            # only cal/test predictions are used, and their drugs never appear in train (drug-grouped split)
            T = tanimoto(FP[u], FP[u][tr])
            nn = np.argsort(-T, 1)[:, :10]; preds["knn10"] = YA[tr][nn].mean(1)
        if others is not None:
            preds["otherlines"] = others[:, top] / sd
        for name, P in preds.items():
            pa = lambda k, v, name=name: put(name, k, v)  # noqa: E731
            pa("Delta2", D2)
            rA = YA[ca] - P[ca]; rB = YB[ca] - P[ca] - delta
            pa("sA2", ((rA - rA.mean(0)) ** 2).mean()); pa("sB2", ((rB - rB.mean(0)) ** 2).mean())
            pa("cross", 2 * (rA.mean(0) * delta).mean())
            sA = rmse(P[ca], YA[ca]); qA = split_cp_q(sA, ALPHA); st = rmse(P[te], YB[te])
            pa("qA", qA); pa("covB_src", (st <= qA).mean())
            pa("covB_src_offset", (rmse(P[te] + delta, YB[te]) <= qA).mean())
            pa("cov_pred", (sA ** 2 + D2 <= qA ** 2).mean())
            pa("covA_test", (rmse(P[te], YA[te]) <= qA).mean())
            pa("qB_oracle", split_cp_q(rmse(P[ca], YB[ca]), ALPHA))
            rho_or = rho_hat(YB[ca] - P[ca]); pa("rho_or", rho_or)
            for m in MS:
                pil = pil_order[:m]; Pp, Yp = P[pil], YB[pil]
                q0 = split_cp_q(rmse(Pp, Yp), ALPHA); pa(f"pilot_q_{m}", q0); pa(f"pilot_cov_{m}", (st <= q0).mean())
                R = (m + 1) / m * loo_radius_offset(Pp, Yp, ALPHA); bb = (Yp - Pp).mean(0)
                sa = rmse(P[te] + bb, YB[te]); pa(f"ad_q_{m}", R); pa(f"ad_cov_{m}", (sa <= R).mean())
                rh = rho_hat(Yp - Pp); adapt = rh > (3 * m + 1) / (m * (m - 1))
                pa(f"rho_hat_{m}", rh); pa(f"rule_adapt_{m}", adapt)
                pa(f"rule_q_{m}", R if adapt else q0); pa(f"rule_cov_{m}", (sa <= R).mean() if adapt else (st <= q0).mean())
                if name in ("ridge10", "mean"):
                    Mp = np.tile(Yp.mean(0), (len(te), 1))           # adapted pilot mean = pilot mean of target responses
                    Rm = (m + 1) / m * loo_radius_offset(np.tile(mu, (m, 1)), Yp, ALPHA)
                    pa(f"madapted_q_{m}", Rm); pa(f"madapted_cov_{m}", (rmse(Mp, YB[te]) <= Rm).mean())
                    h = m // 2; b2 = (Yp[:h] - Pp[:h]).mean(0)
                    qs = split_cp_q(rmse(Pp[h:] + b2, Yp[h:]), ALPHA)
                    pa(f"split_q_{m}", qs); pa(f"split_cov_{m}", (rmse(P[te] + b2, YB[te]) <= qs).mean())
        if s == 0:   # plate diagnostic on all kept units, seed-0 scaling
            d_all = (YB - YA).mean(0); E_all = float((d_all ** 2).mean())
            pe = [float((((YB - YA)[plates == p]).mean(0) ** 2).mean()) for p in np.unique(plates) if (plates == p).sum() >= 20]
            rec.setdefault("_plate", {"Delta2_pooled_all": E_all, "Delta2_within_plate_mean": float(np.mean(pe)), "n_plates": len(pe)})
    out = {"n_units": int(len(u)), "n_genes_finite": int(col.sum()), "n_drugs": int(len(ud))}
    for name, d in rec.items():
        if name == "_plate": out[name] = d; continue
        out[name] = {k: float(np.mean(v)) for k, v in d.items()}
        out[name]["_seeds"] = {k: v for k, v in d.items() if k.startswith(("ad_q_", "pilot_q_", "madapted_q_", "split_q_"))}
    return (a, b), out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--pairs", default="all")
    ap.add_argument("--models", default="core"); ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--dose", default="all"); ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--limit", type=int); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lines = open(a.root + "/lines.txt").read().split("\n"); L = len(lines)
    allp = [(i, j) for i in range(L) for j in range(L) if i != j]
    if a.pairs == "subset100":
        pick = np.random.default_rng(0).choice(len(allp), 100, replace=False); allp = [allp[k] for k in sorted(pick)]
    if a.limit: allp = allp[: a.limit]
    models = {"core": ("core",), "ladder": ("ladder", "otherlines")}[a.models]
    t0 = time.time(); res = {}
    with ProcessPoolExecutor(a.workers, initializer=init, initargs=(a.root, a.dose)) as ex:
        for k, ((i, j), r) in enumerate(ex.map(run_pair, [(i, j, a.seeds, models) for i, j in allp], chunksize=1)):
            res[f"{lines[i]}->{lines[j]}"] = r
            if k % 50 == 0 or k < 5:
                rr = r.get("ridge10", {})
                print(k, f"{lines[i]}->{lines[j]}", r.get("n_units"), "D2 %.3f covsrc %.3f mean %.3f ad50 %.3f" % (
                    rr.get("Delta2", np.nan), rr.get("covB_src", np.nan), r.get("mean", {}).get("covB_src", np.nan),
                    rr.get("ad_cov_50", np.nan)), "%.0fs" % (time.time() - t0), flush=True)
    json.dump({"lines": lines, "args": vars(a), "pairs": res}, open(a.out, "w"))
    print("PAIRS_DONE", len(res), "%.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
