"""Reference radii for the D1 comparisons (2026-10-03): the paper's CURRENT recipe, Proposition A's outer ball, which
reuses all m pilots for the offset and for calibration (no split). Same leak-free design, seeds, test sets and pilots
(first m of the calibration pool) as xmap.py / xmulti.py, so radii can be compared pair by pair with the split-conformal
radii of the learned maps stored in results/xmap_pairs and results/xmulti_pairs.
Arms (all exact, Prop. A outer ball (m+1)/m * R_(k)):
  propA_mean    adapted predict-the-mean (pilot mean)              [single-source design of xmap.py]
  propA_ridge   adapted ridge(GO)                                  [single-source design]
  propA_meas    measured source response + pilot offset            [single-source design]
  propA_avg     average of the other three lines + pilot offset    [multi-source design of xmulti.py]
  propA_mmean   pilot mean                                         [multi-source design]
usage: uv run python src/xpropa.py [--seeds 20]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ceiling import load_pair_named  # noqa: E402
from data import go_annotations, go_embedding  # noqa: E402
from day1 import rmse, split_cp_q  # noqa: E402
from rho_rule import load_line  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "results"


def propa(Q, Y, al):
    """Outer ball of Prop. A for prediction Q + pilot offset; returns (offset, radius)."""
    m = len(Y); E = Y - Q; b = E.mean(0); bl = (E.sum(0)[None] - E) / (m - 1)
    return b, (m + 1) / m * split_cp_q(np.sqrt(((E - bl) ** 2).mean(1)), al)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=20); ap.add_argument("--alpha", type=float, default=0.1)
    a = ap.parse_args(); al = a.alpha
    compact = dict(np.load(ROOT / "data" / "nadig_compact.npz", allow_pickle=True))
    NZ = np.load(ROOT / "data" / "noise.npz", allow_pickle=True); g3 = {str(g): i for i, g in enumerate(compact["genes"])}
    lines = ("ess", "rpe1", "hepg2", "jurkat"); go = go_annotations(); cache = {}; res = {}
    # ---- single-source design (xmap.py) ----
    for A in lines:
        for B in lines:
            if A == B: continue
            YA0, YB0, targets, gnames = load_pair_named(A, B, compact, cache)
            Z, _, _, has = go_embedding(targets, go); YA0, YB0, Z, targets = YA0[has], YB0[has], Z[has], targets[has]; N = len(Z)
            pz = {p: i for i, p in enumerate(NZ[f"{B}_perts"])}; EPS = NZ[f"{B}_eps"].astype(np.float64)
            prow = np.array([pz.get(t, -1) for t in targets]); gcol = np.array([g3.get(g, -1) for g in gnames])
            raw = {}; put = lambda k, v: raw.setdefault(k, []).append(float(v))  # noqa: E731
            for s in range(a.seeds):
                r = np.random.default_rng(s).permutation(N); tr, pool = r[: int(0.4 * N)], r[int(0.7 * N):]
                test, cal = pool[-280:], pool[:-280]
                top = np.argsort(YA0[tr].var(0))[::-1][:1000]; sd = YA0[tr][:, top].std(0) + 1e-8
                YA, YB = YA0[:, top] / sd, YB0[:, top] / sd
                eps = np.full_like(YB, np.nan); okr = prow >= 0; okc = gcol[top] >= 0
                eps[np.ix_(okr, okc)] = EPS[np.ix_(prow[okr], gcol[top][okc])] / sd[okc]; eps[~np.isfinite(eps)] = np.nan
                usable = np.isfinite(eps).mean(1) >= 0.95; gm = np.isfinite(eps[usable]).mean(0) >= 0.99
                YA, YB = YA[:, gm], YB[:, gm]; cal, test = cal[usable[cal]], test[usable[test]]
                Pr = Ridge(alpha=10.0).fit(Z[tr], YA[tr]).predict(Z)
                for m in (50, 100, 200):
                    if m > len(cal): continue
                    pil = cal[:m]
                    for nm, Q in (("mean", np.zeros_like(YB)), ("ridge", Pr), ("meas", YA)):
                        b, R = propa(Q[pil], YB[pil], al)
                        put(f"propA_{nm}_q_{m}", R); put(f"propA_{nm}_cov_{m}", (rmse(Q[test] + b, YB[test]) <= R).mean())
                        put(f"propA_{nm}_mse_{m}", np.mean((Q[test] + b - YB[test]) ** 2))
            res[f"{A}->{B}"] = {k: float(np.mean(v)) for k, v in raw.items()} | {"raw": raw}
            x = res[f"{A}->{B}"]; print(f"{A}->{B} propA q mean/ridge/meas m100 {x['propA_mean_q_100']:.3f}/{x['propA_ridge_q_100']:.3f}/{x['propA_meas_q_100']:.3f}", flush=True)
    # ---- multi-source design (xmulti.py) ----
    L = {k: load_line(k, compact, cache) for k in lines}
    ct = sorted(set.intersection(*[set(L[k][1]) for k in lines])); cg = sorted(set.intersection(*[set(L[k][2]) for k in lines]))
    M = {}
    for k in lines:
        X, t, g = L[k]; it = {x: i for i, x in enumerate(t)}; ig = {x: i for i, x in enumerate(g)}
        M[k] = X[[it[x] for x in ct]][:, [ig[x] for x in cg]].astype(np.float64)
    fin = np.all([np.isfinite(M[k]).all(0) for k in lines], 0); cg = np.array(cg)[fin]
    for k in lines: M[k] = M[k][:, fin]
    N = len(ct)
    for B in lines:
        S = [k for k in lines if k != B]
        pz = {p: i for i, p in enumerate(NZ[f"{B}_perts"])}; EPS = NZ[f"{B}_eps"].astype(np.float64)
        prow = np.array([pz.get(t, -1) for t in ct]); gcol = np.array([g3.get(g, -1) for g in cg])
        raw = {}; put = lambda k, v: raw.setdefault(k, []).append(float(v))  # noqa: E731
        for s in range(a.seeds):
            r = np.random.default_rng(s).permutation(N); tr, pool = r[: int(0.4 * N)], r[int(0.7 * N):]
            test, cal = pool[-280:], pool[:-280]
            v = np.mean([M[k][tr].var(0) for k in S], 0); top = np.argsort(v)[::-1][:1000]
            sd = np.sqrt(np.mean([M[k][tr][:, top].var(0) for k in S], 0)) + 1e-8
            Xs0 = [M[k][:, top] / sd for k in S]; YB = M[B][:, top] / sd
            eps = np.full_like(YB, np.nan); okr = prow >= 0; okc = gcol[top] >= 0
            eps[np.ix_(okr, okc)] = EPS[np.ix_(prow[okr], gcol[top][okc])] / sd[okc]; eps[~np.isfinite(eps)] = np.nan
            usable = np.isfinite(eps).mean(1) >= 0.95; gm = np.isfinite(eps[usable]).mean(0) >= 0.99
            Xs0 = [x[:, gm] for x in Xs0]; YB = YB[:, gm]; cal, test = cal[usable[cal]], test[usable[test]]
            avg = sum(Xs0) / 3
            for m in (25, 50, 100, 200):
                if m > len(cal): continue
                pil = cal[:m]
                for nm, Q in (("mmean", np.zeros_like(YB)), ("avg", avg)):
                    b, R = propa(Q[pil], YB[pil], al)
                    put(f"propA_{nm}_q_{m}", R); put(f"propA_{nm}_cov_{m}", (rmse(Q[test] + b, YB[test]) <= R).mean())
        res[f"others->{B}"] = {k: float(np.mean(v)) for k, v in raw.items()} | {"raw": raw}
        x = res[f"others->{B}"]; print(f"others->{B} propA q mmean/avg m50 {x['propA_mmean_q_50']:.3f}/{x['propA_avg_q_50']:.3f}", flush=True)
    (OUT / f"xpropa_s{a.seeds}_metrics.json").write_text(json.dumps(res))
    print("XPROPA_DONE", flush=True)


if __name__ == "__main__":
    main()
