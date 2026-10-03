"""Exploratory D2 (2026-10-03): DESIGNED fitting pilots + RANDOM calibration pilots.

The cross-line map of xmap.py is learned from m_fit "fitting" pilots and calibrated on m_cal random pilots. The fitting
pilots may be chosen by ANY rule that uses only source-line information (their A responses), because split conformal
only needs the calibration pilots and the test perturbation to be exchangeable: both are drawn at random from the panel
that remains after the fitting pilots are removed. So a lab can choose informative perturbations to learn the map and
still get an exact guarantee.
Designs for the fitting pilots (from the calibration pool, using only Y_A):
  random     uniform
  kmeans     k-means++ centres in the space of the top-20 source PCs (nearest real perturbation to each centre)
  maxmin     farthest-point (max-min distance) in the same space, started at the largest response
  big        the m_fit perturbations with the largest A response (a hand-picked "strong hits" panel)
  dpp        greedy volume (pivoted Cholesky on the linear kernel of PC scores), i.e. D-optimal-like
Map: reduced-rank ridge (rrr) and combo (rrr on [Y_A, ridge-GO]) from xmap.py, hyper-parameters by CV in the fitting
pilots; m_cal = 50 random pilots for split conformal. Metrics: test MSE relative to the pilot mean of the SAME fitting
pilots, gap closed to the noise floor, exact split-conformal radius and coverage.
--common-test (2026-10-03, after the Codex review): one random test set of 200 perturbations is fixed per seed BEFORE the
designs choose, so all designs are scored on the same test perturbations; only rrr is fitted. "kmeans" is k-means++
seeding (nearest real perturbation to each seed), not fitted k-means; "big" = largest centred source-PC norm.
usage: uv run python src/xdesign.py [--seeds 20] [--pairs all|smoke] [--mfit 25,50,100]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import kmeans_plusplus
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ceiling import load_pair_named  # noqa: E402
from data import go_annotations, go_embedding  # noqa: E402
from day1 import rmse, split_cp_q  # noqa: E402
from xmap import cv_pick, p_combo, p_mean, p_rrr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "results"


def design(name, S, pool, k, rng):
    """S: PC scores of all perturbations (source); pool: candidate indices; returns k indices from pool."""
    X = S[pool]
    if name == "random":
        return rng.choice(pool, k, replace=False)
    if name == "big":
        return pool[np.argsort(-np.linalg.norm(X, axis=1))[:k]]
    if name == "kmeans":
        C, _ = kmeans_plusplus(X, k, random_state=int(rng.integers(1 << 30)))
        out = []
        for c in C:
            d = np.linalg.norm(X - c, axis=1); d[out] = np.inf; out.append(int(np.argmin(d)))
        return pool[out]
    if name == "maxmin":
        out = [int(np.argmax(np.linalg.norm(X, axis=1)))]; d = np.linalg.norm(X - X[out[0]], axis=1)
        for _ in range(k - 1):
            j = int(np.argmax(d)); out.append(j); d = np.minimum(d, np.linalg.norm(X - X[j], axis=1))
        return pool[out]
    if name == "dpp":
        K = X @ X.T + 1e-6 * np.eye(len(X)); diag = np.diag(K).copy(); out = []; L = np.zeros((k, len(X)))
        for t in range(k):
            j = int(np.argmax(diag)); out.append(j)
            v = (K[j] - L[:t, j] @ L[:t]) / np.sqrt(max(diag[j], 1e-12)); L[t] = v; diag = diag - v ** 2; diag[out] = -np.inf
        return pool[out]
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=20); ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--pairs", default="all"); ap.add_argument("--mfit", default="25,50,100"); ap.add_argument("--mcal", type=int, default=50); ap.add_argument("--only", default="")
    ap.add_argument("--common-test", action="store_true", help="fix one test set per seed BEFORE design selection (Codex review)")
    a = ap.parse_args(); al = a.alpha; MF = tuple(int(x) for x in a.mfit.split(","))
    compact = dict(np.load(ROOT / "data" / "nadig_compact.npz", allow_pickle=True))
    NZ = np.load(ROOT / "data" / "noise.npz", allow_pickle=True)
    g3 = {str(g): i for i, g in enumerate(compact["genes"])}
    lines = ("ess", "rpe1", "hepg2", "jurkat")
    pairs = [(A, B) for A in lines for B in lines if A != B]
    if a.pairs == "smoke": pairs = [("ess", "rpe1"), ("hepg2", "jurkat")]
    if a.only: pairs = [tuple(a.only.split(","))]
    designs = ("random", "kmeans", "maxmin", "big", "dpp")
    grid = [(k, l) for k in (2, 5, 10, 20) for l in (10.0, 100.0, 1000.0)]
    go = go_annotations(); cache = {}; res = {}
    for A, B in pairs:
        YA0, YB0, targets, gnames = load_pair_named(A, B, compact, cache)
        Z, _, _, has = go_embedding(targets, go); YA0, YB0, Z, targets = YA0[has], YB0[has], Z[has], targets[has]; N = len(Z)
        pz = {p: i for i, p in enumerate(NZ[f"{B}_perts"])}; EPS = NZ[f"{B}_eps"].astype(np.float64)
        prow = np.array([pz.get(t, -1) for t in targets]); gcol = np.array([g3.get(g, -1) for g in gnames])
        raw = {}; put = lambda k, v: raw.setdefault(k, []).append(float(v))  # noqa: E731
        for s in range(a.seeds):
            rng = np.random.default_rng(1000 + s)
            r = np.random.default_rng(s).permutation(N); tr, pool = r[: int(0.4 * N)], r[int(0.7 * N):]
            top = np.argsort(YA0[tr].var(0))[::-1][:1000]; sd = YA0[tr][:, top].std(0) + 1e-8
            YA, YB = YA0[:, top] / sd, YB0[:, top] / sd
            eps = np.full_like(YB, np.nan); okr = prow >= 0; okc = gcol[top] >= 0
            eps[np.ix_(okr, okc)] = EPS[np.ix_(prow[okr], gcol[top][okc])] / sd[okc]; eps[~np.isfinite(eps)] = np.nan
            usable = np.isfinite(eps).mean(1) >= 0.95; gm = np.isfinite(eps[usable]).mean(0) >= 0.99
            YA, YB, eps = YA[:, gm], YB[:, gm], np.where(np.isnan(eps[:, gm]), 0.0, eps[:, gm])
            pool = pool[usable[pool]]
            if a.common_test:                       # same 200 test perturbations for every design; designs pick from the rest
                pool = rng.permutation(pool); test_fixed, pool = pool[:200], pool[200:]
            Pr = Ridge(alpha=10.0).fit(Z[tr], YA[tr]).predict(Z)
            Vt = np.linalg.svd(YA[tr] - YA[tr].mean(0), full_matrices=False)[2][:20]
            S = (YA - YA[tr].mean(0)) @ Vt.T
            for mf in MF:
                if mf + a.mcal + (0 if a.common_test else 100) > len(pool): continue
                for dn in designs:
                    fitp = design(dn, S, pool, mf, rng)
                    rest = np.setdiff1d(pool, fitp); rest = rng.permutation(rest)
                    calp, test = rest[: a.mcal], rest[a.mcal:]          # random calibration pilots; test = the rest
                    if a.common_test: test = test_fixed
                    noise = np.mean(eps[test] ** 2)
                    f0 = p_mean(YA[fitp], YB[fitp], Pr[fitp], None)
                    mse0 = np.mean((f0(YA[test], Pr[test]) - YB[test]) ** 2)
                    q0 = split_cp_q(rmse(f0(YA[calp], Pr[calp]), YB[calp]), al)
                    put(f"{dn}_mean_mse_{mf}", mse0); put(f"{dn}_mean_q_{mf}", q0)
                    for nm, fit in ((("rrr", p_rrr),) if a.common_test else (("rrr", p_rrr), ("combo", p_combo))):
                        hp = cv_pick(fit, grid, YA[fitp], YB[fitp], Pr[fitp]); f = fit(YA[fitp], YB[fitp], Pr[fitp], hp)
                        mse = np.mean((f(YA[test], Pr[test]) - YB[test]) ** 2)
                        q = split_cp_q(rmse(f(YA[calp], Pr[calp]), YB[calp]), al)
                        put(f"{dn}_{nm}_mse_{mf}", mse); put(f"{dn}_{nm}_gap_{mf}", (mse0 - mse) / (mse0 - noise))
                        put(f"{dn}_{nm}_q_{mf}", q); put(f"{dn}_{nm}_cov_{mf}", (rmse(f(YA[test], Pr[test]), YB[test]) <= q).mean())
                    put(f"{dn}_noise_{mf}", noise)
        x = {k: float(np.mean(v)) for k, v in raw.items()}; res[f"{A}->{B}"] = {"summary": x, "raw": raw}
        for mf in MF:
            if f"random_rrr_mse_{mf}" not in x: continue
            base = x[f"random_rrr_mse_{mf}"]
            print(f"{A}->{B} mfit={mf} rrr MSE vs random-design rrr: "
                  + " ".join(f"{dn} {100 * (x[f'{dn}_rrr_mse_{mf}'] / base - 1):+.1f}%" for dn in designs)
                  + " | gap closed (rrr) " + " ".join(f"{dn} {100 * x[f'{dn}_rrr_gap_{mf}']:+.0f}%" for dn in designs)
                  + f" | radius rrr/mean (random) {x[f'random_rrr_q_{mf}'] / x[f'random_mean_q_{mf}']:.3f}"
                  + f" (kmeans) {x[f'kmeans_rrr_q_{mf}'] / x[f'random_mean_q_{mf}']:.3f}"
                  + f" | cov rrr random/kmeans/big {x[f'random_rrr_cov_{mf}']:.3f}/{x[f'kmeans_rrr_cov_{mf}']:.3f}/{x[f'big_rrr_cov_{mf}']:.3f}", flush=True)
    tag = ("smoke" if a.pairs == "smoke" else f"s{a.seeds}") + ("_ct" if a.common_test else "")
    if a.only:
        (OUT / "xdesign_pairs").mkdir(exist_ok=True); (OUT / "xdesign_pairs" / f"{a.only.replace(',', '-')}_{tag}.json").write_text(json.dumps(res))
    else:
        (OUT / f"xdesign_{tag}_metrics.json").write_text(json.dumps(res))
    print("XDESIGN_DONE", flush=True)


if __name__ == "__main__":
    main()
