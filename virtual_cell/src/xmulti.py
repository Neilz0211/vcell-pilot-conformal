"""Exploratory D1b (2026-10-03): predict a target CRISPRi line from the measured responses of the OTHER THREE lines.

Targets shared by all four lines (K562 essential, RPE1, HepG2, Jurkat). For target line B, the sources are the other
three lines; the "source train" split (40 %) defines readout genes (largest mean variance over the sources) and per-gene
scales (pooled SD over the sources), so no target response enters preprocessing. Fixed 280-target test set; pilots =
first m of the calibration pool; only perturbations with split-half noise in B (noise energy of the test set known).
Predictors (fitted on the m pilots, hyper-parameters by 5-fold CV inside the pilots):
  mean      pilot mean of B
  avg_off   average of the three sources + pilot offset                 (mean-offset adapter, Prop A)
  synth     per-gene offset + one weight per source line (ridge on the m x G pilot entries)
  rrr1      reduced-rank ridge from the best single source (by CV)      (xmap.py "rrr")
  rrr3      reduced-rank ridge from the three sources concatenated
  synth_rrr synth + reduced-rank ridge of its residual on the concatenated sources
Metrics: test MSE, gap closed to the noise floor (relative to `mean`), exact split-conformal radius (fit m/2, calibrate
m/2) relative to `mean`, and coverage.
usage: uv run python src/xmulti.py [--seeds 20] [--ms 25,50,100,200]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from day1 import rmse, split_cp_q  # noqa: E402
from rho_rule import load_line  # noqa: E402
from xmap import _rrr, cv_pick, kfold  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "results"
GRID = [(k, l) for k in (2, 5, 10, 20) for l in (10.0, 100.0, 1000.0)]


def f_mean(Xs, Y, hp):
    yb = Y.mean(0); return lambda Xs2: np.tile(yb, (len(Xs2[0]), 1))


def f_avg_off(Xs, Y, hp):
    a = sum(Xs) / len(Xs); b = (Y - a).mean(0); return lambda Xs2: sum(Xs2) / len(Xs2) + b


def f_synth(Xs, Y, hp):
    lam = hp; xb = [x.mean(0) for x in Xs]; yb = Y.mean(0)
    F = np.stack([(x - m).ravel() for x, m in zip(Xs, xb)], 1); w = np.linalg.solve(F.T @ F + lam * np.eye(F.shape[1]), F.T @ (Y - yb).ravel())
    return lambda Xs2: yb + sum(wi * (x - m) for wi, x, m in zip(w, Xs2, xb))


def f_rrr1(Xs, Y, hp):
    j, k, lam = hp; f = _rrr(Xs[j], Y, k, lam); return lambda Xs2: f(Xs2[j])


def f_rrr3(Xs, Y, hp):
    k, lam = hp; f = _rrr(np.hstack(Xs), Y, k, lam); return lambda Xs2: f(np.hstack(Xs2))


def f_synth_rrr(Xs, Y, hp):
    lam, k, lam2 = hp; fs = f_synth(Xs, Y, lam); R = Y - fs(Xs); fr = _rrr(np.hstack(Xs), R, k, lam2)
    return lambda Xs2: fs(Xs2) + fr(np.hstack(Xs2))


def cv(fit, grid, Xs, Y):
    if len(grid) == 1: return grid[0]
    best, bh = np.inf, grid[0]
    for hp in grid:
        e = 0.0
        for f in kfold(len(Y)):
            tr = np.setdiff1d(np.arange(len(Y)), f); p = fit([x[tr] for x in Xs], Y[tr], hp)
            e += ((p([x[f] for x in Xs]) - Y[f]) ** 2).sum()
        if e < best: best, bh = e, hp
    return bh


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=20); ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--ms", default="25,50,100,200"); ap.add_argument("--only", default=""); a = ap.parse_args(); al = a.alpha; MS = tuple(int(x) for x in a.ms.split(","))
    compact = dict(np.load(ROOT / "data" / "nadig_compact.npz", allow_pickle=True))
    NZ = np.load(ROOT / "data" / "noise.npz", allow_pickle=True); g3 = {str(g): i for i, g in enumerate(compact["genes"])}
    lines = ("ess", "rpe1", "hepg2", "jurkat"); cache = {}
    L = {k: load_line(k, compact, cache) for k in lines}
    ct = sorted(set.intersection(*[set(L[k][1]) for k in lines])); cg = sorted(set.intersection(*[set(L[k][2]) for k in lines]))
    M = {}
    for k in lines:
        X, t, g = L[k]; it = {x: i for i, x in enumerate(t)}; ig = {x: i for i, x in enumerate(g)}
        M[k] = X[[it[x] for x in ct]][:, [ig[x] for x in cg]].astype(np.float64)
    fin = np.all([np.isfinite(M[k]).all(0) for k in lines], 0); cg = np.array(cg)[fin]
    for k in lines: M[k] = M[k][:, fin]
    N = len(ct); print("shared targets", N, "genes", len(cg), flush=True)
    res = {}
    for B in ([a.only] if a.only else lines):
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
            Xs0 = [x[:, gm] for x in Xs0]; YB = YB[:, gm]; eps = np.where(np.isnan(eps[:, gm]), 0.0, eps[:, gm])
            cal, test = cal[usable[cal]], test[usable[test]]; put("noise", np.mean(eps[test] ** 2))
            preds = {"mean": (f_mean, [None]), "avg_off": (f_avg_off, [None]),
                     "synth": (f_synth, [1.0, 100.0, 1e4]),
                     "rrr1": (f_rrr1, [(j, k, l) for j in range(3) for k, l in GRID]),
                     "rrr3": (f_rrr3, GRID),
                     "synth_rrr": (f_synth_rrr, [(100.0, k, l) for k, l in GRID])}
            for m in MS:
                if m > len(cal): continue
                pil = cal[:m]; h = m // 2
                for nm, (fit, grid) in preds.items():
                    hp = cv(fit, grid, [x[pil] for x in Xs0], YB[pil]); f = fit([x[pil] for x in Xs0], YB[pil], hp)
                    put(f"{nm}_mse_{m}", np.mean((f([x[test] for x in Xs0]) - YB[test]) ** 2))
                    hp2 = cv(fit, grid, [x[pil[:h]] for x in Xs0], YB[pil[:h]]); f2 = fit([x[pil[:h]] for x in Xs0], YB[pil[:h]], hp2)
                    q = split_cp_q(rmse(f2([x[pil[h:]] for x in Xs0]), YB[pil[h:]]), al)
                    put(f"{nm}_q_{m}", q); put(f"{nm}_cov_{m}", (rmse(f2([x[test] for x in Xs0]), YB[test]) <= q).mean())
        x = {k: float(np.mean(v)) for k, v in raw.items()}
        for m in MS:
            if f"mean_mse_{m}" not in x: continue
            for nm in ("avg_off", "synth", "rrr1", "rrr3", "synth_rrr"):
                x[f"{nm}_gap_{m}"] = float(np.mean([(mm - xx) / (mm - nn) for mm, xx, nn in zip(raw[f"mean_mse_{m}"], raw[f"{nm}_mse_{m}"], raw["noise"])]))
                x[f"{nm}_dq_{m}"] = float(np.mean([q / q0 - 1 for q, q0 in zip(raw[f"{nm}_q_{m}"], raw[f"mean_q_{m}"])]))
            print(f"->{B} m={m} noise/mse_mean {x['noise'] / x[f'mean_mse_{m}']:.2f} | gap closed "
                  + " ".join(f"{nm} {100 * x[f'{nm}_gap_{m}']:+.0f}%" for nm in ("avg_off", "synth", "rrr1", "rrr3", "synth_rrr"))
                  + " | split radius vs mean " + " ".join(f"{nm} {100 * x[f'{nm}_dq_{m}']:+.1f}%" for nm in ("avg_off", "synth", "rrr1", "rrr3", "synth_rrr"))
                  + f" | cov mean/rrr3 {x[f'mean_cov_{m}']:.3f}/{x[f'rrr3_cov_{m}']:.3f}", flush=True)
        res[f"others->{B}"] = {"summary": x, "raw": raw}
    if a.only:
        (OUT / "xmulti_pairs").mkdir(exist_ok=True); (OUT / "xmulti_pairs" / f"others-{a.only}_s{a.seeds}.json").write_text(json.dumps(res))
    else:
        (OUT / f"xmulti_s{a.seeds}_metrics.json").write_text(json.dumps(res))
    print("XMULTI_DONE", flush=True)


if __name__ == "__main__":
    main()
