"""Fit/calibrate allocation for the learned cross-line map (2026-10-03, after the Codex review of main_v2).

xmap.py fits the reduced-rank map on half of the m pilots and calibrates split conformal on the other half. The half/half
split is a choice, not a validity requirement: any fixed allocation of random pilots gives an exact split-conformal set.
Here the fitting fraction varies (0.5, 0.6, 0.7, 0.8; at least 9 calibration pilots so that the 90% set is finite).
Same leak-free design, seeds, test sets and pilots as xmap.py / xpropa.py (noise-availability filter on, for comparability
with Table 2). Output: split-conformal radius and coverage per fraction; compare with xpropa's Prop. A adapted ridge.
The 0.5 fraction reproduces xmap.py's rrr split exactly (same h = m//2), so it can be skipped with --fracs 0.6,0.7.
usage: uv run python src/xsplit.py [--seeds 20] [--pairs ess-rpe1,...] [--fracs 0.6,0.7]"""
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
from xmap import cv_pick, p_rrr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "results"
GRID = [(k, l) for k in (2, 5, 10, 20) for l in (10.0, 100.0, 1000.0)]
FRACS = (0.5, 0.6, 0.7, 0.8)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=20); ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--pairs", default="all"); ap.add_argument("--fracs", default="0.5,0.6,0.7,0.8")
    a = ap.parse_args(); al = a.alpha; FR = tuple(float(x) for x in a.fracs.split(","))
    compact = dict(np.load(ROOT / "data" / "nadig_compact.npz", allow_pickle=True))
    NZ = np.load(ROOT / "data" / "noise.npz", allow_pickle=True); g3 = {str(g): i for i, g in enumerate(compact["genes"])}
    lines = ("ess", "rpe1", "hepg2", "jurkat")
    pairs = [(A, B) for A in lines for B in lines if A != B]
    if a.pairs != "all": pairs = [tuple(p.split("-")) for p in a.pairs.split(",")]
    go = go_annotations(); cache = {}; res = {}
    for A, B in pairs:
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
                for fr in FR:
                    h = int(round(fr * m))
                    if m - h < 9: continue
                    hp = cv_pick(p_rrr, GRID, YA[pil[:h]], YB[pil[:h]], Pr[pil[:h]]); f = p_rrr(YA[pil[:h]], YB[pil[:h]], Pr[pil[:h]], hp)
                    q = split_cp_q(rmse(f(YA[pil[h:]], Pr[pil[h:]]), YB[pil[h:]]), al)
                    put(f"rrr_f{int(fr * 100)}_q_{m}", q); put(f"rrr_f{int(fr * 100)}_cov_{m}", (rmse(f(YA[test], Pr[test]), YB[test]) <= q).mean())
        res[f"{A}->{B}"] = {"summary": {k: float(np.mean(v)) for k, v in raw.items()}, "raw": raw}
        x = res[f"{A}->{B}"]["summary"]
        print(f"{A}->{B} " + " | ".join(f"m{m}: " + " ".join(f"f{int(fr * 100)} {x[f'rrr_f{int(fr * 100)}_q_{m}']:.3f}/{x[f'rrr_f{int(fr * 100)}_cov_{m}']:.3f}"
                                                         for fr in FR if f"rrr_f{int(fr * 100)}_q_{m}" in x) for m in (50, 100, 200)), flush=True)
    tag = "all" if a.pairs == "all" else a.pairs.replace(",", "_")
    (OUT / f"xsplit_{tag}_s{a.seeds}.json").write_text(json.dumps(res))
    print("XSPLIT_DONE", flush=True)


if __name__ == "__main__":
    main()
