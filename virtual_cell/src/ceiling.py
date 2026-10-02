"""Exploratory: how close are the prediction sets to the measurement-noise ceiling?

Per-perturbation measurement noise at the scale of the full pseudobulk is estimated from split halves of the cells
(gem-group parity; pod/sc_halves.py, pod/pseudobulk.py --parity, pod/noise_export.py -> data/noise.npz). An oracle that
knew each perturbation's expected response exactly would still face this noise, so the conformal quantile of the noise
RMSE is a floor for the radius of ANY predictor of the measured response.
Per directed pair, with the leak-free design of paired.py (genes and scales from the source train split, fixed
280-target test set, pilots = first m of the calibration pool):
  floor     90 % quantile (conformal order statistic) of the noise RMSE over the calibration pool
  radii     pilot-only ridge, adapted ridge (Prop A, m = 50), adapted pilot mean, full-pool calibrated ridge
  noise share of the adapted-ridge residual energy: E||eps||^2 / E||Y_B - prediction||^2 on the test set
usage: uv run python src/ceiling.py [--seeds 20]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import go_annotations, go_embedding  # noqa: E402
from day1 import rmse, split_cp_q  # noqa: E402
from rho_rule import load_line  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "results"


def load_pair_named(A, B, compact, cache):
    (XA, tA, gA), (XB, tB, gB) = load_line(A, compact, cache), load_line(B, compact, cache)
    ct = sorted(set(tA) & set(tB)); cg = sorted(set(gA) & set(gB))
    ia = {x: i for i, x in enumerate(tA)}; ib = {x: i for i, x in enumerate(tB)}
    ga = {x: i for i, x in enumerate(gA)}; gb = {x: i for i, x in enumerate(gB)}
    YA = XA[[ia[x] for x in ct]][:, [ga[x] for x in cg]].astype(np.float64)
    YB = XB[[ib[x] for x in ct]][:, [gb[x] for x in cg]].astype(np.float64)
    fin = np.isfinite(YA).all(0) & np.isfinite(YB).all(0)
    return YA[:, fin], YB[:, fin], np.array(ct), np.array(cg)[fin]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=20); ap.add_argument("--alpha", type=float, default=0.1)
    a = ap.parse_args(); al = a.alpha
    compact = dict(np.load(ROOT / "data" / "nadig_compact.npz", allow_pickle=True))
    NZ = np.load(ROOT / "data" / "noise.npz", allow_pickle=True)
    genes3000 = [str(x) for x in compact["genes"]]; g3 = {g: i for i, g in enumerate(genes3000)}
    lines = ("ess", "rpe1", "hepg2", "jurkat"); go = go_annotations(); cache = {}; res = {}
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
                eps[np.ix_(okr, okc)] = EPS[np.ix_(prow[okr], gcol[top][okc])] / sd[okc]
                eps[~np.isfinite(eps)] = np.nan  # z-scores of genes with ~zero control SD in a half blow up
                usable = np.isfinite(eps).mean(1) >= 0.95; put("noise_cov_genes", okc.mean()); put("noise_perts", usable.mean())
                ne = np.sqrt(np.nanmean(eps ** 2, 1))                      # noise RMSE per perturbation
                P = Ridge(alpha=10.0).fit(Z[tr], YA[tr]).predict(Z)
                calu = cal[usable[cal]]
                put("floor", split_cp_q(ne[calu], al))
                put("full_ridge", split_cp_q(rmse(P[cal], YB[cal]), al))
                m = 50; pil = cal[:m]
                for nm, Q in (("ridge", P), ("mean", np.tile(YA[tr].mean(0), (N, 1)))):
                    E = YB[pil] - Q[pil]; bl = (E.sum(0)[None] - E) / (m - 1)
                    put(f"ad_{nm}", (m + 1) / m * split_cp_q(np.sqrt(((E - bl) ** 2).mean(1)), al))
                put("pil_ridge", split_cp_q(rmse(P[pil], YB[pil]), al))
                b = (YB[pil] - P[pil]).mean(0); tu = test[usable[test]]
                put("noise_share", np.nanmean(eps[tu] ** 2) / np.mean((YB[tu] - P[tu] - b) ** 2))
            x = {k: float(np.mean(v)) for k, v in raw.items()}; res[f"{A}->{B}"] = x
            print(f"{A}->{B} floor {x['floor']:.2f} | ad ridge {x['ad_ridge']:.2f} ad mean {x['ad_mean']:.2f} pilot {x['pil_ridge']:.2f} "
                  f"full {x['full_ridge']:.2f} | floor/ad_ridge {x['floor'] / x['ad_ridge']:.2f} | noise share {x['noise_share']:.2f} "
                  f"(genes {x['noise_cov_genes']:.2f}, perts {x['noise_perts']:.2f})", flush=True)
    (OUT / "ceiling_metrics.json").write_text(json.dumps(res, indent=1))
    print("CEILING_DONE", flush=True)


if __name__ == "__main__":
    main()
