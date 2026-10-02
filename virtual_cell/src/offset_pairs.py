"""E-A (discussion_round2.md §4): is the offset a property of the CONTEXT (cell line) or of the SCREEN (batch)?

Pairs: K562-ess <-> K562-gwps (same cell line, different screen), K562-gwps -> RPE1 (second source screen),
K562-ess -> RPE1 (reference). Shared targets (first perturbation per target), shared finite readout genes, top 1000 by
summed variance, standardised per gene within each screen (as context.load_pair).
Reported per pair and model (mean, ridge10 on GO z, fit in A):
  Delta2   ||mean(Y_B) - mean(Y_A)||^2 / G over calibration targets (the screen/context offset energy)
  sA2,sB2  residual energy of the model in A and in B after removing the offset
  covB_src            coverage in B of the source-calibrated ball
  covB_src_offset     same source radius, predictions shifted by the oracle offset
  m=20/50: pilot-only split CP; offset-adapted with jackknife (LOO) radius R and exact outer ball (m+1)/m * R
usage: uv run python src/offset_pairs.py [--seeds 20]"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import anndata as ad
import numpy as np
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapt import loo_radius, fit_adapt  # noqa: E402
from data import DATA, go_annotations, go_embedding, target_gene  # noqa: E402
from day1 import rmse, split_cp_q  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
F = {"ess": "K562_essential_normalized_bulk_01.h5ad", "gwps": "K562_gwps_normalized_bulk_01.h5ad", "rpe1": "rpe1_normalized_bulk_01.h5ad",
     "hepg2": "hepg2_pb.h5ad", "jurkat": "jurkat_pb.h5ad",
     "hepg2e": "hepg2_even_pb.h5ad", "hepg2o": "hepg2_odd_pb.h5ad", "k562m": "k562mine_pb.h5ad"}  # hepg2/jurkat: Nadig 2025 raw counts -> pod/pseudobulk.py


def load_one(f):
    a = ad.read_h5ad(DATA / f)
    names = np.asarray(a.obs_names)
    keep = np.array([not re.search(r"non-targeting|control", n, re.I) for n in names])
    t = np.array([target_gene(n) for n in names[keep]])
    _, first = np.unique(t, return_index=True); first = np.sort(first)
    X = np.asarray(a.X, dtype=np.float32)[keep][first]
    return X, t[first], np.asarray(a.var_names)


def load_pair(A, B, cache, n_genes=1000):
    for k in (A, B):
        if k not in cache:
            cache[k] = load_one(F[k])
    (XA, tA, gA), (XB, tB, gB) = cache[A], cache[B]
    ct = sorted(set(tA) & set(tB)); cg = sorted(set(gA) & set(gB))
    ia = {x: i for i, x in enumerate(tA)}; ib = {x: i for i, x in enumerate(tB)}
    ga = {x: i for i, x in enumerate(gA)}; gb = {x: i for i, x in enumerate(gB)}
    YA = XA[[ia[x] for x in ct]][:, [ga[x] for x in cg]].astype(np.float64)
    YB = XB[[ib[x] for x in ct]][:, [gb[x] for x in cg]].astype(np.float64)
    fin = np.isfinite(YA).all(0) & np.isfinite(YB).all(0)
    YA, YB = YA[:, fin], YB[:, fin]
    top = np.argsort(YA.var(0) + YB.var(0))[::-1][:n_genes]
    YA, YB = YA[:, top], YB[:, top]
    return YA / (YA.std(0) + 1e-8), YB / (YB.std(0) + 1e-8), np.array(ct)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=20); ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--pairs", default="ess:gwps,gwps:ess,gwps:rpe1,ess:rpe1"); ap.add_argument("--out", default="offset_pairs_metrics.json")
    a = ap.parse_args(); alpha = a.alpha
    go = go_annotations(); cache = {}; res = {}
    for A, B in [tuple(x.split(":")) for x in a.pairs.split(",")]:
        YA, YB, targets = load_pair(A, B, cache)
        Z, _, _, has_go = go_embedding(targets, go)
        YA, YB, Z = YA[has_go], YB[has_go], Z[has_go]; N = len(Z)
        cc = float(np.mean([np.corrcoef(YA[i], YB[i])[0, 1] for i in range(N)]))
        rec = {}
        for s in range(a.seeds):
            rng = np.random.default_rng(s); r = rng.permutation(N)
            tr, ca, te_all = r[: int(0.4 * N)], r[int(0.4 * N): int(0.7 * N)], r[int(0.7 * N):]
            te = te_all[100:]
            mods = {"mean": lambda idx, mu=YA[tr].mean(0): np.tile(mu, (len(idx), 1))}
            rg = Ridge(alpha=10.0).fit(Z[tr], YA[tr]); mods["ridge10"] = lambda idx, rg=rg: rg.predict(Z[idx])
            delta = YB[ca].mean(0) - YA[ca].mean(0)
            for name, f in mods.items():
                d = rec.setdefault(name, {})
                put = lambda k, v, d=d: d.setdefault(k, []).append(float(v))  # noqa: E731
                put("Delta2", (delta ** 2).mean())
                rA = YA[ca] - f(ca); rB = YB[ca] - f(ca) - delta
                put("sA2", ((rA - rA.mean(0)) ** 2).mean()); put("sB2", ((rB - rB.mean(0)) ** 2).mean())
                qA = split_cp_q(rmse(f(ca), YA[ca]), alpha)
                put("covB_src", (rmse(f(te), YB[te]) <= qA).mean())
                put("covB_src_offset", (rmse(f(te) + delta, YB[te]) <= qA).mean())
                put("qA", qA); put("qB_oracle", split_cp_q(rmse(f(ca), YB[ca]), alpha))
                for m in (20, 50):
                    pil = te_all[:m]; Pp, Pt = f(pil), f(te)
                    q0 = split_cp_q(rmse(Pp, YB[pil]), alpha); st0 = rmse(Pt, YB[te])
                    put(f"pilot_m{m}_cov", (st0 <= q0).mean()); put(f"pilot_m{m}_q", q0)
                    R = loo_radius("offset", Pp, YB[pil], alpha); st = rmse(fit_adapt("offset", Pp, YB[pil])(Pt), YB[te])
                    put(f"jk_m{m}_cov", (st <= R).mean()); put(f"jk_m{m}_q", R)
                    Rx = (m + 1) / m * R
                    put(f"exact_m{m}_cov", (st <= Rx).mean()); put(f"exact_m{m}_q", Rx)
        agg = {n: {k: float(np.mean(v)) for k, v in d.items()} for n, d in rec.items()}
        key = f"{A}->{B}"; res[key] = {"N": int(N), "cross_corr": cc, **agg}
        print(f"== {key} N={N} per-target cross corr {cc:.3f}", flush=True)
        for n, d in agg.items():
            print(f"  {n:8s} " + " ".join(f"{k}={v:.3f}" for k, v in d.items()), flush=True)
    (OUT / a.out).write_text(json.dumps(res, indent=1))
    print("OFFSET_PAIRS_DONE", flush=True)


if __name__ == "__main__":
    main()
