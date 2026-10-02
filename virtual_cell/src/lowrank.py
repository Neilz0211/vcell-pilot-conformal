"""E-B (discussion_round2.md §4): richer adaptation on the m pilots, beyond the per-gene offset.

Context pair K562-essential <-> RPE1 (context.load_pair). Source model f_A = ridge10(GO z) fit in line A.
Adaptations fit on the m pilots (all symmetric in the pilots):
  offset        f_A(z) + b
  offset_pcK    offset, then project the pilot residuals on their top-K PCs (K=1,2,5) and regress the K coefficients on
                f_A(z)'s own top-K PC scores (ridge, shrunk) -- a low-rank prediction-dependent correction
  offset_zfeat  offset, then ridge (strong, alpha=100) from z to the residual, trained on the m pilots ('fine-tune')
Calibration: LOO radius R (jackknife) with the full-pilot fit at test time; coverage is measured, not assumed (the
(m+1)/m exactness argument is proved only for the pure offset).
usage: uv run python src/lowrank.py [--seeds 20]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context import load_pair  # noqa: E402
from data import go_annotations, go_embedding  # noqa: E402
from day1 import fit_models, rmse, split_cp_q  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"


def fitter(kind, Pp, Zp, Y):
    b = (Y - Pp).mean(0)
    if kind == "offset":
        return lambda P, Zq: P + b
    R = Y - Pp - b
    if kind.startswith("offset_pc"):
        K = int(kind[9:])
        U = np.linalg.svd(R, full_matrices=False)[2][:K]  # K x G residual axes
        V = np.linalg.svd(Pp - Pp.mean(0), full_matrices=False)[2][:K]  # K x G prediction axes
        reg = Ridge(alpha=10.0).fit((Pp - Pp.mean(0)) @ V.T, R @ U.T)
        mu = Pp.mean(0)
        return lambda P, Zq: P + b + reg.predict((P - mu) @ V.T).reshape(len(P), -1) @ U
    if kind == "offset_zfeat":
        reg = Ridge(alpha=100.0).fit(Zp, R)
        return lambda P, Zq: P + b + reg.predict(Zq)
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=20); ap.add_argument("--alpha", type=float, default=0.1)
    a = ap.parse_args(); alpha = a.alpha
    Ys, targets = load_pair()
    Z, _, _, has_go = go_embedding(targets, go_annotations())
    Ys = {k: v[has_go] for k, v in Ys.items()}; Z = Z[has_go]; N = len(Z)
    kinds = ("offset", "offset_pc1", "offset_pc2", "offset_pc5", "offset_zfeat")
    res = {}
    for A, B in (("K562", "RPE1"), ("RPE1", "K562")):
        for m in (20, 50, 100):
            rec = {}
            for s in range(a.seeds):
                rng = np.random.default_rng(s); r = rng.permutation(N)
                tr, te_all = r[: int(0.4 * N)], r[int(0.7 * N):]
                pil, te = te_all[:m], te_all[100:]
                f = fit_models(Z[tr], Ys[A][tr], s)["ridge"]
                Pp, Pt, Yp = f(Z[pil]), f(Z[te]), Ys[B][pil]
                q0 = split_cp_q(rmse(Pp, Yp), alpha)
                rec.setdefault("pilot", []).append(((rmse(Pt, Ys[B][te]) <= q0).mean(), q0))
                for kind in kinds:
                    L = np.empty(m)
                    for i in range(m):
                        kk = np.arange(m) != i
                        L[i] = rmse(fitter(kind, Pp[kk], Z[pil][kk], Yp[kk])(Pp[i:i + 1], Z[pil][i:i + 1]), Yp[i:i + 1])[0]
                    R = split_cp_q(L, alpha)
                    st = rmse(fitter(kind, Pp, Z[pil], Yp)(Pt, Z[te]), Ys[B][te])
                    rec.setdefault(kind, []).append(((st <= R).mean(), R))
            row = {k: [float(np.mean([x[0] for x in v])), float(np.mean([x[1] for x in v]))] for k, v in rec.items()}
            res[f"{A}->{B}_m{m}"] = row
            print(f"{A}->{B} m={m}", " ".join(f"{k}:{c:.3f}/{q:.3f}" for k, (c, q) in row.items()), flush=True)
    (OUT / "lowrank_metrics.json").write_text(json.dumps(res, indent=1))
    print("LOWRANK_DONE", flush=True)


if __name__ == "__main__":
    main()
