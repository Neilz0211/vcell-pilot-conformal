"""Idea 1: use the m pilots twice -- to ADAPT the source model to the new cell line and to CALIBRATE it (leave-one-out).

Context pair K562-essential <-> RPE1. Source model f_A = ridge(GO z) fit on line A. Adapted predictors for line B:
  offset     f_A(z) + b,  b = mean pilot residual (Y_B - f_A) -- a per-gene line offset
  affine     per-gene a_g * f_A(z)_g + b_g fit on pilots by shrunk least squares (a shrunk toward 1)
  pmean      pilot mean response (predict-the-mean in B)
Calibration on the pilots by leave-one-out (jackknife): scores R_i = rmse(predictor fit without pilot i, Y_i), radius =
ceil((m+1)(1-alpha))-th order statistic of the R_i; test sets use the predictor fit on all m pilots. Compared with
plain pilot-only split CP on the unadapted f_A (the E1 baseline). Jackknife coverage is approximate (exact for a
stable estimator); reported, not assumed.
usage: uv run python src/adapt.py [--seeds 30]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context import load_pair  # noqa: E402
from data import go_annotations, go_embedding  # noqa: E402
from day1 import fit_models, rmse, split_cp_q  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"


def fit_adapt(kind, P, Y, lam=5.0):
    """P: source predictions on pilots (m x G), Y: pilot responses in B. Returns a function of source predictions."""
    if kind == "offset":
        b = (Y - P).mean(0); return lambda Q: Q + b
    if kind == "pmean":
        mu = Y.mean(0); return lambda Q: np.tile(mu, (len(Q), 1))
    if kind == "affine":  # per-gene: y = a p + b, ridge-shrink a toward 1
        pm, ym = P.mean(0), Y.mean(0); pc, yc = P - pm, Y - ym
        a = ((pc * yc).sum(0) + lam) / ((pc ** 2).sum(0) + lam)
        b = ym - a * pm
        return lambda Q: a * Q + b
    raise ValueError(kind)


def loo_radius(kind, P, Y, alpha):
    m = len(Y); R = np.empty(m)
    for i in range(m):
        k = np.arange(m) != i
        R[i] = rmse(fit_adapt(kind, P[k], Y[k])(P[i:i + 1]), Y[i:i + 1])[0]
    return split_cp_q(R, alpha)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=30); ap.add_argument("--alpha", type=float, default=0.1)
    a = ap.parse_args(); alpha = a.alpha
    Ys, targets = load_pair()
    Z, _, _, has_go = go_embedding(targets, go_annotations())
    Ys = {k: v[has_go] for k, v in Ys.items()}; Z = Z[has_go]; N = len(Z)
    res = {}
    for A, B in (("K562", "RPE1"), ("RPE1", "K562")):
        for m in (10, 20, 50, 100):
            rec = {}
            for s in range(a.seeds):
                rng = np.random.default_rng(s); r = rng.permutation(N)
                tr, te_all = r[: int(0.4 * N)], r[int(0.7 * N):]
                pil, te = te_all[:m], te_all[100:]  # test set fixed across m (pilots drawn from the first 100)
                f = fit_models(Z[tr], Ys[A][tr], s)
                for base in ("ridge", "mean"):
                    Pp, Pt = f[base](Z[pil]), f[base](Z[te])
                    q0 = split_cp_q(rmse(Pp, Ys[B][pil]), alpha); st0 = rmse(Pt, Ys[B][te])
                    rec.setdefault(f"{base}_pilot", []).append(((st0 <= q0).mean(), q0))
                    for kind in ("offset", "affine", "pmean"):
                        if kind == "pmean" and base == "mean":
                            continue
                        q = loo_radius(kind, Pp, Ys[B][pil], alpha)
                        st = rmse(fit_adapt(kind, Pp, Ys[B][pil])(Pt), Ys[B][te])
                        rec.setdefault(f"{base}_{kind}", []).append(((st <= q).mean(), q))
            row = {k: [float(np.mean([x[0] for x in v])), float(np.mean([x[1] for x in v]))] for k, v in rec.items()}
            res[f"{A}->{B}_m{m}"] = row
            print(f"{A}->{B} m={m}", " ".join(f"{k}:{c:.3f}/{q:.2f}" for k, (c, q) in row.items()), flush=True)
    (OUT / "adapt_metrics.json").write_text(json.dumps(res, indent=1))
    print("ADAPT_DONE", flush=True)


if __name__ == "__main__":
    main()
