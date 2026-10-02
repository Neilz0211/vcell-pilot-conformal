"""E-C (discussion_round2.md §4): adapt-then-calibrate under LIBRARY shift (K562 gwps, essential vs non-essential).

Same scenarios and splits as pilot2.library (natural->essential, disjoint, no shift). Methods on the m essential-library
pilots: pilot-only split CP, offset-adapted exact ball ((m+1)/m * LOO radius). Also Delta2 = offset energy between the
calibration library and the screening library (mean residual difference of the ridge model).
Go: disjoint radius >= 5 % below pilot-only -> library shift also carries an offset. Kill: <= 2 %.
usage: uv run python src/lib_adapt.py [--seeds 30]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapt import fit_adapt, loo_radius  # noqa: E402
from data import go_annotations, go_embedding  # noqa: E402
from day1 import fit_models, rmse, split_cp_q  # noqa: E402
from day2 import draw, load_gwps  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=30); ap.add_argument("--alpha", type=float, default=0.1)
    a = ap.parse_args(); alpha = a.alpha
    Y, targets, ctrl, is_ess = load_gwps()
    Zg, M, _, has_go = go_embedding(targets, go_annotations())
    k = has_go
    Y, Zg, ctrl, is_ess = Y[k], Zg[k], ctrl[k], is_ess[k]
    Z = np.c_[Zg, (ctrl - ctrl.mean()) / ctrl.std()]; N = len(Y)
    res = {}
    for scen, (cs, ts) in {"natural_to_ess": (0.2, 1.0), "disjoint": (0.0, 1.0), "no_shift": (0.2, 0.2)}.items():
        rec = {}
        for s in range(a.seeds):
            rng = np.random.default_rng(s)
            r = rng.permutation(N); tr, pool = r[: int(0.3 * N)], r[int(0.3 * N):]
            f = fit_models(Z[tr], Y[tr], s)["ridge"]
            ca = draw(rng, pool, is_ess, cs, 1500); rest = np.setdiff1d(pool, ca)
            te_all = rng.permutation(draw(rng, rest, is_ess, ts, 700))
            te = te_all[100:]
            put = lambda kk, v: rec.setdefault(kk, []).append(float(v))  # noqa: E731
            d = (Y[te] - f(Z[te])).mean(0) - (Y[ca] - f(Z[ca])).mean(0)
            put("Delta2", (d ** 2).mean())
            qs = split_cp_q(rmse(f(Z[ca]), Y[ca]), alpha); st0 = rmse(f(Z[te]), Y[te])
            put("src_cov", (st0 <= qs).mean()); put("src_q", qs)
            for m in (20, 50, 100):
                pil = te_all[:m]; Pp, Pt = f(Z[pil]), f(Z[te])
                q0 = split_cp_q(rmse(Pp, Y[pil]), alpha)
                put(f"pilot_m{m}_cov", (st0 <= q0).mean()); put(f"pilot_m{m}_q", q0)
                R = (m + 1) / m * loo_radius("offset", Pp, Y[pil], alpha)
                st = rmse(fit_adapt("offset", Pp, Y[pil])(Pt), Y[te])
                put(f"exact_m{m}_cov", (st <= R).mean()); put(f"exact_m{m}_q", R)
        row = {kk: float(np.mean(v)) for kk, v in rec.items()}
        res[scen] = row
        print(scen, " ".join(f"{kk}={v:.3f}" for kk, v in row.items()), flush=True)
    (OUT / "lib_adapt_metrics.json").write_text(json.dumps(res, indent=1))
    print("LIB_ADAPT_DONE", flush=True)


if __name__ == "__main__":
    main()
