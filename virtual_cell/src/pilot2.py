"""E1 (discussion_round1.md §2.2): pilot-anchored conformal, replacing test-then-switch.

Methods (alpha = 0.1), per scenario and pilot count m:
  src      source-calibrated CP (estimated weights for library shift, unweighted for context shift) -- no pilots
  pilot    split CP on the m pilot scores (infinite when m < 1/alpha - 1)
  P2       ratio-CP: r = S / sigma(z), sigma = kNN regression (k=20) of SOURCE calibration scores on z; split CP on the
           pilot ratios -> radius kappa * sigma(z). Exact marginal coverage for any target law (pilots exchangeable
           with test); tighter than `pilot` only if sigma's shape transfers. (With a constant sigma, P2 == pilot.)
  P3_g     transfer-CP: weighted quantile of {source scores (weights normalised to mean 1, times estimated density
           ratio for library shift), pilot scores (weight g each)}, test point mass g (Barber et al. 2022 form;
           coverage >= 1 - alpha - TV * W_src / (W_src + g (m+1))). g in {1, 5, 20}.
Reported: coverage, finite-set rate, mean radius over finite sets, and the KS distance between source and pilot
score distributions (Delta-hat) next to the true source-CP coverage gap.
Scenarios: library shift on gwps (natural->essential, disjoint, no shift, reverse) and context shift
K562-essential <-> RPE1 (ridge model fit in the source line).
usage: uv run python src/pilot2.py [--seeds 30] [--only library|context]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context import load_pair  # noqa: E402
from data import go_annotations, go_embedding  # noqa: E402
from day1 import fit_models, rmse, split_cp_q, weighted_quantile_cp  # noqa: E402
from day2 import draw, load_gwps  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
MS = (5, 9, 10, 20, 50, 100)
GAMMAS = (1, 5, 20)


def evaluate(alpha, sc, wsrc, Zc, sp_all, Zp_all, wp_all, st, Zt, wt):
    """sc/wsrc/Zc: source calibration scores, weights (mean 1), features; sp_all/Zp_all/wp_all: pilot pool (first m used);
    st/Zt/wt: test. wp/wt are estimated density ratios (1 for context shift) used only by `src`."""
    out = {}
    def put(k, cov, rad):  # noqa: E306
        rad = np.asarray(rad, float); fin = np.isfinite(rad)
        out[k] = (float(np.mean(cov)), float(fin.mean()), float(rad[fin].mean()) if fin.any() else np.nan)
    q_src = np.array([weighted_quantile_cp(sc, wsrc, float(w), alpha)[0] for w in wt])
    put("src", st <= q_src, q_src)
    sig = KNeighborsRegressor(20).fit(Zc, sc)
    sig_t = np.maximum(sig.predict(Zt), 1e-6); sig_p_all = np.maximum(sig.predict(Zp_all), 1e-6)
    for m in MS:
        sp = sp_all[:m]
        qp = split_cp_q(sp, alpha) if m >= int(np.ceil(1 / alpha)) - 1 else np.inf
        put(f"pilot_m{m}", st <= qp, np.full(len(st), qp))
        kap = split_cp_q(sp / sig_p_all[:m], alpha) if m >= int(np.ceil(1 / alpha)) - 1 else np.inf
        put(f"P2_m{m}", st <= kap * sig_t, kap * sig_t)
        for g in GAMMAS:
            q3, _ = weighted_quantile_cp(np.r_[sc, sp], np.r_[wsrc, np.full(m, float(g))], float(g), alpha)
            put(f"P3g{g}_m{m}", st <= q3, np.full(len(st), q3))
        out[f"ks_m{m}"] = (float(ks_2samp(sc, sp).statistic), np.nan, np.nan)
    return out


def aggregate(runs):
    keys = runs[0].keys()
    return {k: [float(np.nanmean([r[k][i] for r in runs])) for i in range(3)] for k in keys}


def library(seeds, alpha):
    Y, targets, ctrl, is_ess = load_gwps()
    Zg, M, _, has_go = go_embedding(targets, go_annotations())
    k = has_go
    Y, Zg, M, ctrl, is_ess = Y[k], Zg[k], M[k], ctrl[k], is_ess[k]
    Z = np.c_[Zg, (ctrl - ctrl.mean()) / ctrl.std()]; F = np.c_[M, Z]; N = len(Y)
    res = {}
    for scen, (cs, ts) in {"natural_to_ess": (0.2, 1.0), "disjoint": (0.0, 1.0), "no_shift": (0.2, 0.2), "reverse": (0.75, 0.2)}.items():
        runs, gaps = [], []
        for s in range(seeds):
            rng = np.random.default_rng(s)
            r = rng.permutation(N); tr, pool = r[: int(0.3 * N)], r[int(0.3 * N):]
            f = fit_models(Z[tr], Y[tr], s)["ridge"]
            ca = draw(rng, pool, is_ess, cs, 1500); rest = np.setdiff1d(pool, ca)
            te_all = rng.permutation(draw(rng, rest, is_ess, ts, 600 + max(MS)))
            pil, te = te_all[: max(MS)], te_all[max(MS):]
            c2 = draw(rng, pool, is_ess, cs, 1500); t2 = draw(rng, pool, is_ess, ts, 1500)
            clf = LogisticRegression(C=1.0, max_iter=3000).fit(np.r_[F[c2], F[t2]], np.r_[np.zeros(len(c2)), np.ones(len(t2))])
            pr = lambda x: np.clip(clf.predict_proba(x)[:, 1], 1e-4, 1 - 1e-4)  # noqa: E731
            w = lambda x: pr(x) / (1 - pr(x))  # noqa: E731
            wc = w(F[ca]); norm = wc.mean(); wc = wc / norm
            sc, sp, st = rmse(f(Z[ca]), Y[ca]), rmse(f(Z[pil]), Y[pil]), rmse(f(Z[te]), Y[te])
            o = evaluate(alpha, sc, wc, Z[ca], sp, Z[pil], w(F[pil]) / norm, st, Z[te], w(F[te]) / norm)
            runs.append(o); gaps.append(0.9 - o["src"][0])
        agg = aggregate(runs)
        agg["ks_vs_gap_corr"] = {f"m{m}": float(np.corrcoef([r[f"ks_m{m}"][0] for r in runs], gaps)[0, 1]) for m in (10, 20, 50)}
        res[scen] = agg
        print(scen, json.dumps(summ(agg)), flush=True)
    return res


def context(seeds, alpha):
    Ys, targets = load_pair()
    Z, _, _, has_go = go_embedding(targets, go_annotations())
    Ys = {k: v[has_go] for k, v in Ys.items()}; Z = Z[has_go]; N = len(Z)
    res = {}
    for A, B in (("K562", "RPE1"), ("RPE1", "K562")):
        runs = []
        for s in range(seeds):
            rng = np.random.default_rng(s)
            r = rng.permutation(N)
            tr, ca, te_all = r[: int(0.4 * N)], r[int(0.4 * N): int(0.7 * N)], r[int(0.7 * N):]
            pil, te = te_all[: max(MS)], te_all[max(MS):]
            f = fit_models(Z[tr], Ys[A][tr], s)["ridge"]
            sc = rmse(f(Z[ca]), Ys[A][ca]); sp = rmse(f(Z[pil]), Ys[B][pil]); st = rmse(f(Z[te]), Ys[B][te])
            one = lambda n: np.ones(n)  # noqa: E731
            runs.append(evaluate(alpha, sc, one(len(sc)), Z[ca], sp, Z[pil], one(len(sp)), st, Z[te], one(len(st))))
        agg = aggregate(runs)
        res[f"{A}->{B}"] = agg
        print(f"{A}->{B}", json.dumps(summ(agg)), flush=True)
    return res


def summ(agg):
    keep = ["src"] + [f"{p}_m{m}" for m in (5, 10, 20, 50) for p in ("pilot", "P2", "P3g1", "P3g5", "P3g20")]
    return {k: [round(x, 3) for x in agg[k]] for k in keep if k in agg}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=30); ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--only", choices=("library", "context"))
    a = ap.parse_args()
    res = {}
    if a.only != "library":
        res["context"] = context(a.seeds, a.alpha)
    if a.only != "context":
        res["library"] = library(a.seeds, a.alpha)
    name = "pilot2_metrics.json" if a.only is None else f"pilot2_{a.only}_metrics.json"
    (OUT / name).write_text(json.dumps(res, indent=1))
    print("PILOT2_DONE", flush=True)


if __name__ == "__main__":
    main()
