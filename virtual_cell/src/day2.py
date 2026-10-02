"""Day 2: a NATURAL library shift in the genome-wide K562 screen (Replogle 2022 gwps pseudobulk).

Libraries are defined by essentiality (membership of the target gene in the essential-library screen; essentiality
is known a priori from DepMap-type data, so it is a legitimate pre-experiment covariate). Scenarios:
  mix   calibration = natural genome-wide composition (~25 % essential); screening = essential share 0.25/0.5/0.75/1.0
  rev   calibration = essential-enriched, screening = genome-wide composition
  disj  calibration = non-essential only, screening = essential only (no overlap: should abstain everywhere)
Weights: oracle (on the essentiality covariate, known library design), and estimated (logistic regression on
z + GO memberships + baseline expression, trained on independent library draws). Models: mean, ridge(z), kNN(z).
z = 64-d GO SVD + log baseline expression of the target in control cells (control_expr). Knockdown efficiency is
NOT used (only known after the experiment).
usage: uv run python src/day2.py [--seeds 30]"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import anndata as ad
import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import DATA, go_annotations, go_embedding, target_gene  # noqa: E402
from day1 import ess, fit_models, rmse, split_cp_q, weighted_quantile_cp  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"


def load_gwps(n_genes=1000):
    a = ad.read_h5ad(DATA / "K562_gwps_normalized_bulk_01.h5ad")
    names = np.asarray(a.obs_names)
    keep = np.array([not re.search(r"non-targeting|control", n, re.I) for n in names])
    X = np.asarray(a.X, dtype=np.float64)[keep]
    X = X[:, np.isfinite(X).all(0)]  # 73 readout genes carry ±inf (zero control variance in z-normalisation)
    targets = np.array([target_gene(n) for n in names[keep]])
    raw = np.asarray(a.obs["control_expr"], dtype=np.float64)[keep]
    ctrl = np.log1p(np.nan_to_num(raw, nan=0.0))  # missing = target not detected in controls -> 0
    top = np.argsort(X.var(0))[::-1][:n_genes]
    Y = X[:, top] / (X[:, top].std(0) + 1e-8)
    ess_lib = set(target_gene(n) for n in ad.read_h5ad(DATA / "K562_essential_normalized_bulk_01.h5ad", backed="r").obs_names)
    is_ess = np.array([t in ess_lib for t in targets])
    return Y, targets, ctrl, is_ess


def draw(rng, pool, is_ess, share, size):
    """draw `size` perturbations from pool with an essential share of `share` (without replacement)."""
    e, ne = pool[is_ess[pool]], pool[~is_ess[pool]]
    k = int(round(share * size))
    k = min(k, len(e)); m = min(size - k, len(ne))
    return np.r_[rng.choice(e, k, replace=False), rng.choice(ne, m, replace=False)]


def run_case(Y, Z, F, is_ess, cal_share, test_share, seeds, alpha, n_cal=1500, n_test=600):
    out = {k: [] for k in ("unw", "orc", "est", "ess_orc", "ess_est", "abst_orc", "abst_est", "q_unw", "q_orc", "err_e", "err_ne")}
    N = len(Y)
    for s in range(seeds):
        rng = np.random.default_rng(s)
        r = rng.permutation(N)
        tr, pool = r[: int(0.3 * N)], r[int(0.3 * N):]
        f = fit_models(Z[tr], Y[tr], s)["ridge"]
        ca = draw(rng, pool, is_ess, cal_share, n_cal)
        rest = np.setdiff1d(pool, ca)
        te = draw(rng, rest, is_ess, test_share, n_test)
        sc, st = rmse(f(Z[ca]), Y[ca]), rmse(f(Z[te]), Y[te])
        q = split_cp_q(sc, alpha)
        out["unw"].append(float((st <= q).mean())); out["q_unw"].append(float(q))
        out["err_e"].append(float(np.median(st[is_ess[te]])) if is_ess[te].any() else np.nan)
        out["err_ne"].append(float(np.median(st[~is_ess[te]])) if (~is_ess[te]).any() else np.nan)
        # oracle weights from the library design: ratio of essential shares
        def w_or(e):
            num = np.where(e, test_share, 1 - test_share); den = np.where(e, cal_share, 1 - cal_share)
            return np.where(den > 0, num / np.maximum(den, 1e-12), np.inf)
        wc = w_or(is_ess[ca]); covs, abst, qs = [], [], []
        for i in range(len(te)):
            wt = float(w_or(is_ess[te[i:i + 1]])[0])
            qq, ab = (np.inf, True) if not np.isfinite(wt) else weighted_quantile_cp(sc, wc, wt, alpha)
            covs.append(st[i] <= qq); abst.append(ab); qs.append(qq)
        out["orc"].append(float(np.mean(covs))); out["abst_orc"].append(float(np.mean(abst)))
        out["ess_orc"].append(float(ess(wc))); out["q_orc"].append(float(np.nanmean([x for x in qs if np.isfinite(x)])) if any(np.isfinite(qs)) else np.nan)
        # estimated weights: classifier on features F, trained on fresh library draws
        c2 = draw(rng, pool, is_ess, cal_share, n_cal); t2 = draw(rng, pool, is_ess, test_share, n_cal)
        clf = LogisticRegression(C=1.0, max_iter=3000).fit(np.r_[F[c2], F[t2]], np.r_[np.zeros(len(c2)), np.ones(len(t2))])
        pr = lambda x: np.clip(clf.predict_proba(x)[:, 1], 1e-4, 1 - 1e-4)  # noqa: E731
        we = (pr(F[ca]) / (1 - pr(F[ca]))) * (len(c2) / len(t2)); wt_e = (pr(F[te]) / (1 - pr(F[te]))) * (len(c2) / len(t2))
        covs, abst = [], []
        for i in range(len(te)):
            qq, ab = weighted_quantile_cp(sc, we, float(wt_e[i]), alpha)
            covs.append(st[i] <= qq); abst.append(ab)
        out["est"].append(float(np.mean(covs))); out["abst_est"].append(float(np.mean(abst))); out["ess_est"].append(float(ess(we)))
    return {k: float(np.nanmean(v)) if len(v) else None for k, v in out.items()} | {"unw_sd": float(np.std(out["unw"]))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=30); ap.add_argument("--alpha", type=float, default=0.1)
    a = ap.parse_args()
    Y, targets, ctrl, is_ess = load_gwps()
    Zg, M, vocab, has_go = go_embedding(targets, go_annotations())
    k = has_go
    Y, Zg, M, ctrl, is_ess, targets = Y[k], Zg[k], M[k], ctrl[k], is_ess[k], targets[k]
    Z = np.c_[Zg, (ctrl - ctrl.mean()) / ctrl.std()]
    F = np.c_[M, Z]
    print(f"gwps perturbations with GO: {len(Y)}; essential-library targets {is_ess.sum()} ({is_ess.mean():.2f})", flush=True)
    nat = float(is_ess.mean())
    cases = {"mix": [(nat, s) for s in (nat, 0.5, 0.75, 1.0)], "rev": [(0.75, nat)], "disj": [(0.0, 1.0)]}
    res = {"N": int(len(Y)), "essential_share": nat, "alpha": a.alpha, "cases": {}}
    for name, pairs in cases.items():
        for cs, ts in pairs:
            r = run_case(Y, Z, F, is_ess, cs, ts, a.seeds, a.alpha)
            key = f"{name}_cal{cs:.2f}_test{ts:.2f}"
            res["cases"][key] = r
            print(key, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    # evaluation lens on gwps: random split, model vs mean set radius
    lens = {m: [] for m in ("mean", "ridge", "knn")}
    N = len(Y)
    for s in range(a.seeds):
        r = np.random.default_rng(500 + s).permutation(N)
        tr, ca = r[: N // 2], r[N // 2:]
        for m, f in fit_models(Z[tr], Y[tr], s).items():
            lens[m].append(split_cp_q(rmse(f(Z[ca]), Y[ca]), a.alpha))
    res["lens"] = {m: float(np.mean(v)) for m, v in lens.items()}
    print("lens", res["lens"], flush=True)
    (OUT / "day2_metrics.json").write_text(json.dumps(res, indent=1))
    print("DAY2_DONE", flush=True)


if __name__ == "__main__":
    main()
