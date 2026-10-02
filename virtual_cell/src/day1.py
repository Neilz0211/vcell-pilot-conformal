"""Day-1 go/no-go (virtual_cell/proposal.md §6.3) on Replogle K562 essential pseudobulk.

Q1 perturbation-level split CP on random perturbation hold-out: coverage vs nominal, set size.
Q2 library shift along GO categories: calibration library under-samples a category that the screening library
   over-samples; unweighted CP vs weighted CP (oracle weights / classifier-estimated weights on z), ESS, abstention.
Q3 score-error AUC of cheap uncertainty proxies (z-space distance to training perturbations, bootstrap spread).

Score s(p) = RMSE over the G standardised readout genes between prediction and observed pseudobulk response; the
prediction set is the ball {Δ : RMSE(Δ̂, Δ) ≤ q̂}, so q̂ is the set size (in per-gene SD units).
usage: uv run python src/day1.py [--seeds 50] [--alpha 0.1]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import go_annotations, go_embedding, load_responses  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def weighted_quantile_cp(scores, w_cal, w_test_point, alpha):
    """Tibshirani et al. 2019: quantile of sum_i p_i δ_{S_i} + p_{n+1} δ_{+inf}; returns (q̂, abstain flag)."""
    tot = w_cal.sum() + w_test_point
    p = w_cal / tot
    p_inf = w_test_point / tot
    if p_inf >= alpha:  # the point mass at +inf alone exceeds alpha -> infinite quantile -> abstain
        return np.inf, True
    o = np.argsort(scores)
    c = np.cumsum(p[o])
    k = np.searchsorted(c, 1 - alpha)
    return (scores[o][k] if k < len(o) else np.inf), k >= len(o)


def split_cp_q(scores, alpha):
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha))) - 1
    return np.sort(scores)[min(k, n - 1)] if k < n else np.inf


def fit_models(Ztr, Ytr, seed):
    mean = Ytr.mean(0)
    ridge = Ridge(alpha=10.0).fit(Ztr, Ytr)
    nn = NearestNeighbors(n_neighbors=min(10, len(Ztr))).fit(Ztr)

    def knn(Z):
        _, idx = nn.kneighbors(Z)
        return Ytr[idx].mean(1)
    return {"mean": lambda Z: np.tile(mean, (len(Z), 1)), "ridge": ridge.predict, "knn": knn}


def rmse(a, b):
    return np.sqrt(((a - b) ** 2).mean(1))


def ess(w):
    return w.sum() ** 2 / (w ** 2).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=0.1)
    a = ap.parse_args()
    D = load_responses()
    Y, targets = D["Y"], D["targets"]
    Z, M, vocab, has_go = go_embedding(targets, go_annotations())
    keep = has_go
    Y, Z, M, targets = Y[keep], Z[keep], M[keep], targets[keep]
    N = len(Y)
    print(f"perturbations with GO: {N}/{len(has_go)}; readout genes {Y.shape[1]}; GO terms {len(vocab)}", flush=True)
    res = {"N": int(N), "alpha": a.alpha}

    # ---------- Q1 random hold-out ----------
    q1 = {m: {"cov": [], "q": []} for m in ("mean", "ridge", "knn")}
    for s in range(a.seeds):
        r = np.random.default_rng(s).permutation(N)
        tr, ca, te = r[: N // 2], r[N // 2: 3 * N // 4], r[3 * N // 4:]
        models = fit_models(Z[tr], Y[tr], s)
        for m, f in models.items():
            sc, st = rmse(f(Z[ca]), Y[ca]), rmse(f(Z[te]), Y[te])
            q = split_cp_q(sc, a.alpha)
            q1[m]["cov"].append(float((st <= q).mean())); q1[m]["q"].append(float(q))
    res["Q1"] = {m: {"coverage_mean": float(np.mean(v["cov"])), "coverage_sd": float(np.std(v["cov"])),
                     "q_mean": float(np.mean(v["q"]))} for m, v in q1.items()}
    print("Q1", json.dumps(res["Q1"], indent=1), flush=True)

    # ---------- Q2 library shift along GO categories ----------
    # categories = the most populated GO terms among the perturbation targets (broad but not universal)
    counts = M.sum(0)
    cand = [j for j in np.argsort(counts)[::-1] if 0.08 * N <= counts[j] <= 0.35 * N][:6]
    res["Q2"] = {}
    for j in cand:
        cat = M[:, j] > 0
        rows = []
        for beta in (0.0, 1.0, 2.0, 3.0):
            out = {"unw": [], "w_oracle": [], "w_est": [], "ess_oracle": [], "ess_est": [], "abst_est": [], "q_unw": [], "q_w": []}
            for s in range(a.seeds):
                rng = np.random.default_rng(1000 + s)
                r = rng.permutation(N)
                tr, pool = r[: N // 2], r[N // 2:]
                # calibration library favours non-category, screening library favours category (odds e^{±beta/2})
                pc = np.where(cat[pool], np.exp(-beta / 2), np.exp(beta / 2)); pc /= pc.sum()
                pt = np.where(cat[pool], np.exp(beta / 2), np.exp(-beta / 2)); pt /= pt.sum()
                half = len(pool) // 2
                ca = rng.choice(pool, size=half, replace=False, p=pc)
                rest = np.setdiff1d(pool, ca)
                pt_r = np.where(cat[rest], np.exp(beta / 2), np.exp(-beta / 2)); pt_r /= pt_r.sum()
                te = rng.choice(rest, size=min(len(rest), half // 2), replace=False, p=pt_r)
                f = fit_models(Z[tr], Y[tr], s)["ridge"]
                sc, st = rmse(f(Z[ca]), Y[ca]), rmse(f(Z[te]), Y[te])
                q = split_cp_q(sc, a.alpha)
                out["unw"].append(float((st <= q).mean())); out["q_unw"].append(float(q))
                # oracle density ratio between libraries depends only on the category
                w_or = lambda c: np.where(c, np.exp(beta), np.exp(-beta))  # noqa: E731  (∝ odds ratio squared)
                wc = w_or(cat[ca]); cov_or = []
                for i in range(len(te)):
                    qq, _ = weighted_quantile_cp(sc, wc, float(w_or(cat[te[i:i + 1]])[0]), a.alpha)
                    cov_or.append(st[i] <= qq)
                out["w_oracle"].append(float(np.mean(cov_or))); out["ess_oracle"].append(float(ess(wc)))
                # estimated weights: classifier cal-vs-screening on z, trained on an independent draw of both libraries
                ca2 = rng.choice(pool, size=half, replace=False, p=pc)
                te2 = rng.choice(pool, size=half, replace=False, p=pt)
                clf = LogisticRegression(C=1.0, max_iter=2000).fit(np.r_[Z[ca2], Z[te2]], np.r_[np.zeros(half), np.ones(half)])
                pr = lambda zz: np.clip(clf.predict_proba(zz)[:, 1], 1e-3, 1 - 1e-3)  # noqa: E731
                w_est = lambda zz: pr(zz) / (1 - pr(zz))  # noqa: E731
                wce = w_est(Z[ca]); wte = w_est(Z[te]); cov_e, ab = [], []
                for i in range(len(te)):
                    qq, abst = weighted_quantile_cp(sc, wce, float(wte[i]), a.alpha)
                    cov_e.append(st[i] <= qq); ab.append(abst)
                    out["q_w"].append(float(qq) if np.isfinite(qq) else np.nan)
                out["w_est"].append(float(np.mean(cov_e))); out["ess_est"].append(float(ess(wce))); out["abst_est"].append(float(np.mean(ab)))
                # second estimator: logistic regression on the raw GO-term memberships (what defines the libraries)
                clf2 = LogisticRegression(C=1.0, max_iter=3000).fit(np.r_[M[ca2], M[te2]], np.r_[np.zeros(half), np.ones(half)])
                pr2 = lambda mm: np.clip(clf2.predict_proba(mm)[:, 1], 1e-3, 1 - 1e-3)  # noqa: E731
                wce2 = pr2(M[ca]) / (1 - pr2(M[ca])); wte2 = pr2(M[te]) / (1 - pr2(M[te])); cov2, ab2 = [], []
                for i in range(len(te)):
                    qq, abst = weighted_quantile_cp(sc, wce2, float(wte2[i]), a.alpha)
                    cov2.append(st[i] <= qq); ab2.append(abst)
                out.setdefault("w_est_go", []).append(float(np.mean(cov2)))
                out.setdefault("ess_est_go", []).append(float(ess(wce2)))
                out.setdefault("abst_est_go", []).append(float(np.mean(ab2)))
            row = {"beta": beta, **{k: float(np.nanmean(v)) for k, v in out.items()},
                   "unw_cov_sd": float(np.std(out["unw"])),
                   "cat_err_ratio": None}
            rows.append(row)
        # how different are errors in vs out of the category (drives undercoverage)?
        f = fit_models(Z[: N // 2], Y[: N // 2], 0)["ridge"]
        e = rmse(f(Z), Y)
        res["Q2"][vocab[j]] = {"n_cat": int(cat.sum()), "err_in": float(np.median(e[cat])), "err_out": float(np.median(e[~cat])), "rows": rows}
        print(vocab[j], json.dumps(res["Q2"][vocab[j]]), flush=True)

    # ---------- Q3 score-error AUC ----------
    aucs = {"knn_dist": [], "boot_sd": []}
    for s in range(min(a.seeds, 20)):
        r = np.random.default_rng(2000 + s).permutation(N)
        tr, te = r[: N // 2], r[N // 2:]
        f = fit_models(Z[tr], Y[tr], s)["ridge"]
        err = rmse(f(Z[te]), Y[te])
        hi = err >= np.quantile(err, 0.75)
        d, _ = NearestNeighbors(n_neighbors=5).fit(Z[tr]).kneighbors(Z[te])
        aucs["knn_dist"].append(roc_auc_score(hi, d.mean(1)))
        rng = np.random.default_rng(s)
        preds = [Ridge(alpha=10.0).fit(Z[tr][b], Y[tr][b]).predict(Z[te]) for b in (rng.integers(0, len(tr), len(tr)) for _ in range(10))]
        aucs["boot_sd"].append(roc_auc_score(hi, np.stack(preds).std(0).mean(1)))
    res["Q3"] = {k: {"auc_mean": float(np.mean(v)), "auc_sd": float(np.std(v))} for k, v in aucs.items()}
    print("Q3", json.dumps(res["Q3"]), flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "day1_metrics.json").write_text(json.dumps(res, indent=1))
    print("DAY1_DONE", flush=True)


if __name__ == "__main__":
    main()
