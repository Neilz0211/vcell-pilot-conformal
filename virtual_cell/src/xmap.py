"""Exploratory D1 (2026-10-03): can m target pilots LEARN a cross-line map of perturbation responses?

Scenario as transfer_measured.py: perturbations were screened in source line A; in target line B only m random pilots
are measured. Predict every other perturbation's B response from what is known about it in A (its measured A response,
GO features) plus the m pilots. Leak-free design of ceiling.py (genes / scales from the source train split, fixed
280-target test set, pilots = first m of the calibration pool, only perturbations with split-half noise so that the
noise energy of the test set is known).

Predictors (all fitted on the m pilots only; hyper-parameters chosen by 5-fold CV inside the pilots):
  mean      pilot mean of B (adapted predict-the-mean; the paper's reference)
  ridge     ridge(GO-SVD) fitted in A + pilot offset (the paper's adapted ridge)
  measc     c * Y_A + b, one scalar c (paper, Section 5.3)
  gene      per-gene slope c_g * Y_A[:, g] + b_g, shrunk toward the global c (ridge on c_g - c)
  pcmap     map in source-PC space: B-scores on K source PCs = ridge(A-scores on K PCs) + per-gene mean; K, lambda by CV
  rrr       reduced-rank ridge from Y_A (all genes) to Y_B, rank k, lambda by CV (dual form, m << G)
  knn       pilot mean + similarity-weighted pilot deviations, weights from the correlation of A responses (kernel map)
  combo     rrr on [Y_A, ridge-GO prediction] (both sources)
Metrics on the test set: energy (MSE per gene) relative to `mean`; gap closed = (MSE_mean - MSE_x) / (MSE_mean - noise)
with noise = mean split-half noise energy of the test perturbations; exact split-conformal radius (fit on m/2 pilots,
calibrate on the other m/2) and its coverage.
  pmap      PerturbMap-style baseline (Cui et al., arXiv 2607.28090): blend of a full-rank ridge expert and ridge-GO + offset
usage: uv run python src/xmap.py [--seeds 20] [--pairs all|smoke] [--ms 50,100,200] [--only A,B]"""
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

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "results"
LAMS = (0.1, 1.0, 10.0, 100.0, 1000.0)


def kfold(m, k=5, seed=0):
    idx = np.random.default_rng(seed).permutation(m); return [idx[i::k] for i in range(k)]


# ---------- predictors: fit(XA_pil, YB_pil, aux_pil, hp) -> predict(XA_new, aux_new) ----------
def p_mean(XA, Y, aux, hp):
    yb = Y.mean(0); return lambda XA2, aux2: np.tile(yb, (len(XA2), 1))


def p_ridge(XA, Y, aux, hp):          # aux = ridge-GO prediction made in A
    b = (Y - aux).mean(0); return lambda XA2, aux2: aux2 + b


def p_measc(XA, Y, aux, hp):
    xm, ym = XA.mean(0), Y.mean(0); xc, yc = XA - xm, Y - ym
    c = (xc * yc).sum() / max((xc * xc).sum(), 1e-12); return lambda XA2, aux2: ym + c * (XA2 - xm)


def p_gene(XA, Y, aux, hp):
    lam = hp; xm, ym = XA.mean(0), Y.mean(0); xc, yc = XA - xm, Y - ym
    c0 = (xc * yc).sum() / max((xc * xc).sum(), 1e-12)
    cg = ((xc * yc).sum(0) + lam * c0) / ((xc * xc).sum(0) + lam)
    return lambda XA2, aux2: ym + cg * (XA2 - xm)


def make_pcmap(V):                    # V: G x Kmax source-PC loadings (from the source train split)
    def p_pcmap(XA, Y, aux, hp):
        K, lam = hp; W = V[:, :K]; xm, ym = XA.mean(0), Y.mean(0)
        SA, SB = (XA - xm) @ W, (Y - ym) @ W
        M = Ridge(alpha=lam, fit_intercept=False).fit(SA, SB).coef_.T          # K x K
        return lambda XA2, aux2: ym + ((XA2 - xm) @ W @ M) @ W.T
    return p_pcmap


def _rrr(X, Y, k, lam):
    xm, ym = X.mean(0), Y.mean(0); Xc, Yc = X - xm, Y - ym
    A = np.linalg.solve(Xc @ Xc.T + lam * np.eye(len(Xc)), Yc)               # dual ridge: B = Xc' A
    F = Xc @ (Xc.T @ A)                                                       # fitted values (m x G)
    Vt = np.linalg.svd(F, full_matrices=False)[2][:k]                         # rank-k projection of the outputs
    Bk = (Xc.T @ A) @ Vt.T @ Vt
    return lambda X2: ym + (X2 - xm) @ Bk


def p_rrr(XA, Y, aux, hp):
    k, lam = hp; f = _rrr(XA, Y, k, lam); return lambda XA2, aux2: f(XA2)


def p_combo(XA, Y, aux, hp):
    k, lam = hp; f = _rrr(np.hstack([XA, aux]), Y, k, lam); return lambda XA2, aux2: f(np.hstack([XA2, aux2]))


def p_knn(XA, Y, aux, hp):
    tau = hp; ym = Y.mean(0); D = Y - ym
    def pred(XA2, aux2):
        a = XA2 - XA2.mean(1, keepdims=True); b = XA - XA.mean(1, keepdims=True)
        C = (a @ b.T) / (np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=1)[None] + 1e-12)
        W = np.exp(C / tau); W /= W.sum(1, keepdims=True)
        return ym + W @ D - (W.mean(0) @ D) * 0          # weights centred implicitly: sum_j W_ij (Y_j - ym)
    return pred


def p_pmap(XA, Y, aux, hp):
    """PerturbMap-style (Cui et al. 2026): full-rank ridge expert from the source response, blended with a
    recipient-local base (here ridge-GO + pilot offset) by weight w; lambda and w chosen by CV in the pilots."""
    lam, w = hp; xm, ym = XA.mean(0), Y.mean(0); Xc, Yc = XA - xm, Y - ym
    A = np.linalg.solve(Xc @ Xc.T + lam * np.eye(len(Xc)), Yc); Bm = Xc.T @ A; b = (Y - aux).mean(0)
    return lambda XA2, aux2: w * (ym + (XA2 - xm) @ Bm) + (1 - w) * (aux2 + b)


def cv_pick(fit, grid, XA, Y, aux):
    if len(grid) == 1: return grid[0]
    best, bh = np.inf, grid[0]
    for hp in grid:
        err = 0.0
        for f in kfold(len(Y)):
            tr = np.setdiff1d(np.arange(len(Y)), f); pr = fit(XA[tr], Y[tr], aux[tr], hp)
            err += ((pr(XA[f], aux[f]) - Y[f]) ** 2).sum()
        if err < best: best, bh = err, hp
    return bh


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=20); ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--pairs", default="all"); ap.add_argument("--ms", default="50,100,200"); ap.add_argument("--only", default="")
    a = ap.parse_args(); al = a.alpha; MS = tuple(int(x) for x in a.ms.split(","))
    compact = dict(np.load(ROOT / "data" / "nadig_compact.npz", allow_pickle=True))
    NZ = np.load(ROOT / "data" / "noise.npz", allow_pickle=True)
    g3 = {str(g): i for i, g in enumerate(compact["genes"])}
    lines = ("ess", "rpe1", "hepg2", "jurkat")
    pairs = [(A, B) for A in lines for B in lines if A != B]
    if a.pairs == "smoke": pairs = [("ess", "rpe1"), ("hepg2", "jurkat")]
    if a.only: pairs = [tuple(a.only.split(","))]
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
            usable = np.isfinite(eps).mean(1) >= 0.95
            gm = np.isfinite(eps[usable]).mean(0) >= 0.99
            YA, YB, eps = YA[:, gm], YB[:, gm], np.where(np.isnan(eps[:, gm]), 0.0, eps[:, gm])
            cal, test = cal[usable[cal]], test[usable[test]]
            noise = float(np.mean(eps[test] ** 2)); put("noise", noise); put("ncal", len(cal))
            Pr = Ridge(alpha=10.0).fit(Z[tr], YA[tr]).predict(Z)
            V = np.linalg.svd(YA[tr] - YA[tr].mean(0), full_matrices=False)[2][:50].T
            preds = {"mean": (p_mean, [None]), "ridge": (p_ridge, [None]), "measc": (p_measc, [None]),
                     "gene": (p_gene, list(LAMS)), "pcmap": (make_pcmap(V), [(K, l) for K in (5, 10, 20, 50) for l in (1.0, 10.0, 100.0)]),
                     "rrr": (p_rrr, [(k, l) for k in (2, 5, 10, 20) for l in (10.0, 100.0, 1000.0)]),
                     "knn": (p_knn, [0.05, 0.1, 0.2, 0.5]),
                     "pmap": (p_pmap, [(l, w) for l in (100.0, 1000.0, 1e4) for w in (0.25, 0.5, 0.75, 1.0)]),
                     "combo": (p_combo, [(k, l) for k in (2, 5, 10, 20) for l in (10.0, 100.0, 1000.0)])}
            for m in MS:
                if m > len(cal): continue
                pil = cal[:m]; h = m // 2
                for nm, (fit, grid) in preds.items():
                    hp = cv_pick(fit, grid, YA[pil], YB[pil], Pr[pil])
                    f = fit(YA[pil], YB[pil], Pr[pil], hp); P = f(YA[test], Pr[test])
                    put(f"{nm}_mse_{m}", np.mean((P - YB[test]) ** 2))
                    # exact split conformal: fit on first half (hp re-picked inside it), calibrate on second half
                    hp2 = cv_pick(fit, grid, YA[pil[:h]], YB[pil[:h]], Pr[pil[:h]])
                    f2 = fit(YA[pil[:h]], YB[pil[:h]], Pr[pil[:h]], hp2)
                    q = split_cp_q(rmse(f2(YA[pil[h:]], Pr[pil[h:]]), YB[pil[h:]]), al)
                    put(f"{nm}_q_{m}", q); put(f"{nm}_cov_{m}", (rmse(f2(YA[test], Pr[test]), YB[test]) <= q).mean())
        x = {k: float(np.mean(v)) for k, v in raw.items()}
        for m in MS:
            if f"mean_mse_{m}" not in x: continue
            for nm in ("ridge", "measc", "gene", "pcmap", "rrr", "knn", "combo", "pmap"):
                x[f"{nm}_gap_{m}"] = float(np.mean([(mm - xx) / (mm - nn) for mm, xx, nn in
                                                    zip(raw[f"mean_mse_{m}"], raw[f"{nm}_mse_{m}"], raw["noise"])]))
                x[f"{nm}_dq_{m}"] = float(np.mean([qq / q0 - 1 for qq, q0 in zip(raw[f"{nm}_q_{m}"], raw[f"mean_q_{m}"])]))
        res[f"{A}->{B}"] = {"summary": x, "raw": raw}
        for m in MS:
            if f"mean_mse_{m}" not in x: continue
            print(f"{A}->{B} m={m} noise/mse_mean {x['noise'] / x[f'mean_mse_{m}']:.2f} | gap closed "
                  + " ".join(f"{nm} {100 * x[f'{nm}_gap_{m}']:+.0f}%" for nm in ("ridge", "measc", "gene", "pcmap", "rrr", "knn", "combo", "pmap"))
                  + " | split radius vs mean " + " ".join(f"{nm} {100 * x[f'{nm}_dq_{m}']:+.1f}%" for nm in ("ridge", "measc", "gene", "pcmap", "rrr", "combo", "pmap"))
                  + f" | cov mean/rrr {x[f'mean_cov_{m}']:.3f}/{x[f'rrr_cov_{m}']:.3f}", flush=True)
    tag = "smoke" if a.pairs == "smoke" else f"s{a.seeds}"
    if a.only:
        (OUT / "xmap_pairs").mkdir(exist_ok=True); (OUT / "xmap_pairs" / f"{a.only.replace(',', '-')}_{tag}.json").write_text(json.dumps(res))
    else:
        (OUT / f"xmap_{tag}_metrics.json").write_text(json.dumps(res))
    print("XMAP_DONE", flush=True)


if __name__ == "__main__":
    main()
