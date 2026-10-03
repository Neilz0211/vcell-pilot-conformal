"""Exploratory D1/D3 on Tahoe-100M (2026-10-03): predict a NEW cell line's drug responses from the other 49 lines.

Every Tahoe well pools all lines, so each unit (plate x drug x dose) has a response in (almost) every line. For a target
line t, the units are split at random into a pilot pool and a test set (unit-level sampling: pilots and test units are
exchangeable within t). Readout: the 3,000 pool genes, each scaled by its SD over the OTHER lines (no target data).
Line features for zero-pilot weighting: each line's baseline expression = median over its units of log1p(DESeq2
baseMean) on the pool genes (trt + ctrl mean; a proxy for the line's DMSO profile).
Predictors for t's response to a test unit u (m = number of pilots measured in t):
  other     mean over the other lines of their response to u                          (m = 0)
  simw      baseline-similarity-weighted mean over the other lines (top-k lines, k by leave-line-out CV)   (m = 0)
  pmean     pilot mean of t                                                         (m > 0)
  other_off other + pilot offset (the paper's mean-offset adapter)                   (m > 0)
  synth     "synthetic control": Y_t[u, g] ~ b_g + sum_l w_l Y_l[u, g], weights w by ridge on the m x G pilot entries,
            lambda by CV over pilots                                                (m > 0)
  synth_rr  synth + reduced-rank ridge correction of the residual from the other-lines mean   (m >= 25)
Metrics per (t, m): test MSE per gene; exact split conformal (fit on m/2, calibrate on m/2) radius and coverage for
pmean / other_off / synth; for m = 0, a cross-line borrowed radius: scores of the same zero-pilot predictor on the
OTHER target lines (their test units), pooled and normalised by each line's median score, times t's scale predicted
from its log median control-cell count (as tahoe_zeroshot.py; validity rests on exchangeable lines).
usage: python tahoe_xline.py /root/tahoe --seeds 5 --workers 30 --out tahoe_xline.json"""
import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np

AL = 0.1
D = {}


def q_conf(s, al=AL):
    n = len(s); k = math.ceil((1 - al) * (n + 1))
    return float("inf") if k > n else float(np.sort(s)[k - 1])


def rmse(P, Y):
    return np.sqrt(((P - Y) ** 2).mean(1))


def prepare(root):
    """Once, in the parent: line baselines and control-cell covariate, cached next to the compact arrays."""
    import os
    c = root + "/compact"
    if os.path.exists(c + "/xline_base.npy"): return
    lines = open(c + "/lines.txt").read().split("\n"); genes = open(c + "/genes.txt").read().split("\n")
    all_lines = open(root + "/lines.txt").read().split("\n"); all_genes = open(root + "/genes.txt").read().split("\n")
    gi = {g: i for i, g in enumerate(all_genes)}; gidx = np.array([gi[g] for g in genes])
    bm = np.load(root + "/basemean.npy", mmap_mode="r"); nall = np.load(root + "/ncells_trt.npy")
    nctrl = np.load(root + "/ncells_ctrl.npy"); base, ctrl = [], []
    for l in lines:
        i = all_lines.index(l); pres = np.where(nall[i] >= 0)[0]
        B = np.asarray(bm[i][pres], np.float32)[:, gidx]
        base.append(np.nanmedian(np.log1p(np.where(np.isfinite(B), B, np.nan)), 0))
        v = nctrl[i][nctrl[i] >= 0]; ctrl.append(np.median(v) if len(v) else np.nan)
        print("base", l, flush=True)
    np.save(c + "/xline_base.npy", np.array(base)); np.save(c + "/xline_ctrlcov.npy", np.log(np.array(ctrl, float)))


def init(root):
    c = root + "/compact"
    D.update(Y=np.load(c + "/Y.npy", mmap_mode="r"), nc=np.load(c + "/ncells.npy"),
             lines=open(c + "/lines.txt").read().split("\n"), base=np.load(c + "/xline_base.npy"),
             ctrlcov=np.load(c + "/xline_ctrlcov.npy"))


def ridge_w(X, y, lam):
    """X: n x L, y: n. Ridge without intercept (data already centred)."""
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


def fit_synth(Xp, Yp, lams=(1e1, 1e2, 1e3, 1e4)):
    """Xp: m x L x G (other lines' responses to the pilots), Yp: m x G. Returns predict(X: n x L x G)."""
    m = len(Yp)
    def fit(idx, lam):
        b = Yp[idx].mean(0); xb = Xp[idx].mean(0)                      # per-gene centring
        Xc = (Xp[idx] - xb).transpose(0, 2, 1).reshape(-1, Xp.shape[1]); yc = (Yp[idx] - b).reshape(-1)
        w = ridge_w(Xc, yc, lam)
        return lambda X: b + np.einsum("l,nlg->ng", w, X - xb)
    if m >= 10:
        folds = [np.arange(m)[i::5] for i in range(5)]; best, bl = np.inf, lams[0]
        for lam in lams:
            e = 0.0
            for f in folds:
                tr = np.setdiff1d(np.arange(m), f); e += ((fit(tr, lam)(Xp[f]) - Yp[f]) ** 2).sum()
            if e < best: best, bl = e, lam
    else:
        bl = 1e3
    return fit(np.arange(m), bl)


def rrr_fit(X, Y, k, lam):
    xm, ym = X.mean(0), Y.mean(0); Xc, Yc = X - xm, Y - ym
    A = np.linalg.solve(Xc @ Xc.T + lam * np.eye(len(Xc)), Yc); F = Xc @ (Xc.T @ A)
    Vt = np.linalg.svd(F, full_matrices=False)[2][:k]; Bk = (Xc.T @ A) @ Vt.T @ Vt
    return lambda X2: ym + (X2 - xm) @ Bk


def run(args):
    t, seed = args
    Y, nc, base = D["Y"], D["nc"], D["base"]; L = Y.shape[0]
    others = np.array([l for l in range(L) if l != t])
    units = np.where((nc[t] >= 30) & np.isfinite(Y[t]).all(1))[0]
    if len(units) < 400: return None
    rng = np.random.default_rng(seed); units = rng.permutation(units); test, pool = units[:300], units[300:]
    Xo = np.asarray(Y)[others][:, units]                                              # (L-1) x n x G
    Xo = np.asarray(Xo, np.float32); sd = (np.nanstd(Xo.reshape(-1, Xo.shape[2]), 0) + 1e-6).astype(np.float32)
    fill = np.nanmean(Xo, 0)                                              # other-lines mean, gaps filled with it
    Xo = np.where(np.isfinite(Xo), Xo, fill[None]) / sd; Yt = np.asarray(Y[t][units], np.float32) / sd
    pos = {u: i for i, u in enumerate(units)}; te = np.array([pos[u] for u in test]); po = np.array([pos[u] for u in pool])
    X = Xo.transpose(1, 0, 2)                                             # n x (L-1) x G
    other = X.mean(1)
    sim = np.array([np.corrcoef(base[t], base[l])[0, 1] for l in others]); order = np.argsort(-sim)
    out = {"t": int(t), "seed": seed, "sim_top": float(sim[order[0]])}
    out["mse_other"] = float(np.mean((other[te] - Yt[te]) ** 2))
    for k in (3, 5, 10, 20):
        P = X[:, order[:k]].mean(1); out[f"mse_simw{k}"] = float(np.mean((P[te] - Yt[te]) ** 2))
    out["s_other"] = rmse(other[te], Yt[te]).tolist()                     # for cross-line borrowing (zero pilots)
    out["s_simw5"] = rmse(X[te][:, order[:5]].mean(1), Yt[te]).tolist()
    for m in (5, 10, 25, 50, 100):
        if m > len(po): continue
        pil = po[:m]; h = m // 2
        pm = Yt[pil].mean(0)
        out[f"mse_pmean_{m}"] = float(np.mean((pm - Yt[te]) ** 2))
        off = (Yt[pil] - other[pil]).mean(0)
        out[f"mse_otheroff_{m}"] = float(np.mean((other[te] + off - Yt[te]) ** 2))
        f = fit_synth(X[pil], Yt[pil]); out[f"mse_synth_{m}"] = float(np.mean((f(X[te]) - Yt[te]) ** 2))
        if m >= 25:
            fs = fit_synth(X[pil], Yt[pil]); R = Yt[pil] - fs(X[pil])
            best = None
            for kk, lam in ((2, 100.0), (5, 100.0), (5, 1000.0), (10, 1000.0)):
                fr = rrr_fit(other[pil], R, kk, lam); e = np.mean((fs(X[te]) + fr(other[te]) - Yt[te]) ** 2)
                best = e if best is None else min(best, e)    # NOTE: test-selected; reported only as an optimistic bound
            out[f"mse_synthrr_opt_{m}"] = float(best)
        if m >= 10:
            for nm, mk in (("pmean", lambda idx: (lambda Xn: np.tile(Yt[idx].mean(0), (len(Xn), 1)))),
                           ("otheroff", lambda idx: (lambda Xn, o=(Yt[idx] - other[idx]).mean(0): Xn.mean(1) + o)),
                           ("synth", lambda idx: fit_synth(X[idx], Yt[idx]))):
                fh = mk(pil[:h]); q = q_conf(rmse(fh(X[pil[h:]]), Yt[pil[h:]]))
                out[f"q_{nm}_{m}"] = q; out[f"cov_{nm}_{m}"] = float((rmse(fh(X[te]), Yt[te]) <= q).mean())
    print(f"done t={t} seed={seed}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=30); ap.add_argument("--out", default="tahoe_xline.json")
    ap.add_argument("--lines", type=int, default=0)
    a = ap.parse_args(); prepare(a.root); init(a.root); L = D["Y"].shape[0]
    tl = range(L) if a.lines == 0 else range(a.lines)
    jobs = [(t, s) for t in tl for s in range(a.seeds)]
    with ProcessPoolExecutor(a.workers, initializer=init, initargs=(a.root,)) as ex:
        res = [r for r in ex.map(run, jobs) if r is not None]
    # zero-pilot cross-line borrowed calibration (leave-target-line-out pooling of normalised scores)
    by_seed = {}
    for r in res: by_seed.setdefault(r["seed"], []).append(r)
    cov = D["ctrlcov"]
    for s, rs in by_seed.items():
        for nm in ("other", "simw5"):
            med = {r["t"]: np.median(r[f"s_{nm}"]) for r in rs}
            for r in rs:
                rest = [x for x in rs if x["t"] != r["t"]]
                ts = np.array([x["t"] for x in rest]); y = np.log([med[x] for x in ts])
                A = np.c_[np.ones(len(ts)), cov[ts]]; coef = np.linalg.lstsq(A, y, rcond=None)[0]
                scale = math.exp(coef[0] + coef[1] * cov[r["t"]])
                pooled = np.concatenate([np.array(x[f"s_{nm}"]) / med[x["t"]] for x in rest])
                q = q_conf(pooled) * scale; r[f"q0_{nm}"] = q; r[f"cov0_{nm}"] = float((np.array(r[f"s_{nm}"]) <= q).mean())
    for r in res:
        r.pop("s_other", None); r.pop("s_simw5", None)
    json.dump({"lines": D["lines"], "results": res}, open(a.out, "w"))
    keys = sorted({k for r in res for k in r if k.startswith(("mse_", "cov", "q_", "q0_"))})
    for k in keys:
        v = np.array([r[k] for r in res if k in r]); print(f"{k:24s} mean {v.mean():.4f}  p5 {np.percentile(v, 5):.4f}  n {len(v)}")
    print("XLINE_DONE", flush=True)


if __name__ == "__main__":
    main()
