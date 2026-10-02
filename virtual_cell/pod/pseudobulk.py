"""Pod-side: raw single-cell Perturb-seq h5ad -> z-normalised pseudobulk (Replogle-2022-style), chunked, backed mode.

Per cell: library-size normalise to the median UMI total. Per gem group (if present): z-score each gene against the
non-targeting control cells of that gem group (mean/sd; sd floor). Pseudobulk = mean z over a perturbation's cells.
Genes kept: mean normalised expression in controls >= MIN_MEAN. Output h5ad: obs_names '<i>_<GENE>_nadig', var_names =
the input var index, obs columns n_cells, control_expr (target gene's mean normalised control expression, NaN if not
measured).
usage: python pseudobulk.py IN.h5ad OUT.h5ad [--inspect]"""
from __future__ import annotations

import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

MIN_MEAN = 0.05
CHUNK = 20000


def pick(cols, cands):
    for c in cands:
        if c in cols:
            return c
    return None


def main():
    src, dst = sys.argv[1], sys.argv[2]
    a = ad.read_h5ad(src, backed="r")
    print(a, flush=True)
    print(a.obs.head().to_string(), flush=True)
    print(a.var.head().to_string(), flush=True)
    if "--inspect" in sys.argv:
        return
    obs = a.obs.copy()
    pcol = pick(obs.columns, ["gene", "target_gene", "gene_symbol", "perturbation", "guide_target", "target"])
    gcol = pick(obs.columns, ["gem_group", "gemgroup", "batch", "lane", "library"])
    print("perturbation col", pcol, "gem col", gcol, flush=True)
    pert = obs[pcol].astype(str).values
    is_ctrl = np.array(["non-targeting" in p.lower() or p.lower() in ("control", "nt", "non_targeting") for p in pert])
    gem = obs[gcol].astype(str).values if gcol else np.zeros(len(obs), dtype=str)
    par = [x for x in sys.argv[3:] if x.startswith("--parity=")]
    if par:  # same-line control: keep only gem groups of one parity (cells of the other half are treated as absent)
        want = int(par[0].split("=")[1])
        drop = np.array([int(g) % 2 != want for g in gem])
        pert = np.where(drop, "__dropped__", pert); is_ctrl = is_ctrl & ~drop
        print(f"parity {want}: dropping {drop.sum()} cells", flush=True)
    n, G = a.shape
    print(f"cells {n} genes {G} controls {is_ctrl.sum()} perts {len(set(pert[~is_ctrl]))}", flush=True)
    # pass 1: library sizes
    tot = np.zeros(n)
    for s in range(0, n, CHUNK):
        X = a.X[s:s + CHUNK]
        X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
        tot[s:s + CHUNK] = np.asarray(X.sum(1)).ravel()
    med = np.median(tot)
    # pass 2: control stats per gem group
    gems = np.unique(gem)
    s1 = {g: np.zeros(G) for g in gems}; s2 = {g: np.zeros(G) for g in gems}; cnt = {g: 0 for g in gems}
    for s in range(0, n, CHUNK):
        sel = np.where(is_ctrl[s:s + CHUNK])[0]
        if not len(sel):
            continue
        X = a.X[s:s + CHUNK]
        X = (X.tocsr() if sp.issparse(X) else sp.csr_matrix(X))[sel]
        Xn = sp.diags(med / np.maximum(tot[s + sel], 1)) @ X
        for g in np.unique(gem[s + sel]):
            m = gem[s + sel] == g
            Xg = Xn[m]
            s1[g] += np.asarray(Xg.sum(0)).ravel(); s2[g] += np.asarray(Xg.multiply(Xg).sum(0)).ravel(); cnt[g] += int(m.sum())
    mu = {g: s1[g] / max(cnt[g], 1) for g in gems}
    sd = {g: np.sqrt(np.maximum(s2[g] / max(cnt[g], 1) - mu[g] ** 2, 0)) for g in gems}
    allmu = sum(s1.values()) / max(sum(cnt.values()), 1)
    keep = allmu >= MIN_MEAN
    floor = np.median(np.concatenate([sd[g][keep] for g in gems])) * 0.1
    print(f"gem groups {len(gems)}; control cells per group min {min(cnt.values())}; genes kept {keep.sum()}", flush=True)
    # pass 3: accumulate z per perturbation
    perts = sorted(set(pert[~is_ctrl])); pidx = {p: i for i, p in enumerate(perts)}
    acc = np.zeros((len(perts), keep.sum())); nc = np.zeros(len(perts))
    kidx = np.where(keep)[0]
    for s in range(0, n, CHUNK):
        sel = np.where(~is_ctrl[s:s + CHUNK])[0]
        if not len(sel):
            continue
        X = a.X[s:s + CHUNK]
        X = (X.tocsr() if sp.issparse(X) else sp.csr_matrix(X))[sel][:, kidx]
        Xn = (sp.diags(med / np.maximum(tot[s + sel], 1)) @ X).toarray()
        for g in np.unique(gem[s + sel]):
            m = gem[s + sel] == g
            Zc = (Xn[m] - mu[g][kidx]) / np.maximum(sd[g][kidx], floor)
            rows = np.array([pidx[p] for p in pert[s + sel][m]])
            np.add.at(acc, rows, Zc); np.add.at(nc, rows, 1)
        print(f"  {min(s + CHUNK, n)}/{n}", flush=True)
    P = (acc / np.maximum(nc, 1)[:, None]).astype(np.float32)
    ok = np.array([p != "__dropped__" for p in perts])
    P, nc, perts = P[ok], nc[ok], [p for p, k in zip(perts, ok) if k]
    var = a.var.iloc[kidx].copy()
    sym_col = pick(var.columns, ["gene_name", "gene_symbols", "symbol", "gene_symbol"])
    syms = var[sym_col].astype(str).values if sym_col else var.index.astype(str).values
    sym2mu = dict(zip(syms, allmu[kidx]))
    ctrl_expr = np.array([sym2mu.get(p, np.nan) for p in perts])
    o = pd.DataFrame({"n_cells": nc, "control_expr": ctrl_expr}, index=[f"{i}_{p}_nadig" for i, p in enumerate(perts)])
    out = ad.AnnData(P, obs=o, var=var)
    out.var["gene_name"] = syms
    out.write_h5ad(dst)
    print("WROTE", dst, out.shape, flush=True)


if __name__ == "__main__":
    main()
