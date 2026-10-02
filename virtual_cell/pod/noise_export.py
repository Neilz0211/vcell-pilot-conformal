"""Pod-side: export per-perturbation measurement-noise vectors from split-half pseudobulk, on the paper's 3,000 genes.

For a perturbation with n0 and n1 cells in the two halves, h0 - h1 has noise variance sigma^2 (1/n0 + 1/n1), and the full
measurement's noise has variance sigma^2 / (n0 + n1); so eps_full ~ (h0 - h1) * sqrt(n0 n1) / (n0 + n1) is an unbiased
noise draw at the scale of the full pseudobulk (up to half-specific gem-group effects).
usage: python noise_export.py genes.txt OUT.npz"""
import sys

import anndata as ad
import numpy as np

genes = open(sys.argv[1]).read().split()
out = {}
def put(line, perts, g, H0, H1, n0, n1):
    gi = {x: i for i, x in enumerate(g)}; cols = [gi.get(x, -1) for x in genes]
    ok = (n0 >= 5) & (n1 >= 5)
    H0, H1, n0, n1, perts = H0[ok], H1[ok], n0[ok], n1[ok], np.asarray(perts)[ok]
    E = np.full((len(perts), len(genes)), np.nan, np.float32)
    have = np.array([c >= 0 for c in cols])
    E[:, have] = ((H0 - H1)[:, [c for c in cols if c >= 0]] * (np.sqrt(n0 * n1) / (n0 + n1))[:, None]).astype(np.float32)
    out[f"{line}_perts"] = perts; out[f"{line}_eps"] = E.astype(np.float16); out[f"{line}_n"] = (n0 + n1)
    print(line, E.shape, "genes found", have.sum(), "median cells", np.median(n0 + n1), flush=True)

# all four lines go through pseudobulk.py --parity (Arc's K562/RPE1 single-cell files hold normalised counts, not z-scores)
for line in ("ess", "rpe1", "hepg2", "jurkat"):
    a, b = ad.read_h5ad(f"{line}_par0.h5ad"), ad.read_h5ad(f"{line}_par1.h5ad")
    pa = [n.split("_")[1] for n in a.obs_names]; pb = [n.split("_")[1] for n in b.obs_names]
    common = sorted(set(pa) & set(pb)); ia = {p: i for i, p in enumerate(pa)}; ib = {p: i for i, p in enumerate(pb)}
    gb = {str(x): j for j, x in enumerate(b.var_names)}           # halves keep slightly different gene sets
    g = [str(x) for x in a.var_names if str(x) in gb]; ja = [j for j, x in enumerate(a.var_names) if str(x) in gb]
    jb = [gb[x] for x in g]
    A = np.asarray(a.X)[[ia[p] for p in common]][:, ja]; B = np.asarray(b.X)[[ib[p] for p in common]][:, jb]
    put(line, common, g, A, B, a.obs["n_cells"].values[[ia[p] for p in common]].astype(float),
        b.obs["n_cells"].values[[ib[p] for p in common]].astype(float))
np.savez_compressed(sys.argv[2], **out)
print("WROTE", sys.argv[2], flush=True)
