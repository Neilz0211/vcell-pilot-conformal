"""Pod-side: split-half pseudobulk from Replogle's gem-group-normalised single cells (K562 essential, RPE1).

Cells are split by the parity of their gem group; each half's pseudobulk is the mean normalised expression over a
perturbation's cells in that half. The two halves share the published per-gem-group normalisation (each gem group is in
one half only, so the halves are disjoint in cells and in control-normalisation groups).
usage: python sc_halves.py IN.h5ad OUT.npz"""
import sys

import anndata as ad
import numpy as np
import scipy.sparse as sp

src, dst = sys.argv[1], sys.argv[2]
a = ad.read_h5ad(src, backed="r")
print(a, flush=True)
obs = a.obs
pc = "gene" if "gene" in obs else [c for c in obs.columns if "gene" in c.lower()][0]
gc = "gem_group" if "gem_group" in obs else [c for c in obs.columns if "gem" in c.lower()][0]
pert = obs[pc].astype(str).values; gem = obs[gc].astype(str).values
par = np.array([int(float(g)) % 2 for g in gem])
perts = sorted(set(pert) - {"non-targeting"}); pi = {p: i for i, p in enumerate(perts)}
idx = np.array([pi.get(p, -1) for p in pert])
n, G = a.shape
acc = np.zeros((2, len(perts), G)); cnt = np.zeros((2, len(perts)))
CH = 20000
for s in range(0, n, CH):
    X = a.X[s:s + CH]; X = X.toarray() if sp.issparse(X) else np.asarray(X)
    ii, pp = idx[s:s + CH], par[s:s + CH]
    for h in (0, 1):
        m = (ii >= 0) & (pp == h)
        np.add.at(acc[h], ii[m], X[m]); np.add.at(cnt[h], ii[m], 1)
    print(f"{min(s + CH, n)}/{n}", flush=True)
H = acc / np.maximum(cnt, 1)[:, :, None]
genes = np.asarray(a.var_names).astype(str)  # Ensembl ids, as in the published bulk files
np.savez_compressed(dst, perts=np.array(perts), genes=genes, H0=H[0].astype(np.float32), H1=H[1].astype(np.float32), n0=cnt[0], n1=cnt[1])
print("WROTE", dst, H.shape, flush=True)
