"""Diagnostic for the weak gwps functional embedding in ladder.py (discussion_round2.md section 4).

Checks: (1) var / obs naming in the gwps h5ad and the target-to-column match rate; (2) PCA explained variance of the
64 whitened PCs (whitening equalises noise PCs for kNN); (3) neighbour predictiveness in line A (K562-essential):
for test targets, correlation between the target's own response and the mean response of its k = 20 nearest
training targets, under GO z, the column embedding F (whitened / unwhitened / 16 PCs), and the gene's own gwps
knockdown ROW restricted to the shared readout genes (a same-line replicate -- an upper reference, not a model
feature); the same for the RPE1 responses to see what transfers.
usage: uv run python src/gwps_embed_check.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import anndata as ad
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context import load_pair  # noqa: E402
from data import DATA, go_annotations, go_embedding, target_gene  # noqa: E402

Ys, targets = load_pair()
Zg, _, _, has_go = go_embedding(targets, go_annotations())
a = ad.read_h5ad(DATA / "K562_gwps_normalized_bulk_01.h5ad")
print("var columns:", list(a.var.columns)[:10], "| var_names[:3]", list(a.var_names[:3]), "| obs_names[:2]", list(a.obs_names[:2]))
names = np.asarray(a.obs_names)
keep = np.array([not re.search(r"non-targeting|control", n, re.I) for n in names])
X = np.nan_to_num(np.asarray(a.X, dtype=np.float32)[keep], nan=0.0, posinf=0.0, neginf=0.0)
sym = np.asarray(a.var["gene_name"]).astype(str); vn = np.asarray(a.var_names).astype(str)
rows_t = np.array([target_gene(n) for n in names[keep]])
del a
print("gwps X", X.shape, "| var_names look like symbols:", bool(re.match(r"^[A-Z0-9-]+$", vn[0])), "| gene_name==var_names frac", float((sym == vn).mean()))
idx = {g: i for i, g in enumerate(sym)}; ridx = {g: i for i, g in enumerate(rows_t)}
have_col = np.array([g in idx for g in targets]); have_row = np.array([g in ridx for g in targets])
print(f"targets {len(targets)}: with gwps readout column {have_col.sum()}, with gwps knockdown row {have_row.sum()}, GO {has_go.sum()}")
keep_t = has_go & have_col & have_row
T = np.where(keep_t)[0]; print("targets kept for the check:", len(T))
cols = np.clip(X[:, [idx[targets[i]] for i in T]].T, -10, 10)
pca = PCA(64, random_state=0).fit(cols - cols.mean(0))
ev = pca.explained_variance_ratio_; print("PCA explained variance: first 5", np.round(ev[:5], 3), "sum64", round(float(ev.sum()), 3), "pc64/pc1", round(float(ev[63] / ev[0]), 4))
E = pca.transform(cols - cols.mean(0)); Ew = E / (E.std(0) + 1e-8)
# the knockdown row of each target, restricted to the readout genes used by load_pair (shared, top-variance)
# load_pair's readout genes are not returned, so use the gwps columns of the readout set by name via a second load
rows = np.clip(X[[ridx[targets[i]] for i in T]], -10, 10)
rowP = PCA(64, random_state=0).fit_transform(rows - rows.mean(0))
del X
Z = Zg[T]
feats = {"GO_z": Z, "F_whitened64": Ew, "F_raw64": E, "F_raw16": E[:, :16], "ROW_pca64(same-line replicate)": rowP}
rng = np.random.default_rng(0)
for line in ("K562", "RPE1"):
    Y = Ys[line][T]
    out = {}
    for s in range(5):
        r = rng.permutation(len(T)); tr, te = r[: len(T) // 2], r[len(T) // 2:]
        for name, F in feats.items():
            nn = NearestNeighbors(n_neighbors=20).fit(F[tr]); nb = nn.kneighbors(F[te])[1]
            pred = Y[tr][nb].mean(1); mu = Y[tr].mean(0)
            # RMSE of kNN prediction vs predict-the-mean, and corr of (pred - mu) with (Y - mu)
            r_knn = np.sqrt(((pred - Y[te]) ** 2).mean()); r_mu = np.sqrt(((mu - Y[te]) ** 2).mean())
            c = np.mean([np.corrcoef(pred[i] - mu, Y[te][i] - mu)[0, 1] for i in range(len(te))])
            out.setdefault(name, []).append((r_knn / r_mu, c))
    print(f"== responses in {line}: kNN20 RMSE / mean RMSE, mean corr(pred-mu, y-mu)")
    for name, v in out.items():
        v = np.array(v); print(f"  {name:32s} {v[:, 0].mean():.3f}  {v[:, 1].mean():.3f}")
print("GWPS_CHECK_DONE")
