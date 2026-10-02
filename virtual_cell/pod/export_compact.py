"""Pod-side: compact export of the Nadig pseudobulks for local experiments.
Keeps readout genes present in all four lines (K562-ess, RPE1, HepG2, Jurkat), finite everywhere, top 3000 by summed
variance; float16. Writes /root/compact.npz with per-line matrices, obs names, n_cells, control_expr and gene ids."""
import anndata as ad
import numpy as np

D = "/root/vc/data/"
F = {"ess": "K562_essential_normalized_bulk_01.h5ad", "rpe1": "rpe1_normalized_bulk_01.h5ad",
     "hepg2": "hepg2_pb.h5ad", "jurkat": "jurkat_pb.h5ad"}
A = {k: ad.read_h5ad(D + f) for k, f in F.items()}
common = sorted(set.intersection(*[set(a.var_names) for a in A.values()]))
mats = {k: np.asarray(a[:, common].X, dtype=np.float64) for k, a in A.items()}
fin = np.all([np.isfinite(m).all(0) for m in mats.values()], axis=0)
v = sum(np.nan_to_num(m).var(0) for m in mats.values())
v[~fin] = -1
top = np.argsort(v)[::-1][:3000]
out = {"genes": np.array(common)[top]}
for k in ("hepg2", "jurkat"):
    a = A[k]
    out[f"{k}_X"] = mats[k][:, top].astype(np.float16)
    out[f"{k}_obs"] = np.asarray(a.obs_names)
    out[f"{k}_ncells"] = np.asarray(a.obs["n_cells"], dtype=np.float32)
    out[f"{k}_ctrl"] = np.asarray(a.obs["control_expr"], dtype=np.float32)
np.savez_compressed("/root/compact.npz", **out)
print("EXPORTED", {k: v.shape for k, v in out.items()})
