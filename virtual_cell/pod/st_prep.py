"""Prepare STATE (Arc Institute) runs on Replogle-Nadig single cells with our leak-free source-only protocol.

- Keeps cells whose perturbation has a GO-SVD feature (the same 64-d features as the paper's ridge) plus controls.
- Writes data/rn.h5ad (X = obsm X_hvg = 2,000 HVGs, log-normalised; obs gene, cell_line, gem_group),
  feat.pt (gene -> 64-d tensor; non-targeting -> zeros),
  truth.npz (per line: pseudobulk delta = mean perturbed cells - mean control cells, over the 2,000 HVGs),
  splits/seed{s}.json (genes permuted: first 35% train, next 5% val, rest held out), and
  toml/{A}_s{s}.toml (one permutation of the union of genes per seed): train on A's train genes only, val = A's val genes, test = A's held-out genes, every other
  cell line entirely in test (zero-shot; predicted from its own control cells).
usage: python st_prep.py /root/st/data/replogle.h5ad st_feat.npz /root/st --seeds 0 1
"""
import argparse
import json
import os

import anndata as ad
import numpy as np
import torch

ap = argparse.ArgumentParser(); ap.add_argument("h5ad"); ap.add_argument("feat"); ap.add_argument("out")
ap.add_argument("--seeds", type=int, nargs="+", default=[0])
a = ap.parse_args()
os.makedirs(a.out + "/data", exist_ok=True); os.makedirs(a.out + "/splits", exist_ok=True); os.makedirs(a.out + "/toml", exist_ok=True)
F = np.load(a.feat, allow_pickle=True); fg = [str(x) for x in F["genes"]]; Z = F["Z"]
A = ad.read_h5ad(a.h5ad)
gene = A.obs["gene"].astype(str).values
keep = np.isin(gene, fg) | (gene == "non-targeting")
H = np.asarray(A.obsm["X_hvg"][keep], dtype=np.float32)
hvg = np.asarray(A.var_names[A.var["highly_variable"].values]) if "highly_variable" in A.var else np.arange(H.shape[1]).astype(str)
if len(hvg) != H.shape[1]: hvg = np.array([f"hvg{i}" for i in range(H.shape[1])])
obs = A.obs.loc[keep, ["gene", "cell_line", "gem_group"]].copy()
for c in ("gene", "cell_line"): obs[c] = obs[c].astype(str).astype("category")
obs["gem_group"] = obs["gem_group"].astype(str).astype("category")
B = ad.AnnData(X=H, obs=obs); B.var_names = hvg; B.obsm["X_hvg"] = H
del A
B.write_h5ad(a.out + "/data/rn.h5ad")
print("cells kept", B.n_obs, "of", len(gene), flush=True)

feat = {g: torch.tensor(z, dtype=torch.float32) for g, z in zip(fg, Z)}
feat["non-targeting"] = torch.zeros(Z.shape[1])
torch.save(feat, a.out + "/feat.pt")

lines = sorted(obs.cell_line.unique()); g = obs.gene.astype(str).values; cl = obs.cell_line.astype(str).values
truth = {}
for l in lines:
    m = cl == l; ctrl = H[m & (g == "non-targeting")].mean(0)
    ps = sorted(set(g[m]) - {"non-targeting"})
    D = np.stack([H[m & (g == p)].mean(0) - ctrl for p in ps]); n = np.array([(m & (g == p)).sum() for p in ps])
    truth[l] = (np.array(ps), D, n, ctrl)
np.savez(a.out + "/truth.npz", lines=np.array(lines), hvg=hvg,
         **{f"{l}_perts": truth[l][0] for l in lines}, **{f"{l}_delta": truth[l][1] for l in lines},
         **{f"{l}_n": truth[l][2] for l in lines}, **{f"{l}_ctrl": truth[l][3] for l in lines})
common = sorted(set.intersection(*[set(truth[l][0]) for l in lines]))
print("lines", lines, "perts per line", [len(truth[l][0]) for l in lines], "common to all", len(common), flush=True)

union = sorted(set.union(*[set(truth[l][0]) for l in lines]))
for s in a.seeds:   # one permutation of the union of genes; a gene's role is the same whichever line is the source
    r = np.random.default_rng(s).permutation(union); n = len(r)
    tr, va, ho = r[: int(0.35 * n)], r[int(0.35 * n): int(0.40 * n)], r[int(0.40 * n):]
    json.dump({"train": tr.tolist(), "val": va.tolist(), "heldout": ho.tolist()}, open(f"{a.out}/splits/seed{s}.json", "w"))
    for src in lines:
        allp = set(truth[src][0])
        t = [f'[datasets]\nrn = "{a.out}/data"\n', '[training]\nrn = "train"\n', "[zeroshot]"]
        t += [f'"rn.{l}" = "test"' for l in lines if l != src]
        t += ["", "[fewshot]", f'[fewshot."rn.{src}"]', f"val = {json.dumps(sorted(set(va) & allp))}",
              f"test = {json.dumps(sorted(set(ho) & allp))}"]
        open(f"{a.out}/toml/{src}_s{s}.toml", "w").write("\n".join(t) + "\n")
        print(src, "seed", s, "train", len(set(tr) & allp), "val", len(set(va) & allp), "heldout", len(set(ho) & allp), flush=True)
print("PREP_DONE", flush=True)
