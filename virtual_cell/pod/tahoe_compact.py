"""Tahoe-100M dense arrays -> compact analysis input (pod-side), per PREREG_tahoe.md.

Gene pool (symmetric, once): median baseMean >= 10 over all present line x unit blocks and finite LFC in >= 95% of
blocks; top 3,000 by overall LFC variance. Units: drug x dose, DMSO excluded. Features: Morgan fingerprint (radius 2,
2048 bits) from the drug SMILES, truncated SVD to 64 dimensions over all drugs, plus log10 dose.

Output OUT/compact/: Y.npy (float32 [lines, units, 3000], NaN where absent), ncells.npy ([lines, units]),
units.tsv, lines.txt, genes.txt, Z.npy ([units, 65]), fp.npy (bool [units, 2048]), report.json.
usage: python tahoe_compact.py /root/tahoe
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.decomposition import TruncatedSVD

RDLogger.DisableLog("rdApp.*")
root = sys.argv[1]; out = root + "/compact"; os.makedirs(out, exist_ok=True)
lines = open(root + "/lines.txt").read().split("\n"); genes = np.array(open(root + "/genes.txt").read().split("\n"))
S = pd.read_csv(root + "/samples.tsv", sep="\t", dtype={"plate": str})
nc = np.load(root + "/ncells_trt.npy")
lfc = np.load(root + "/lfc.npy", mmap_mode="r"); bm = np.load(root + "/basemean.npy", mmap_mode="r")
li = np.where((nc >= 0).any(1))[0]
ui = np.where((nc[li] >= 0).any(0) & (S.drug.str.upper() != "DMSO_TF").values & (S.concentration > 0).values)[0]
present = nc[np.ix_(li, ui)] >= 0
print("lines with data", len(li), "units", len(ui), "present blocks", present.sum(), flush=True)

G = len(genes); med = np.empty(G, np.float32); fin = np.empty(G, np.float32); var = np.empty(G, np.float64)
BM = np.empty((len(li), len(ui), G), np.float16); LF = np.empty((len(li), len(ui), G), np.float16)
for k, l in enumerate(li):  # contiguous per-line reads (~170 MB each)
    BM[k] = np.asarray(bm[l])[ui]; LF[k] = np.asarray(lfc[l])[ui]
BM = BM[present]; LF = LF[present]   # blocks x genes, in RAM (~2 x 8 GB)
print("loaded", BM.shape, flush=True)
CH = 2000
for g0 in range(0, G, CH):
    sl = slice(g0, min(G, g0 + CH))
    b = BM[:, sl].astype(np.float32)
    x = LF[:, sl].astype(np.float32)
    med[sl] = np.nanmedian(np.where(np.isfinite(b), b, np.inf), 0)
    f = np.isfinite(x); fin[sl] = f.mean(0)
    var[sl] = np.nanvar(np.where(f, x, np.nan), 0)
    print("genes", sl.stop, flush=True)
ok = (med >= 10) & (fin >= 0.95)
order = np.argsort(np.where(ok, var, -1))[::-1][:3000]
pool = np.sort(order)
print("genes passing filters", ok.sum(), "pool", len(pool), flush=True)

Y = np.full((len(li), len(ui), len(pool)), np.nan, np.float32)
Y[present] = LF[:, pool].astype(np.float32)
del BM, LF
np.save(out + "/Y.npy", Y); np.save(out + "/ncells.npy", nc[np.ix_(li, ui)])
U = S.iloc[ui].reset_index(drop=True); U.to_csv(out + "/units.tsv", sep="\t", index=False)
open(out + "/lines.txt", "w").write("\n".join(lines[i] for i in li))
open(out + "/genes.txt", "w").write("\n".join(genes[pool]))

D = pd.read_parquet(root + "/meta/metadata/drug_metadata.parquet") if os.path.exists(root + "/meta/metadata/drug_metadata.parquet") else None
if D is None:
    from huggingface_hub import hf_hub_download
    D = pd.read_parquet(hf_hub_download("tahoebio/Tahoe-100M", "metadata/drug_metadata.parquet", repo_type="dataset", local_dir=root + "/meta"))
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fps = {}
for d, smi in zip(D.drug, D.canonical_smiles):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if m is not None:
        fps[d.strip()] = gen.GetFingerprintAsNumPy(m).astype(bool)
drugs = sorted(fps); FP = np.array([fps[d] for d in drugs])
svd = TruncatedSVD(64, random_state=0).fit(FP / np.maximum(np.linalg.norm(FP, axis=1, keepdims=True), 1e-8))
emb = dict(zip(drugs, svd.transform(FP / np.maximum(np.linalg.norm(FP, axis=1, keepdims=True), 1e-8))))
has = np.array([d.strip() in fps for d in U.drug])  # drug names in the DE table can carry trailing spaces
Z = np.full((len(U), 65), np.nan, np.float32); F = np.zeros((len(U), 2048), bool)
for i, (d, c) in enumerate(zip(U.drug, U.concentration)):
    if d.strip() in fps:
        Z[i, :64] = emb[d.strip()]; Z[i, 64] = np.log10(c); F[i] = fps[d.strip()]
np.save(out + "/Z.npy", Z); np.save(out + "/fp.npy", F); np.save(out + "/has_fp.npy", has)
rep = {"lines": len(li), "units": len(ui), "drugs": int(U.drug.nunique()), "units_without_fp": int((~has).sum()),
       "drugs_without_fp": sorted(set(U.drug[~has])), "smiles_unparsed": sorted(set(D.drug) - set(fps)), "genes_passing": int(ok.sum()), "pool": len(pool),
       "present_blocks": int(present.sum())}
json.dump(rep, open(out + "/report.json", "w"), indent=1); print(rep, flush=True)
print("COMPACT_DONE", flush=True)
