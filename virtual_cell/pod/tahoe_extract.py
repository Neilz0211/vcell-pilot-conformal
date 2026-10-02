"""Tahoe-100M pseudobulk DE table -> dense arrays (pod-side).

Input: tahoebio/Tahoe-100M metadata/pseudobulk_differential_expression (DESeq2 per plate x drug x dose x cell line
vs the plate's DMSO wells, 62,710 genes per block; CC0). Blocks can be split across shard boundaries, so every shard
writes the genes it holds straight into a preallocated memmap indexed by (cell line, sample).

Output in OUT/: lfc.npy, stat.npy, basemean.npy (float16, [lines, samples, genes], NaN where absent),
ncells_trt.npy / ncells_ctrl.npy (int32, [lines, samples], -1 where absent), genes.txt,
lines.txt, samples.tsv (sample, plate, drug, concentration).

Usage: python tahoe_extract.py OUT [--workers 30] [--shards N]   (--shards for a smoke test)
"""
import argparse
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

REPO = "tahoebio/Tahoe-100M"
DE = "metadata/pseudobulk_differential_expression/"
COLS = ["gene_name", "baseMean", "log2FoldChange", "stat", "plate", "drug", "concentration", "Cell_Name_Vevo",
        "n_cells_trt", "n_cells_ctrl"]


def meta(out):
    p = hf_hub_download(REPO, "metadata/sample_metadata.parquet", repo_type="dataset", local_dir=out + "/meta")
    s = pd.read_parquet(p)
    conc = s.drugname_drugconc.str.extract(r"\[\('.*?', ([\d.eE+-]+), ")[0].astype(float)
    s = pd.DataFrame({"sample": s["sample"], "plate": s.plate.str.replace("plate", "", regex=False),
                      "drug": s.drug, "concentration": conc}).reset_index(drop=True)
    p = hf_hub_download(REPO, "metadata/cell_line_metadata.parquet", repo_type="dataset", local_dir=out + "/meta")
    lines = sorted(pd.read_parquet(p).cell_name.unique())
    return s, lines


def key(plate, drug, conc):
    return f"{plate}|{drug}|{round(float(conc), 4)}"


def work(args):
    path, out, shape = args
    genes = open(out + "/genes.txt").read().split("\n")
    lines = open(out + "/lines.txt").read().split("\n")
    s = pd.read_csv(out + "/samples.tsv", sep="\t", dtype={"plate": str})
    skey = {key(p, d, c): i for i, (p, d, c) in enumerate(zip(s.plate, s.drug, s.concentration))}
    lidx = {l: i for i, l in enumerate(lines)}
    t = pq.read_table(path, columns=COLS).to_pandas()
    gi = pd.Categorical(t.gene_name, categories=genes).codes
    if (gi < 0).any():
        return path, "unknown genes %d" % (gi < 0).sum()
    keys = [key(p, d, c) for p, d, c in zip(t.plate, t.drug, t.concentration)]
    si = pd.Series(keys).map(skey)
    li = t.Cell_Name_Vevo.map(lidx)
    bad = si.isna() | li.isna()
    if bad.any():
        return path, "unmatched rows %d (%s)" % (bad.sum(), sorted(set(np.array(keys)[bad.values]))[:3])
    si = si.values.astype(np.int64); li = li.values.astype(np.int64)
    for name, col in [("lfc", "log2FoldChange"), ("stat", "stat"), ("basemean", "baseMean")]:
        m = np.load(out + f"/{name}.npy", mmap_mode="r+")
        m[li, si, gi] = t[col].values.astype(np.float16)
        m.flush(); del m
    for name, col in [("ncells_trt", "n_cells_trt"), ("ncells_ctrl", "n_cells_ctrl")]:
        m = np.load(out + f"/{name}.npy", mmap_mode="r+")
        m[li, si] = t[col].values.astype(np.int32)
        m.flush(); del m
    return path, "ok %d rows" % len(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--workers", type=int, default=30); ap.add_argument("--shards", type=int)
    a = ap.parse_args()
    out = a.out; os.makedirs(out, exist_ok=True)
    s, lines = meta(out)
    files = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f.startswith(DE))
    if a.shards: files = files[:a.shards]
    first = hf_hub_download(REPO, files[0], repo_type="dataset", local_dir=out + "/shards")
    t0 = pq.read_table(first, columns=["gene_name", "Cell_Name_Vevo"]).to_pandas()
    genes = list(dict.fromkeys(t0.gene_name))
    print("genes", len(genes), "lines", len(lines), "samples", len(s), "shards", len(files), flush=True)
    open(out + "/genes.txt", "w").write("\n".join(genes))
    open(out + "/lines.txt", "w").write("\n".join(lines))
    s.to_csv(out + "/samples.tsv", sep="\t", index=False)
    shape = (len(lines), len(s), len(genes))
    for name in ["lfc", "stat", "basemean"]:
        if not os.path.exists(out + f"/{name}.npy"):
            m = np.lib.format.open_memmap(out + f"/{name}.npy", mode="w+", dtype=np.float16, shape=shape)
            for i in range(shape[0]): m[i] = np.nan
            m.flush(); del m
    for name in ["ncells_trt", "ncells_ctrl"]:
        if not os.path.exists(out + f"/{name}.npy"):
            m = np.lib.format.open_memmap(out + f"/{name}.npy", mode="w+", dtype=np.int32, shape=shape[:2])
            m[:] = -1; m.flush(); del m
    done = set()
    if os.path.exists(out + "/done.txt"): done = set(open(out + "/done.txt").read().split())
    todo = [f for f in files if f not in done]
    log = open(out + "/done.txt", "a")
    # download with threads inside each worker call: hf_hub_download is per file and cached
    def dl(f):
        return hf_hub_download(REPO, f, repo_type="dataset", local_dir=out + "/shards")
    from concurrent.futures import ThreadPoolExecutor
    n = 0
    with ThreadPoolExecutor(16) as tp, ProcessPoolExecutor(a.workers) as pp:
        futs = {}
        for f, path in zip(todo, tp.map(dl, todo)):
            futs[pp.submit(work, (path, out, shape))] = f
        for fu in as_completed(futs):
            f = futs[fu]; path, msg = fu.result(); n += 1
            if msg.startswith("ok"):
                log.write(f + "\n"); log.flush(); os.remove(path)
            print(n, f.split("/")[-1], msg, flush=True)
    print("FIN", flush=True)


if __name__ == "__main__":
    main()
