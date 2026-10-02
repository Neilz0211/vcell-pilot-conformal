"""Replogle 2022 K562 essential pseudobulk -> perturbation-level responses, plus GO-based target-gene embeddings.

Unit = one perturbation (one pseudobulk z-normalised profile). Responses are restricted to the G most variable
readout genes and standardised per gene. z(p) = truncated SVD of the target gene's GO term memberships.
"""
from __future__ import annotations

import gzip
import re
from collections import defaultdict
from pathlib import Path

import anndata as ad
import numpy as np
from sklearn.decomposition import TruncatedSVD

DATA = Path(__file__).resolve().parents[1] / "data"


def target_gene(name: str) -> str:
    """obs names look like '<id>_<GENE>_<promoter>_<ENSG...>'; return the gene symbol."""
    parts = name.split("_")
    return parts[1] if len(parts) > 2 else name


def load_responses(path=DATA / "K562_essential_normalized_bulk_01.h5ad", n_genes=1000):
    a = ad.read_h5ad(path)
    names = np.asarray(a.obs_names)
    genes = np.array([target_gene(n) for n in names])
    keep = np.array([not re.search(r"non-targeting|control", n, re.I) for n in names])
    X = np.asarray(a.X if not hasattr(a.X, "toarray") else a.X.toarray(), dtype=np.float64)[keep]
    genes, names = genes[keep], names[keep]
    X = X[:, np.isfinite(X).all(0)]  # drop readout genes with ±inf (zero control variance)
    var = X.var(0)
    top = np.argsort(var)[::-1][:n_genes]
    Y = X[:, top]
    sd = Y.std(0) + 1e-8
    return {"Y": Y / sd, "targets": genes, "names": names, "readout": np.asarray(a.var_names)[top], "obs": a.obs.iloc[np.where(keep)[0]]}


def go_annotations(path=DATA / "goa_human.gaf.gz"):
    """gene symbol -> set of GO ids (all aspects, excluding NOT qualifiers and ND evidence)."""
    m = defaultdict(set)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("!"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 7 or "NOT" in c[3] or c[6] == "ND":
                continue
            m[c[2]].add(c[4])
    return m


def go_embedding(targets, go_map, min_genes=5, dim=64, seed=0):
    terms = defaultdict(int)
    for g in set(targets):
        for t in go_map.get(g, ()):
            terms[t] += 1
    vocab = sorted(t for t, k in terms.items() if k >= min_genes)
    idx = {t: i for i, t in enumerate(vocab)}
    M = np.zeros((len(targets), len(vocab)), np.float32)
    for r, g in enumerate(targets):
        for t in go_map.get(g, ()):
            if t in idx:
                M[r, idx[t]] = 1.0
    has_go = M.sum(1) > 0
    Z = TruncatedSVD(dim, random_state=seed).fit_transform(M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-6))
    return Z, M, vocab, has_go
