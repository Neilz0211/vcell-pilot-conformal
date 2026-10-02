"""What is in the context offset? (GPT review item 4)

For each directed line pair, the offset delta = mean(Y_B) - mean(Y_A) over shared targets (1,000 shared readout genes,
standardised within each line, as in the transfer matrix). Readout genes ranked by delta; the top 50 up and top 50
down genes are tested for GO Biological Process enrichment against the 1,000-gene readout background (one-sided
hypergeometric, BH within pair, terms with 5-300 background genes). Also reports the share of offset energy carried
by a few named programmes (cell cycle GO:0007049, p53 signalling GO:0072331, response to unfolded protein
GO:0006986, ribosome biogenesis GO:0042254) relative to their share of genes.
Also a split-half stability check: delta estimated on two disjoint random halves of the targets (correlation).
usage: uv run python src/offset_enrich.py"""
from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import anndata as ad
import numpy as np
from scipy.stats import hypergeom

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import DATA  # noqa: E402
from rho_rule import load_line  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
PROGRAMS = {"cell cycle": "GO:0007049", "p53 signalling": "GO:0072331", "unfolded protein response": "GO:0006986",
            "ribosome biogenesis": "GO:0042254"}


def go_bp_with_names():
    """gene symbol -> set of GO BP ids (direct annotations, propagated up is_a/part_of using go-basic.obo)."""
    parents, names, ns = defaultdict(set), {}, {}
    cur = None
    for line in open(DATA / "go-basic.obo", encoding="utf-8"):
        line = line.strip()
        if line == "[Term]":
            cur = None
        elif line.startswith("id: GO:"):
            cur = line[4:]
        elif cur and line.startswith("name: "):
            names[cur] = line[6:]
        elif cur and line.startswith("namespace: "):
            ns[cur] = line[11:]
        elif cur and line.startswith("is_a: "):
            parents[cur].add(line[6:16])
        elif cur and line.startswith("relationship: part_of "):
            parents[cur].add(line[22:32])
    anc = {}

    def up(t):
        if t in anc:
            return anc[t]
        s = {t}
        for p in parents.get(t, ()):
            s |= up(p)
        anc[t] = s
        return s
    sys.setrecursionlimit(10000)
    g2t = defaultdict(set)
    with gzip.open(DATA / "goa_human.gaf.gz", "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("!"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 9 or "NOT" in c[3] or c[8] != "P":
                continue
            g2t[c[2]] |= {t for t in up(c[4]) if ns.get(t) == "biological_process"}
    return g2t, names


def main():
    compact = dict(np.load(Path(__file__).resolve().parents[1] / "data" / "nadig_compact.npz", allow_pickle=True))
    ens2sym = dict(zip(ad.read_h5ad(DATA / "K562_essential_normalized_bulk_01.h5ad", backed="r").var_names,
                       ad.read_h5ad(DATA / "K562_essential_normalized_bulk_01.h5ad", backed="r").var["gene_name"].astype(str)))
    g2t, names = go_bp_with_names()
    cache = {}; res = {}
    lines = ("ess", "rpe1", "hepg2", "jurkat")
    for A in lines:
        for B in lines:
            if A >= B:
                continue
            (XA, tA, gA), (XB, tB, gB) = load_line(A, compact, cache), load_line(B, compact, cache)
            ct = sorted(set(tA) & set(tB)); cg = sorted(set(gA) & set(gB))
            ia = {x: i for i, x in enumerate(tA)}; ib = {x: i for i, x in enumerate(tB)}
            ga = {x: i for i, x in enumerate(gA)}; gb = {x: i for i, x in enumerate(gB)}
            YA = XA[[ia[x] for x in ct]][:, [ga[x] for x in cg]].astype(np.float64)
            YB = XB[[ib[x] for x in ct]][:, [gb[x] for x in cg]].astype(np.float64)
            fin = np.isfinite(YA).all(0) & np.isfinite(YB).all(0); YA, YB = YA[:, fin], YB[:, fin]
            cgf = np.array(cg)[fin]
            top = np.argsort(YA.var(0) + YB.var(0))[::-1][:1000]
            YA, YB, genes = YA[:, top] / (YA[:, top].std(0) + 1e-8), YB[:, top] / (YB[:, top].std(0) + 1e-8), cgf[top]
            syms = np.array([ens2sym.get(g, g) for g in genes])
            delta = YB.mean(0) - YA.mean(0)
            rng = np.random.default_rng(0); perm = rng.permutation(len(YA)); h1, h2 = perm[: len(perm) // 2], perm[len(perm) // 2:]
            split_corr = float(np.corrcoef(YB[h1].mean(0) - YA[h1].mean(0), YB[h2].mean(0) - YA[h2].mean(0))[0, 1])
            bg = [g2t.get(s, set()) for s in syms]
            term_bg = defaultdict(int)
            for ts in bg:
                for t in ts:
                    term_bg[t] += 1
            out = {"Delta2": float((delta ** 2).mean()), "split_half_corr": split_corr}
            for label, idx in (("up_in_" + B, np.argsort(delta)[::-1][:50]), ("up_in_" + A, np.argsort(delta)[:50])):
                cnt = defaultdict(int)
                for i in idx:
                    for t in bg[i]:
                        cnt[t] += 1
                rows = []
                for t, k in cnt.items():
                    K = term_bg[t]
                    if 5 <= K <= 300 and k >= 3:
                        rows.append((hypergeom.sf(k - 1, len(syms), K, len(idx)), t, k, K))
                rows.sort()
                n = len(rows); q = []
                for r_, (p, t, k, K) in enumerate(rows):
                    q.append(min(1.0, p * n / (r_ + 1)))
                for j in range(len(q) - 2, -1, -1):
                    q[j] = min(q[j], q[j + 1])
                out[label] = [{"term": t, "name": names.get(t, t), "k": k, "K": K, "p": p, "q": qq}
                              for (p, t, k, K), qq in zip(rows[:8], q[:8])]
                out[label + "_genes"] = syms[idx[:15]].tolist()
            share = {}
            tot = (delta ** 2).sum()
            for pname, t in PROGRAMS.items():
                mask = np.array([t in ts for ts in bg])
                share[pname] = {"gene_frac": float(mask.mean()), "energy_frac": float((delta[mask] ** 2).sum() / tot)}
            out["program_share"] = share
            res[f"{A}-{B}"] = out
            print(f"== {A} vs {B}: Delta2 {out['Delta2']:.3f}, split-half corr {split_corr:.2f}", flush=True)
            for label in ("up_in_" + B, "up_in_" + A):
                print(f"  {label}: genes {', '.join(out[label + '_genes'][:10])}", flush=True)
                for r_ in out[label][:4]:
                    print(f"    {r_['name'][:55]:55s} {r_['k']}/{r_['K']} q={r_['q']:.2g}", flush=True)
            print("  programme energy share vs gene share: " + "; ".join(f"{k} {v['energy_frac']:.2f}/{v['gene_frac']:.2f}" for k, v in share.items()), flush=True)
    (OUT / "offset_enrich.json").write_text(json.dumps(res, indent=1))
    print("ENRICH_DONE", flush=True)


if __name__ == "__main__":
    main()
