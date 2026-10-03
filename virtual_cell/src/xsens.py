"""Sensitivity run for the D1 radius comparison (2026-10-03, after the Codex review).

xmap.py / xpropa.py restrict every quantity to perturbations and genes with a split-half noise estimate in the target
line, a filter that inspects the target pool (including test perturbations). Here that filter is dropped: all
perturbations of the leak-free design are used (genes: the 1,000 of largest source-train variance among genes finite in
both lines, as in paired.py), so no target-line information beyond gene measurability enters. Compared on the same
seeds, test sets and pilots:
  propA_ridge   adapted ridge(GO), Prop. A outer ball (the paper's current recipe; all m pilots reused)
  propA_mean    pilot mean, Prop. A outer ball
  rrr           reduced-rank ridge map from the measured source response, fit on m/2 pilots (CV inside), split
                conformal on the other m/2 (exact)
usage: uv run python src/xsens.py [--seeds 20]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ceiling import load_pair_named  # noqa: E402
from data import go_annotations, go_embedding  # noqa: E402
from day1 import rmse, split_cp_q  # noqa: E402
from xmap import cv_pick, p_rrr  # noqa: E402
from xpropa import propa  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "results"
GRID = [(k, l) for k in (2, 5, 10, 20) for l in (10.0, 100.0, 1000.0)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=20); ap.add_argument("--alpha", type=float, default=0.1)
    a = ap.parse_args(); al = a.alpha
    compact = dict(np.load(ROOT / "data" / "nadig_compact.npz", allow_pickle=True))
    lines = ("ess", "rpe1", "hepg2", "jurkat"); go = go_annotations(); cache = {}; res = {}
    for A in lines:
        for B in lines:
            if A == B: continue
            YA0, YB0, targets, _ = load_pair_named(A, B, compact, cache)
            Z, _, _, has = go_embedding(targets, go); YA0, YB0, Z = YA0[has], YB0[has], Z[has]; N = len(Z)
            raw = {}; put = lambda k, v: raw.setdefault(k, []).append(float(v))  # noqa: E731
            for s in range(a.seeds):
                r = np.random.default_rng(s).permutation(N); tr, pool = r[: int(0.4 * N)], r[int(0.7 * N):]
                test, cal = pool[-280:], pool[:-280]
                top = np.argsort(YA0[tr].var(0))[::-1][:1000]; sd = YA0[tr][:, top].std(0) + 1e-8
                YA, YB = YA0[:, top] / sd, YB0[:, top] / sd
                Pr = Ridge(alpha=10.0).fit(Z[tr], YA[tr]).predict(Z)
                for m in (50, 100, 200):
                    if m > len(cal): continue
                    pil = cal[:m]; h = m // 2
                    for nm, Q in (("mean", np.zeros_like(YB)), ("ridge", Pr)):
                        b, R = propa(Q[pil], YB[pil], al)
                        put(f"propA_{nm}_q_{m}", R); put(f"propA_{nm}_cov_{m}", (rmse(Q[test] + b, YB[test]) <= R).mean())
                    hp = cv_pick(p_rrr, GRID, YA[pil[:h]], YB[pil[:h]], Pr[pil[:h]]); f = p_rrr(YA[pil[:h]], YB[pil[:h]], Pr[pil[:h]], hp)
                    q = split_cp_q(rmse(f(YA[pil[h:]], Pr[pil[h:]]), YB[pil[h:]]), al)
                    put(f"rrr_q_{m}", q); put(f"rrr_cov_{m}", (rmse(f(YA[test], Pr[test]), YB[test]) <= q).mean())
            x = {k: float(np.mean(v)) for k, v in raw.items()}; res[f"{A}->{B}"] = {"summary": x, "raw": raw}
            print(f"{A}->{B} N={N} | rrr vs propA ridge m50/100/200 "
                  + " ".join(f"{100 * np.mean(np.array(raw[f'rrr_q_{m}']) / np.array(raw[f'propA_ridge_q_{m}']) - 1):+.1f}%" for m in (50, 100, 200))
                  + " | cov rrr " + "/".join(f"{x[f'rrr_cov_{m}']:.3f}" for m in (50, 100, 200)), flush=True)
    (OUT / f"xsens_s{a.seeds}_metrics.json").write_text(json.dumps(res))
    print("XSENS_DONE", flush=True)


if __name__ == "__main__":
    main()
