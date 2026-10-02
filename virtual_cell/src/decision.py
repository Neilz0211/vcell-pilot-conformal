"""Exploratory T2 + T3: decision-level outputs in a new line from pilots.

T2 (ellipsoidal score): residual covariance learned in the SOURCE line only (residuals of the source ridge model on the
otherwise unused 40-70 % source split; rank-k PCA plus an isotropic floor). Score s = sqrt(e' S^-1 e / G) with
e = Y_B - (prediction + pilot offset); calibrated on target pilots with the leave-one-out offset construction and the
(m+1)/m factor (as Prop A; for a Mahalanobis score this factor is heuristic, coverage measured) and, exactly, with the
pilot-only (unadapted) split version. Readouts: coverage; per-perturbation null exclusion (0 outside the set);
simultaneous per-gene intervals |y_g - c_g| <= q sqrt(G S_gg) and the share of (perturbation, gene) pairs with
|true response| > 1 SD whose interval excludes 0 (true sign recovered), versus the same readouts for the RMSE ball.
T3 (conformal selection, Jin & Candes 2023): "strong responder" = RMS response in B above c, with c the 90th
percentile of source-train RMS responses (fixed from the source). Score = predicted RMS of (prediction + pilot offset).
Conformal p-values from the pilots, Benjamini-Hochberg at q = 0.1 over the 280 test perturbations. Reports power and
realised FDR for the ridge and measured-response predictors, m in {50, 100, 200}.
usage: uv run python src/decision.py [--seeds 10] [--k 20]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import go_annotations, go_embedding  # noqa: E402
from day1 import rmse, split_cp_q  # noqa: E402
from rho_rule import load_pair  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"


def cov_model(E, k):
    Ec = E - E.mean(0); U, S, Vt = np.linalg.svd(Ec, full_matrices=False)
    lam = S ** 2 / (len(E) - 1); G = E.shape[1]
    floor = lam[k:].sum() / (G - k)
    V = Vt[:k].T; l = lam[:k]
    inv_diag = 1 / l - 1 / floor               # S^-1 = I/floor + V diag(1/l - 1/floor) V'
    def maha(X):                                # rows: e' S^-1 e / G
        proj = X @ V
        return ((X ** 2).sum(1) / floor + (proj ** 2 * inv_diag).sum(1)) / G
    Sgg = floor + (V ** 2 * (l - floor)).sum(1)
    share = lam[:k].sum() / lam.sum()
    return maha, Sgg, share


def bh(p, q):
    n = len(p); o = np.argsort(p); thr = q * np.arange(1, n + 1) / n
    ok = np.where(p[o] <= thr)[0]
    sel = np.zeros(n, bool)
    if len(ok): sel[o[: ok.max() + 1]] = True
    return sel


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.1)
    a = ap.parse_args(); al = a.alpha
    compact = dict(np.load(Path(__file__).resolve().parents[1] / "data" / "nadig_compact.npz", allow_pickle=True))
    lines = ("ess", "rpe1", "hepg2", "jurkat"); go = go_annotations(); cache = {}; res = {}
    for A in lines:
        for B in lines:
            if A == B: continue
            YA0, YB0, targets = load_pair(A, B, compact, cache, prep="raw")
            Z, _, _, has = go_embedding(targets, go); YA0, YB0, Z = YA0[has], YB0[has], Z[has]; N = len(Z)
            raw = {}; put = lambda k, v: raw.setdefault(k, []).append(float(v))  # noqa: E731
            for s in range(a.seeds):
                r = np.random.default_rng(s).permutation(N)
                tr, mid, pool = r[: int(0.4 * N)], r[int(0.4 * N): int(0.7 * N)], r[int(0.7 * N):]
                test, cal = pool[-280:], pool[:-280]
                top = np.argsort(YA0[tr].var(0))[::-1][:1000]; sd = YA0[tr][:, top].std(0) + 1e-8
                YA, YB = YA0[:, top] / sd, YB0[:, top] / sd; G = YA.shape[1]
                P = Ridge(alpha=10.0).fit(Z[tr], YA[tr]).predict(Z)
                maha, Sgg, share = cov_model(YA[mid] - P[mid], a.k); put("pc_share", share)
                big = np.abs(YB[test]) > 1.0
                # ---- T2 at m = 50
                m = 50; pil = cal[:m]; E = YB[pil] - P[pil]; b = E.mean(0); bl = (E.sum(0)[None] - E) / (m - 1)
                C = P[test] + b; et = YB[test] - C
                Rb = (m + 1) / m * split_cp_q(np.sqrt(((E - bl) ** 2).mean(1)), al)
                Re = (m + 1) / m * split_cp_q(np.sqrt(maha(E - bl)), al)
                put("ball_cov", (np.sqrt((et ** 2).mean(1)) <= Rb).mean()); put("ell_cov", (np.sqrt(maha(et)) <= Re).mean())
                put("ball_null", (np.sqrt((C ** 2).mean(1)) > Rb).mean()); put("ell_null", (np.sqrt(maha(C)) > Re).mean())
                hw_b = Rb * np.sqrt(G); hw_e = Re * np.sqrt(G * Sgg)          # simultaneous per-gene half-widths
                put("ball_gene_hw", hw_b); put("ell_gene_hw_med", np.median(hw_e))
                for nm, hw in (("ball", np.full(G, hw_b)), ("ell", hw_e)):
                    lo, hi = C - hw, C + hw; ex = (lo > 0) | (hi < 0)
                    put(f"{nm}_gene_simcov", ((YB[test] >= lo) & (YB[test] <= hi)).all(1).mean())
                    put(f"{nm}_gene_call_big", (ex & big).sum() / max(big.sum(), 1))
                    put(f"{nm}_gene_signerr", ((ex & (np.sign(C) != np.sign(YB[test])))).sum() / max(ex.sum(), 1))
                q0 = split_cp_q(np.sqrt(maha(YB[pil] - P[pil])), al)                # exact pilot-only Mahalanobis
                put("ell_pilot_cov", (np.sqrt(maha(YB[test] - P[test])) <= q0).mean())
                # ---- T3 selection
                c = np.quantile(np.sqrt((YA[tr] ** 2).mean(1)), 0.9)
                strong = np.sqrt((YB[test] ** 2).mean(1)) > c; put("base_rate", strong.mean())
                for mm in (50, 100, 200):
                    if mm > len(cal): continue
                    pl = cal[:mm]
                    for nm, Q in (("ridge", P), ("meas", YA)):
                        bb = (YB[pl] - Q[pl]).mean(0)
                        V = lambda rows: np.sqrt(((Q[rows] + bb) ** 2).mean(1))  # noqa: E731
                        vc, vt = V(pl), V(test); null_c = np.sqrt((YB[pl] ** 2).mean(1)) <= c
                        p = np.array([(1 + ((vc >= v) & null_c).sum()) / (mm + 1) for v in vt])
                        sel = bh(p, 0.1)
                        put(f"sel_{nm}_{mm}_n", sel.sum()); put(f"sel_{nm}_{mm}_power", (sel & strong).sum() / max(strong.sum(), 1))
                        put(f"sel_{nm}_{mm}_fdp", (sel & ~strong).sum() / max(sel.sum(), 1))
            x = {k: float(np.mean(v)) for k, v in raw.items()}; res[f"{A}->{B}"] = x
            print(f"{A}->{B} top{a.k} PCs {x['pc_share']:.2f} | cov ball/ell/ellpilot {x['ball_cov']:.3f}/{x['ell_cov']:.3f}/{x['ell_pilot_cov']:.3f}"
                  f" null-excl ball/ell {x['ball_null']:.3f}/{x['ell_null']:.3f} | gene hw ball {x['ball_gene_hw']:.1f} ell med {x['ell_gene_hw_med']:.1f}"
                  f" simcov {x['ball_gene_simcov']:.3f}/{x['ell_gene_simcov']:.3f} calls(|y|>1) {x['ball_gene_call_big']:.3f}/{x['ell_gene_call_big']:.3f}"
                  f" | SEL base {x['base_rate']:.2f} " + " ".join(
                      f"m{mm} ridge {x.get(f'sel_ridge_{mm}_power', np.nan):.2f}/{x.get(f'sel_ridge_{mm}_fdp', np.nan):.2f} meas {x.get(f'sel_meas_{mm}_power', np.nan):.2f}/{x.get(f'sel_meas_{mm}_fdp', np.nan):.2f}"
                      for mm in (50, 100, 200)), flush=True)
    (OUT / "decision_metrics.json").write_text(json.dumps(res))
    print("DECISION_DONE", flush=True)


if __name__ == "__main__":
    main()
