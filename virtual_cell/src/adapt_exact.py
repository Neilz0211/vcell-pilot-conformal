"""Real-data companion of adapt_theory_sim.py (discussion_round2.md section 1 and 3), K562-essential <-> RPE1.

For the source ridge model (alpha = 10, as in adapt.py) adapted by the pilot-mean offset:
  exact      coverage of the exact full-conformal set (= jackknife+ depth set), membership computed pointwise
  jk         jackknife ball, radius R_(k) of the LOO scores (what adapt.py reports)
  outer      exactly valid ball, radius (m+1)/m * R_(k)
  inner      ball of radius (m-1)/m * R_(k) (contained in the exact set)
Falsification test for "the gap is an offset": coverage in B of the offset-adapted model under the SOURCE radius q_A
(no pilot calibration): cov_srcq_adapt. Also the offset energy ratio rho_hat = Delta^2/sigma_B^2 estimated on the B
calibration targets, for mean / ridge(1) / ridge(10) / kNN(20), with the predicted and observed adapt/pilot radius ratio.
usage: uv run python src/adapt_exact.py [--seeds 30]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapt_theory_sim import full_cp_member, k_of  # noqa: E402
from context import load_pair  # noqa: E402
from data import go_annotations, go_embedding  # noqa: E402
from day1 import rmse, split_cp_q  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"


def models(Ztr, Ytr):
    M = {"mean": lambda Z: np.tile(Ytr.mean(0), (len(Z), 1))}
    for al in (1, 10):
        r = Ridge(alpha=al).fit(Ztr, Ytr); M[f"ridge{al}"] = lambda Z, r=r: r.predict(Z)
    nn = NearestNeighbors(n_neighbors=20).fit(Ztr); M["knn20"] = lambda Z: Ytr[nn.kneighbors(Z)[1]].mean(1)
    return M


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=30); ap.add_argument("--alpha", type=float, default=0.1)
    a = ap.parse_args(); alpha = a.alpha
    Ys, targets = load_pair()
    Z, _, _, has_go = go_embedding(targets, go_annotations())
    Ys = {k: v[has_go] for k, v in Ys.items()}; Z = Z[has_go]; N = len(Z); G = Z.shape[0] and Ys["K562"].shape[1]
    res = {}
    for A, B in (("K562", "RPE1"), ("RPE1", "K562")):
        rec = {}
        def put(k, v):  # noqa: E306
            rec.setdefault(k, []).append(float(v))
        for s in range(a.seeds):
            rng = np.random.default_rng(s); r = rng.permutation(N)
            tr, ca, te_all = r[: int(0.4 * N)], r[int(0.4 * N): int(0.7 * N)], r[int(0.7 * N):]
            te = te_all[100:]
            M = models(Z[tr], Ys[A][tr])
            for name, f in M.items():
                Pc, Pt = f(Z[ca]), f(Z[te])
                rA = Ys[A][ca] - Pc; rB = Ys[B][ca] - Pc
                qA = split_cp_q(rmse(Pc, Ys[A][ca]), alpha); put(f"{name}_qA", qA)
                dB = rB.mean(0); sigB2 = ((rB - dB) ** 2).mean(); n = len(ca)
                Delta2 = (dB ** 2).mean() - sigB2 / n  # bias-corrected offset energy per gene
                sigA2 = ((rA - rA.mean(0)) ** 2).mean()
                put(f"{name}_Delta2", Delta2); put(f"{name}_sigB2", sigB2); put(f"{name}_sigA2", sigA2); put(f"{name}_rho", Delta2 / sigB2)
                put(f"{name}_DeltaA2", (rA.mean(0) ** 2).mean() - sigA2 / n)
                stB = rmse(Pt, Ys[B][te]); put(f"{name}_covB_src", (stB <= qA).mean())
                put(f"{name}_qB_oracle", split_cp_q(rmse(Pc, Ys[B][ca]), alpha))
                # oracle offset (from all B calibration targets) under the source radius
                put(f"{name}_covB_srcq_oracleoffset", (rmse(Pt + dB, Ys[B][te]) <= qA).mean())
                put(f"{name}_qB_oracle_adapt", split_cp_q(rmse(Pc + dB, Ys[B][ca]), alpha))  # in-sample, n large: ~ oracle adapted radius
                for m in (10, 20, 50, 100):
                    pil = te_all[:m]; Pp = f(Z[pil]); rp = Ys[B][pil] - Pp
                    sp = np.linalg.norm(rp, axis=1) / np.sqrt(G); q0 = split_cp_q(sp, alpha)
                    put(f"{name}_m{m}_q_pilot", q0); put(f"{name}_m{m}_cov_pilot", (stB <= q0).mean())
                    rbar = rp.mean(0); e = rp - rbar; en = np.linalg.norm(e, axis=1) / np.sqrt(G)
                    k = k_of(m, alpha); Rk = np.sort(m / (m - 1) * en)[k - 1]
                    sa = rmse(Pt + rbar, Ys[B][te])
                    put(f"{name}_m{m}_q_jk", Rk); put(f"{name}_m{m}_cov_jk", (sa <= Rk).mean())
                    put(f"{name}_m{m}_cov_outer", (sa <= (m + 1) / m * Rk).mean()); put(f"{name}_m{m}_cov_inner", (sa <= (m - 1) / m * Rk).mean())
                    put(f"{name}_m{m}_cov_srcq_adapt", (sa <= qA).mean())
                    put(f"{name}_m{m}_ratio_pred", np.sqrt((1 + 1 / m) / (1 + Delta2 / sigB2)))
                    if name == "ridge10":
                        u = (Ys[B][te] - Pt) - rbar
                        put(f"{name}_m{m}_cov_exact", full_cp_member(e, u, k).mean())
        row = {k: float(np.mean(v)) for k, v in rec.items()}
        res[f"{A}->{B}"] = row
        print(f"== {A}->{B}", flush=True)
        for name in ("mean", "ridge1", "ridge10", "knn20"):
            print(f"  {name:8s} qA {row[name+'_qA']:.3f} qB_or {row[name+'_qB_oracle']:.3f} qB_or_adapt {row[name+'_qB_oracle_adapt']:.3f} "
                  f"Delta2 {row[name+'_Delta2']:.3f} sigB2 {row[name+'_sigB2']:.3f} sigA2 {row[name+'_sigA2']:.3f} rho {row[name+'_rho']:.3f} "
                  f"covB_src {row[name+'_covB_src']:.3f} covB_srcq_oracleoffset {row[name+'_covB_srcq_oracleoffset']:.3f}", flush=True)
            for m in (10, 20, 50, 100):
                p = f"{name}_m{m}_"
                ex = f" exact {row[p+'cov_exact']:.3f}" if p + "cov_exact" in row else ""
                print(f"      m={m:3d} pilot {row[p+'cov_pilot']:.3f}/{row[p+'q_pilot']:.3f}  jk {row[p+'cov_jk']:.3f}/{row[p+'q_jk']:.3f}"
                      f"  inner {row[p+'cov_inner']:.3f} outer {row[p+'cov_outer']:.3f}{ex}  srcq_adapt {row[p+'cov_srcq_adapt']:.3f}"
                      f"  ratio obs {row[p+'q_jk']/row[p+'q_pilot']:.3f} pred {row[p+'ratio_pred']:.3f}", flush=True)
    (OUT / "adapt_exact_metrics.json").write_text(json.dumps(res, indent=1))
    print("ADAPT_EXACT_DONE", flush=True)


if __name__ == "__main__":
    main()
