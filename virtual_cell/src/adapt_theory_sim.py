"""Numerical check of discussion_round2.md section 1 (adapt-then-calibrate theory), synthetic data only.

Setting. Pilot residuals r_i = Y_i - f_A(z_i) in R^G, i = 1..m, and a test residual r_{m+1}; the adapted predictor is
f_A + b with b the mean residual; the score is the RMSE ball ||.||/sqrt(G).
(A) Full conformal with the augmented mean b(y) = (sum_i r_i + r)/(m+1). With e_i = r_i - rbar, u = r - rbar:
      y in C_full  <=>  #{i : S_i(y) < S_{m+1}(y)} <= k-1,  k = ceil((m+1)(1-alpha)),
      S_i = ||e_i - u/(m+1)||, S_{m+1} = m/(m+1) ||u||.
    Claim 1: {i not beaten} = {u in B_i}, B_i = ball(-e_i/(m-1), m||e_i||/(m-1)) = the jackknife+ LOO ball
      (centre r_bar_{-i} - r_bar, radius R_i^LOO = ||r_i - rbar_{-i}||), so C_full == the jackknife+ depth set.
    Claim 2: ball(0, e_(k)) subset C_full subset ball(0, (m+1)/(m-1) e_(k)), e_(k) the k-th smallest ||e_i||;
      in LOO units: ((m-1)/m) R_(k) <= C_full <= ((m+1)/m) R_(k).
    Checked: pointwise identity of memberships, the two inclusions on random probe points, and the coverages of the
    exact set, the inner ball, the jackknife ball R_(k) and the outer ball, Gaussian and heavy-tailed residuals.
(C) Gaussian efficiency model: r = delta + eps, eps ~ N(0, sigma^2 I_G), rho = ||delta||^2/(G sigma^2).
    Prediction: radius(adapt, m)/radius(pilot-only, m) ~ sqrt((1 + 1/m)/(1 + rho)).
usage: uv run python src/adapt_theory_sim.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "results"


def k_of(m, alpha):
    return int(np.ceil((m + 1) * (1 - alpha)))


def split_q(scores, alpha):
    m = len(scores); k = k_of(m, alpha)
    return np.sort(scores)[k - 1] if k <= m else np.inf


def full_cp_member(e, u, k):
    """e: m x G, u: n x G. Membership of each u in the exact full-conformal set (in rbar-centred coordinates)."""
    m = len(e)
    S_i = np.linalg.norm(e[None, :, :] - u[:, None, :] / (m + 1), axis=2)  # n x m
    S_t = m / (m + 1) * np.linalg.norm(u, axis=1)  # n
    beaten = (S_i < S_t[:, None]).sum(1)
    return beaten <= k - 1


def jackknife_plus_member(e, u, k):
    m = len(e)
    centres = -e / (m - 1)  # = rbar_{-i} - rbar
    radii = m * np.linalg.norm(e, axis=1) / (m - 1)  # = R_i^LOO
    inside = (np.linalg.norm(u[:, None, :] - centres[None, :, :], axis=2) <= radii[None, :]).sum(1)
    return inside >= m - k + 1


def part_a(alpha=0.1, G=100, reps=1500, n_test=40, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for dist in ("gauss", "heavy"):
        for m in (10, 20, 50):
            k = k_of(m, alpha)
            cov = {"exact": [], "jk_plus": [], "inner": [], "jackknife": [], "outer": [], "minmax_union": []}
            ident, inc_in, inc_out, probes = 0, 0, 0, 0
            for _ in range(reps):
                delta = rng.normal(0, 0.5, G)
                scale = lambda n: (np.exp(rng.normal(0, 0.6, (n, 1))) if dist == "heavy" else np.ones((n, 1)))  # noqa: E731
                r = delta + rng.normal(0, 1, (m, G)) * scale(m)
                rt = delta + rng.normal(0, 1, (n_test, G)) * scale(n_test)
                rbar = r.mean(0); e = r - rbar; u = rt - rbar
                en = np.linalg.norm(e, axis=1); ek = np.sort(en)[k - 1]
                R_loo = m / (m - 1) * en; Rk = np.sort(R_loo)[k - 1]
                un = np.linalg.norm(u, axis=1)
                mem = full_cp_member(e, u, k); jk = jackknife_plus_member(e, u, k)
                ident += int((mem == jk).all())
                cov["exact"].append(mem.mean()); cov["jk_plus"].append(jk.mean())
                cov["inner"].append((un <= ek).mean()); cov["jackknife"].append((un <= Rk).mean())
                cov["outer"].append((un <= (m + 1) / (m - 1) * ek).mean())
                # multivariate jackknife-minmax (Barber et al. Thm 3 proof with |.| -> ||.||): union of LOO balls of radius R_(k)
                dmin = np.linalg.norm(u[:, None, :] + e[None, :, :] / (m - 1), axis=2).min(1)
                cov["minmax_union"].append((dmin <= Rk).mean())
                # inclusion probes: random directions at radii spanning the sandwich
                d = rng.normal(0, 1, (n_test, G)); d /= np.linalg.norm(d, axis=1, keepdims=True)
                rad = ek * rng.uniform(0.5, 1.6, (n_test, 1)); p = d * rad
                pm = full_cp_member(e, p, k); pn = rad[:, 0]
                inc_in += int((~pm[pn <= ek]).sum() == 0)  # inner ball points all inside
                inc_out += int((pm[pn > (m + 1) / (m - 1) * ek]).sum() == 0)  # points outside outer ball all outside
                probes += 1
            out[f"{dist}_m{m}"] = {kk: float(np.mean(v)) for kk, v in cov.items()} | {
                "identity_fullCP_eq_jackknifePlus": ident / reps, "inner_subset_ok": inc_in / probes,
                "outer_superset_ok": inc_out / probes, "k": k, "need_in_balls": m - k + 1}
            print(dist, m, json.dumps(out[f"{dist}_m{m}"]), flush=True)
    return out


def part_c(alpha=0.1, G=200, reps=1500, seed=1):
    rng = np.random.default_rng(seed)
    out = {}
    for rho in (0.0, 0.1, 0.4):
        for m in (10, 20, 50, 100):
            k = k_of(m, alpha)
            delta = rng.normal(0, 1, G); delta *= np.sqrt(rho * G) / max(np.linalg.norm(delta), 1e-12)
            qp, qa, qo, cp, ca, co = [], [], [], [], [], []
            for _ in range(reps):
                r = delta + rng.normal(0, 1, (m, G)); rt = delta + rng.normal(0, 1, (200, G))
                sp = np.linalg.norm(r, axis=1) / np.sqrt(G); st = np.linalg.norm(rt, axis=1) / np.sqrt(G)
                q0 = split_q(sp, alpha)
                rbar = r.mean(0); en = np.linalg.norm(r - rbar, axis=1) / np.sqrt(G)
                Rk = np.sort(m / (m - 1) * en)[k - 1]; q_out = (m + 1) / m * Rk
                sa = np.linalg.norm(rt - rbar, axis=1) / np.sqrt(G)
                qp.append(q0); qa.append(Rk); qo.append(q_out)
                cp.append((st <= q0).mean()); ca.append((sa <= Rk).mean()); co.append((sa <= q_out).mean())
            pred = float(np.sqrt((1 + 1 / m) / (1 + rho)))
            out[f"rho{rho}_m{m}"] = {"q_pilot": float(np.mean(qp)), "q_adapt_jk": float(np.mean(qa)), "q_adapt_outer": float(np.mean(qo)),
                                     "ratio_obs": float(np.mean(qa) / np.mean(qp)), "ratio_pred": pred,
                                     "cov_pilot": float(np.mean(cp)), "cov_adapt_jk": float(np.mean(ca)), "cov_adapt_outer": float(np.mean(co))}
            print(f"rho={rho} m={m}", json.dumps({a: round(b, 4) for a, b in out[f"rho{rho}_m{m}"].items()}), flush=True)
    return out


if __name__ == "__main__":
    res = {"A": part_a(), "C": part_c()}
    OUT.mkdir(exist_ok=True)
    (OUT / "adapt_theory_sim.json").write_text(json.dumps(res, indent=1))
    print("SIM_DONE", flush=True)
