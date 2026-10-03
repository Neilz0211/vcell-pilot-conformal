"""Numerical checks for theory/vc_minimax.md (DRAFT): minimax price of adaptation in the Gaussian location model.

usage (from virtual_cell/):  uv run python src/sim_minimax.py   -> prints a table, writes results/sim_minimax.json

Model: e_i = delta + eps_i, eps_i ~ N(0, sigma^2 I_G) iid, i = 1..m+1; alpha = 0.1; k = ceil((1-alpha)(m+1)).
Quantities
  r_LB  = s_m q_{G,1-alpha},  s_m = sigma sqrt(1+1/m),  q = (1-alpha)-quantile of chi_G      (Theorem 1 lower bound)
  r_par = r_LB (parametric ball B(ebar, r_LB); exact coverage 1-alpha, attains the bound)
  r_out = ((m+1)/m) R_(k),  R_(k) = (m/(m-1)) d_(k)   (adapted outer ball, Prop. A; Theorem 2 predicts r_out/r_LB -> sqrt((m+1)/(m-1)))
  t*(u) = k-th smallest of  -<u,d_i>/(m-1) + sqrt(<u,d_i>^2/(m-1)^2 + ||d_i||^2 (m+1)/(m-1))   (radial function of C, Lemma 3)
Checks
  M1 ratio r_out / r_LB vs sqrt((m+1)/(m-1)); coverage of outer and parametric balls
  M2 radial function t*(u) == exact membership along the ray; t*(u)/r_LB -> 1 in random directions
  M3 pilot-only radius vs r_LB at delta = 0 and at rho = 3/m (not equivariant: below the bound at 0, above it for large delta)
  M4 pilot threshold: m = 8 < 1/alpha - 1: any finite-ball procedure fails on the exchangeable 'one far point' bag
  M5 subspace class dim d: lower-bound factor (1+1/m)^{d/(2G)} vs projected adapter ||e_new - P ebar|| quantile
  M6 expected-radius caveat: randomised 'empty set w.p. alpha-beta' procedure is valid with E rad -> (1-alpha+beta) r_LB
  M7 k = m: C is the union of the LOO balls, (vol C/omega_G)^{1/G} >= max_i r_i, ratio to r_LB -> m/sqrt(m^2-1) > 1 (not volume-minimax)
  M1 coverage of the outer ball is tested against k/(m+1) (Prop. A), the parametric ball against 1-alpha exactly
  M8 G = 1, m = 2, alpha = 1/3: Thm 2(ii) identity, sign of X_(k)/q, outer-ball coverage and Lemma 3 with heavy ties
  M9 Thm 2(iii): empirical frequency inside the two-sided bound >= 1 - 2m exp(-t^2/2); the G condition at G = 1
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import chi

OUT = Path(__file__).resolve().parents[1] / "results"
ALPHA = 0.1
RES: dict = {}
rng = np.random.default_rng(20261002)


def k_of(m: int, alpha: float = ALPHA) -> int:
    return int(np.ceil((1 - alpha) * (m + 1)))


def r_lb(G: int, m: int, sigma: float = 1.0, alpha: float = ALPHA) -> float:
    return sigma * np.sqrt(1 + 1 / m) * chi.ppf(1 - alpha, G)


def draw(reps: int, m: int, G: int, sigma: float = 1.0):
    """pilots E (reps x m x G) and test Z (reps x G), delta = 0 (equivariant statistics do not depend on delta)."""
    E = sigma * rng.standard_normal((reps, m, G))
    Z = sigma * rng.standard_normal((reps, G))
    return E, Z


def outer_radius(E, k):
    m = E.shape[1]
    d = E - E.mean(1, keepdims=True)
    dn = np.sort(np.linalg.norm(d, axis=-1), axis=1)
    Rk = m / (m - 1) * dn[:, k - 1]
    return (m + 1) / m * Rk


def radial_C(E, u, k):
    """radial function t*(u) of the exact full-conformal region C, from the pilot mean, direction u (unit, reps x G)."""
    m = E.shape[1]
    d = E - E.mean(1, keepdims=True)                     # reps x m x G
    a = np.einsum("rmg,rg->rm", d, u)                    # <u, d_i>
    dn2 = (d ** 2).sum(-1)
    tplus = -a / (m - 1) + np.sqrt(a ** 2 / (m - 1) ** 2 + dn2 * (m + 1) / (m - 1))
    return np.sort(tplus, axis=1)[:, k - 1]


def depth_member(E, Z, k):
    m = E.shape[1]
    B = (E.sum(1, keepdims=True) - E) / (m - 1)
    r = np.linalg.norm(E - B, axis=-1)
    dist = np.linalg.norm(Z[:, None, :] - B, axis=-1)
    return (dist <= r).sum(1) >= m + 1 - k


# ---------------- M1: outer ball vs lower bound ----------------
def m1():
    rows = []
    for G in (10, 50, 200, 1000):
        for m in (10, 20, 50):
            reps = 2000 if G <= 200 else 600
            k = k_of(m)
            E, Z = draw(reps, m, G)
            rout = outer_radius(E, k)
            rlb = r_lb(G, m)
            ratio = rout / rlb
            pred = np.sqrt((m + 1) / (m - 1))
            cov_out = (np.linalg.norm(Z - E.mean(1), axis=-1) <= rout).mean()
            cov_par = (np.linalg.norm(Z - E.mean(1), axis=-1) <= rlb).mean()
            row = dict(G=G, m=m, k=k, ratio_mean=float(ratio.mean()), ratio_sd=float(ratio.std()),
                       pred_sqrt=float(pred), cov_outer=float(cov_out), cov_param=float(cov_par),
                       nominal=float(k / (m + 1)), mc=float(np.sqrt(0.09 / reps)))
            rows.append(row)
            print(f"M1 G={G:5d} m={m:3d}  r_out/r_LB = {row['ratio_mean']:.4f} (sd {row['ratio_sd']:.4f})  "
                  f"sqrt((m+1)/(m-1)) = {pred:.4f}  cov_out={cov_out:.3f} cov_par={cov_par:.3f} nominal={k/(m+1):.3f}")
    # outer ball contains C, whose coverage is >= k/(m+1) (Prop. A); parametric ball covers exactly 1-alpha
    ok = all(r["cov_outer"] >= r["nominal"] - 3 * r["mc"] and abs(r["cov_param"] - (1 - ALPHA)) <= 3 * r["mc"] for r in rows)
    big = [r for r in rows if r["G"] == 1000]
    ok &= all(abs(r["ratio_mean"] / r["pred_sqrt"] - 1) < 0.03 for r in big)
    RES["M1"] = dict(ok=bool(ok), rows=rows)
    print(f"[{'PASS' if ok else 'FAIL'}] M1\n")


# ---------------- M2: radial function of C ----------------
def m2():
    rows = []
    mism = 0
    tot = 0
    for G in (10, 50, 200, 1000):
        for m in (10, 20, 50):
            reps = 1000 if G <= 200 else 300
            k = k_of(m)
            E, _ = draw(reps, m, G)
            u = rng.standard_normal((reps, G))
            u /= np.linalg.norm(u, axis=-1, keepdims=True)
            ts = radial_C(E, u, k)
            # membership check: just inside / just outside along the ray
            eb = E.mean(1)
            inside = depth_member(E, eb + (ts * (1 - 1e-6))[:, None] * u, k)
            outside = depth_member(E, eb + (ts * (1 + 1e-6))[:, None] * u, k)
            mism += int((~inside).sum() + outside.sum())
            tot += 2 * reps
            rlb = r_lb(G, m)
            rows.append(dict(G=G, m=m, t_over_rlb_mean=float((ts / rlb).mean()), t_over_rlb_sd=float((ts / rlb).std()),
                             outer_over_rlb=float((outer_radius(E, k) / rlb).mean())))
            print(f"M2 G={G:5d} m={m:3d}  t*(u)/r_LB = {rows[-1]['t_over_rlb_mean']:.4f} (sd {rows[-1]['t_over_rlb_sd']:.4f})"
                  f"   outer/r_LB = {rows[-1]['outer_over_rlb']:.4f}")
    big = [r for r in rows if r["G"] == 1000]
    ok = mism == 0 and all(abs(r["t_over_rlb_mean"] - 1) < 0.03 for r in big)
    RES["M2"] = dict(ok=bool(ok), membership_mismatches=mism, probes=tot, rows=rows)
    print(f"[{'PASS' if ok else 'FAIL'}] M2  radial-formula mismatches {mism}/{tot}  "
          "(random rays from ebar only: a ratio statement along typical directions, NOT a volume or enclosing-radius test)\n")


# ---------------- M3: pilot-only is not equivariant ----------------
def m3():
    rows = []
    G, m, reps = 200, 20, 2000
    k = k_of(m)
    rlb = r_lb(G, m)
    for rho in (0.0, 1 / m, 3 / m, 1.0):
        delta = np.zeros(G)
        delta[0] = np.sqrt(rho * G)
        E, Z = draw(reps, m, G)
        E = E + delta
        Z = Z + delta
        rpo = np.sort(np.linalg.norm(E, axis=-1), axis=1)[:, k - 1]
        cov = (np.linalg.norm(Z, axis=-1) <= rpo).mean()
        rows.append(dict(rho=rho, pilot_only_over_rlb=float((rpo / rlb).mean()), cov=float(cov),
                         outer_over_rlb=float((outer_radius(E, k) / rlb).mean())))
        print(f"M3 rho={rho:.3f}  pilot-only/r_LB = {rows[-1]['pilot_only_over_rlb']:.4f} (cov {cov:.3f})   "
              f"outer/r_LB = {rows[-1]['outer_over_rlb']:.4f}")
    ok = rows[0]["pilot_only_over_rlb"] < 1 < rows[-1]["pilot_only_over_rlb"]
    RES["M3"] = dict(ok=bool(ok), rows=rows)
    print(f"[{'PASS' if ok else 'FAIL'}] M3  (below the minimax bound at delta=0, above it for large delta: sup over delta is what is bounded)\n")


# ---------------- M4: pilot threshold via the exchangeable bag ----------------
def m4():
    G, m, reps = 5, 8, 200000
    k = k_of(m)
    assert k == m + 1, "m < 1/alpha - 1 should give k = m+1 (set = R^G for the paper's procedures)"
    # a hypothetical finite procedure: ball centred at the pilot mean with radius 3 * max_i ||d_i||
    X = rng.standard_normal((reps, m, G))
    y = np.zeros(G)
    y[0] = 1e3
    # exchangeable bag {X_1..X_m, y}: test = y w.p. 1/(m+1), then pilots are X (iid N(0,I)); otherwise test = one X_j and y is a pilot
    is_far = rng.random(reps) < 1 / (m + 1)
    cov = np.empty(reps, dtype=bool)
    # case test = y
    Ef = X[is_far]
    rad = 3 * np.linalg.norm(Ef - Ef.mean(1, keepdims=True), axis=-1).max(1)
    cov[is_far] = np.linalg.norm(y - Ef.mean(1), axis=-1) <= rad
    # case test = X_m (by exchangeability any X_j), pilots = {X_1..X_{m-1}, y}
    En = X[~is_far]
    pil = np.concatenate([En[:, : m - 1], np.broadcast_to(y, (En.shape[0], 1, G))], axis=1)
    rad = 3 * np.linalg.norm(pil - pil.mean(1, keepdims=True), axis=-1).max(1)
    cov[~is_far] = np.linalg.norm(En[:, m - 1] - pil.mean(1), axis=-1) <= rad
    c = cov.mean()
    p_far = cov[is_far].mean()
    # Prop. 4 gives coverage <= 1 - (1/(m+1)) P(y not in C | test = y); the ceiling 1 - 1/(m+1) is reached only as
    # P(y in C | test=y) -> 0, which for Gaussian pilots happens only in the limit ||y|| -> infinity (here it is 0 to MC precision)
    bound = 1 - (1 - p_far) / (m + 1)
    mc = float(np.sqrt(0.1 / reps))
    print(f"M4 m={m}, alpha={ALPHA}: k={k}=m+1 (paper's sets are R^G). finite-ball procedure on the far-point bag: "
          f"coverage {c:.4f}; Prop-4 bound 1-(1-P(y in C|test=y))/(m+1) = {bound:.4f} (P(y in C|test=y) = {p_far:.4f}); "
          f"both < 0.9 (MC s.e. {mc:.4f})")
    ok = c < 1 - ALPHA and c <= bound + 3 * mc
    RES["M4"] = dict(ok=bool(ok), m=m, k=k, coverage=float(c), prop4_bound=float(bound),
                     p_far_covered=float(p_far), mc=mc)
    print(f"[{'PASS' if ok else 'FAIL'}] M4\n")


# ---------------- M5: subspace class ----------------
def m5():
    # the projected-adapter score ||e_new - P ebar||^2 = sigma^2 [chi2_{G-d} + (1+1/m) chi2_d] exactly (independent parts),
    # so its (1-alpha)-quantile is computed from 4e6 direct chi-square draws (relative MC error ~ 4e-5), not from G-sized arrays
    G, m, n = 200, 20, 4_000_000
    q = chi.ppf(1 - ALPHA, G)
    rows = []
    for d in (0, 20, 100, 200):
        s2 = (rng.chisquare(G - d, n) if d < G else 0.0) + (1 + 1 / m) * (rng.chisquare(d, n) if d > 0 else 0.0)
        qa = np.sqrt(np.quantile(s2, 1 - ALPHA))
        lb_factor = (1 + 1 / m) ** (d / (2 * G))      # G-th-root volume factor of the HPD ellipsoid vs oracle (Thm 3)
        ball_factor = qa / q                          # G-th-root volume factor of the projected-adapter ball vs oracle
        rows.append(dict(d=d, lb_factor=float(lb_factor), ball_factor=float(ball_factor),
                         first_order=float(1 + d / (2 * m * G))))
        print(f"M5 d={d:3d}: minimax (G-th-root volume) factor (1+1/m)^(d/2G) = {lb_factor:.5f}   "
              f"projected-adapter ball factor = {ball_factor:.5f}   1 + d/(2mG) = {rows[-1]['first_order']:.5f}")
    tol = 2e-4   # ~5 relative MC s.e. of the 0.9-quantile from 4e6 draws
    ok = all(r["ball_factor"] >= r["lb_factor"] - tol for r in rows)
    ok &= all(abs(rows[i]["ball_factor"] - rows[i]["lb_factor"]) <= tol for i in (0, 3))   # equality at d = 0 and d = G
    RES["M5"] = dict(ok=bool(ok), rows=rows, tolerance=tol, n=n,
                     note="ball factor uses the exact score law and a 4e6-draw quantile; equality expected at d=0 and d=G")
    print(f"[{'PASS' if ok else 'FAIL'}] M5  (ball factor >= lower-bound factor within MC tolerance {tol}; "
          "both ~ 1 + d/(2mG); at d = 0 and d = G the ball IS the HPD set, so equality is expected)\n")


# ---------------- M6: expected radius caveat ----------------
def m6():
    # the adapted score ||e_new - ebar|| is exactly s_m * chi_G, so simulate it directly (large G without G-sized arrays)
    # Cor. 1 construction: empty w.p. alpha - beta, otherwise the ball of coverage c_beta = (1-alpha)/(1-alpha+beta)
    m, reps, beta = 20, 200000, 1e-4
    c_beta = (1 - ALPHA) / (1 - ALPHA + beta)
    rows = []
    for G in (200, 2000, 20000):
        rlb = r_lb(G, m)
        score = np.sqrt(1 + 1 / m) * chi.rvs(G, size=reps, random_state=rng)
        r_big = np.sqrt(1 + 1 / m) * chi.ppf(c_beta, G)         # ball with coverage exactly c_beta
        empty = rng.random(reps) < ALPHA - beta
        rad = np.where(empty, 0.0, r_big)
        cov = (~empty) & (score <= r_big)
        rows.append(dict(G=G, coverage=float(cov.mean()), E_rad_over_rlb=float(rad.mean() / rlb),
                         exact_E_rad_over_rlb=float((1 - ALPHA + beta) * r_big / rlb),
                         p_rad_ge_rlb=float((rad >= rlb).mean()), r_big_over_rlb=float(r_big / rlb)))
        print(f"M6 G={G:6d} Cor-1 randomised procedure (beta={beta:g}, c_beta={c_beta:.5f}): coverage {cov.mean():.4f} (exact 0.9), "
              f"E rad / r_LB = {rad.mean()/rlb:.4f} (exact {(1-ALPHA+beta)*r_big/rlb:.4f}), P(rad >= r_LB) = {(rad >= rlb).mean():.4f}")
    ok = all(r["coverage"] >= 1 - ALPHA - 3 * np.sqrt(0.09 / reps) for r in rows) and rows[-1]["E_rad_over_rlb"] < 1
    RES["M6"] = dict(ok=bool(ok), rows=rows, beta=beta, c_beta=float(c_beta), limit=float(1 - ALPHA + beta))
    print(f"[{'PASS' if ok else 'FAIL'}] M6  (E rad can fall below r_LB, towards (1-alpha+beta) r_LB = {1-ALPHA+beta:.4f} r_LB "
          "as G grows; the quantile statement P(rad >= r_0) >= 1 - alpha/(1-F) cannot be cheated)\n")


# ---------------- M8: G = 1, m = 2, ties ----------------
def m8():
    """Smallest case: G = 1, m = 2, alpha = 1/3 (k = 2 = m). (a) Thm 2(ii) exact representation under Gaussian noise and the
    sign of X_(k)/q (below 1 on average here, so no general sign claim). (b) Integer-valued residuals with many ties:
    outer-ball coverage >= k/(m+1) and the radial formula of Lemma 3 (ties included) agrees with exact membership."""
    G, m, alpha, reps = 1, 2, 1 / 3, 200000
    k = int(np.ceil((1 - alpha) * (m + 1)))
    assert k == 2
    # (a) Gaussian: rad_out = sigma sqrt((m+1)^2/(m(m-1))) X_(k) with X_i = |d_i| / sqrt((m-1)/m)
    E, Z = draw(reps, m, G)
    rout = outer_radius(E, k)
    X = np.sort(np.abs(E - E.mean(1, keepdims=True))[:, :, 0] / np.sqrt((m - 1) / m), axis=1)[:, k - 1]
    rep = np.sqrt((m + 1) ** 2 / (m * (m - 1))) * X
    ident_err = float(np.max(np.abs(rout - rep) / np.maximum(rep, 1e-300)))
    q = chi.ppf(1 - alpha, G)
    ratio_mean = float((X / q).mean())
    cov_gauss = float((np.abs(Z[:, 0] - E.mean(1)[:, 0]) <= rout).mean())
    # (b) ties: residuals uniform on {-1, 0, 1}
    Ei = rng.integers(-1, 2, size=(reps, m, G)).astype(float)
    Zi = rng.integers(-1, 2, size=(reps, G)).astype(float)
    k_i = k
    routi = outer_radius(Ei, k_i)
    cov_ties = float((np.abs(Zi[:, 0] - Ei.mean(1)[:, 0]) <= routi).mean())
    u = np.where(rng.random(reps) < 0.5, -1.0, 1.0)[:, None]
    ts = radial_C(Ei, u, k_i)
    eb = Ei.mean(1)
    inside = depth_member(Ei, eb + (ts * (1 - 1e-9))[:, None] * u, k_i)
    outside = depth_member(Ei, eb + (ts + 1e-6)[:, None] * u, k_i)
    mism = int((~inside).sum() + outside.sum())
    mc = float(np.sqrt(0.25 / reps))
    ok = ident_err < 1e-12 and cov_gauss >= k / (m + 1) - 3 * mc and cov_ties >= k / (m + 1) - 3 * mc and mism == 0 and ratio_mean < 1
    RES["M8"] = dict(ok=bool(ok), G=G, m=m, alpha=alpha, k=k, identity_max_rel_err=ident_err, X_over_q_mean=ratio_mean,
                     cov_gauss=cov_gauss, cov_ties=cov_ties, nominal=k / (m + 1), radial_mismatches_ties=mism, mc=mc)
    print(f"M8 G=1 m=2 alpha=1/3 k=2: Thm2(ii) identity max rel err {ident_err:.1e}; mean X_(k)/q = {ratio_mean:.4f} (<1 here);"
          f" outer-ball coverage Gaussian {cov_gauss:.4f}, ties {cov_ties:.4f} (nominal k/(m+1) = {k/(m+1):.4f}, MC {mc:.4f});"
          f" radial formula with ties: {mism} mismatches")
    print(f"[{'PASS' if ok else 'FAIL'}] M8\n")


# ---------------- M9: Thm 2(iii) two-sided bound and its G condition ----------------
def m9():
    rows = []
    for (G, m, t) in ((200, 20, 3.0), (200, 20, 4.0), (50, 10, 4.0)):
        reps = 20000
        k = k_of(m)
        assert G - 1 > 2 * np.log(1 / (1 - ALPHA)), "upper bound needs G-1 > 2 log(1/(1-alpha))"
        E, _ = draw(reps, m, G)
        ratio = outer_radius(E, k) / r_lb(G, m)
        c = np.sqrt((m + 1) / (m - 1))
        lo = c * (np.sqrt(G - 1) - t) / (np.sqrt(G) + np.sqrt(2 * np.log(1 / ALPHA)))
        hi = c * (np.sqrt(G) + t) / (np.sqrt(G - 1) - np.sqrt(2 * np.log(1 / (1 - ALPHA))))
        frac = float(((ratio >= lo) & (ratio <= hi)).mean())
        guar = 1 - 2 * m * np.exp(-t * t / 2)
        rows.append(dict(G=G, m=m, t=t, lo=float(lo), hi=float(hi), frac_inside=frac, guaranteed=float(guar)))
        print(f"M9 G={G} m={m} t={t}: bounds [{lo:.4f}, {hi:.4f}], fraction inside {frac:.4f} >= guaranteed {max(guar,0):.4f}")
    # the condition is necessary: at G = 1 the quantile lower bound is negative
    neg = np.sqrt(0) - np.sqrt(2 * np.log(1 / (1 - ALPHA)))
    print(f"M9 G=1, alpha=0.1: quantile lower bound sqrt(G-1) - sqrt(2 log(1/(1-alpha))) = {neg:.3f} < 0, so the upper bound is not asserted there")
    ok = all(r["frac_inside"] >= r["guaranteed"] for r in rows) and neg < 0
    RES["M9"] = dict(ok=bool(ok), rows=rows, G1_quantile_lower_bound=float(neg))
    print(f"[{'PASS' if ok else 'FAIL'}] M9  (Thm 2(iii): empirical frequency inside the two-sided bound >= 1 - 2m exp(-t^2/2))\n")


# ---------------- M7: k = m -> volume of C is NOT minimax ----------------
def m7():
    # alpha = 0.1, m = 9: k = ceil(0.9*10) = 9 = m, so C = union of the m LOO balls and (vol C/omega_G)^{1/G} >= max_i r_i
    m = 9
    k = k_of(m)
    assert k == m
    rows = []
    for G in (200, 1000, 5000):
        reps = 400 if G <= 1000 else 100
        E, _ = draw(reps, m, G)
        d = E - E.mean(1, keepdims=True)
        rmax = m / (m - 1) * np.linalg.norm(d, axis=-1).max(1)
        ratio = rmax / r_lb(G, m)
        rows.append(dict(G=G, maxr_over_rlb=float(ratio.mean()), sd=float(ratio.std())))
        print(f"M7 G={G:5d} m={m} k={k}=m:  max_i r_i / r_LB = {ratio.mean():.4f} (sd {ratio.std():.4f}); limit m/sqrt(m^2-1) = {m/np.sqrt(m*m-1):.4f}")
    ok = abs(rows[-1]["maxr_over_rlb"] - m / np.sqrt(m * m - 1)) < 0.01 and rows[-1]["maxr_over_rlb"] > 1
    RES["M7"] = dict(ok=bool(ok), rows=rows, limit=float(m / np.sqrt(m * m - 1)))
    print(f"[{'PASS' if ok else 'FAIL'}] M7  ((vol C/omega_G)^(1/G) >= max r_i > r_LB: the exact region is not volume-minimax at k = m)\n")


if __name__ == "__main__":
    t0 = time.time()
    m1(); m2(); m3(); m4(); m5(); m6(); m7(); m8(); m9()
    RES["_meta"] = dict(seed=20261002, alpha=ALPHA, seconds=round(time.time() - t0, 1))
    OUT.mkdir(exist_ok=True)
    (OUT / "sim_minimax.json").write_text(json.dumps(RES, indent=1))
    print(f"all pass: {all(v["ok"] for kk, v in RES.items() if not kk.startswith('_'))}   ({RES['_meta']['seconds']}s)")
