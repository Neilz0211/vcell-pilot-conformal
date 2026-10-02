"""Adversarial numerical checks of theory.md (draft v1) for discussion_round3.md.  numpy only, CPU, ~1-2 min.

usage (from virtual_cell/):  uv run python src/theory_checks.py        -> prints PASS/FAIL lines, writes results/theory_checks.json

Conventions (full conformal, m pilots, one test point, k = ceil((1-alpha)(m+1))):
  C = { z : #{ i <= m : S_i(z) < S_{m+1}(z) } <= k-1 }   (ties favour inclusion; closed balls)
    = { z : z lies in at least m+1-k closed LOO balls B(b_{-i}, r_i) }.
Checks
  A1 identity   ||z-mu||^2 - ||e_i-mu||^2 = ((m-1)/(m+1)) (||z-b_{-i}||^2 - r_i^2), m >= 2
  A2 membership full CP == depth >= m+1-k, incl. tie-heavy discrete residuals and m = 2
  A3 coverage   P(e_{m+1} in C) >= 1-alpha; for continuous laws == k/(m+1) up to MC error (detects an off-by-one k)
  A4 sandwich   Ball(ebar, ((m-1)/m) R_(k)) subset C subset Ball(ebar, ((m+1)/m) R_(k)); C subset minmax union; m = 2 sharpness;
                k = m+1 => C = R^G
  B1 minmax     union_i Ball(mu_{-i}, R_(k)) has coverage >= 1-alpha for non-linear symmetric adapters (median, trimmed mean)
  C1 moments    E||eps_new - epsbar_m||^2 = (1+1/m) tr Sigma (correlated, non-Gaussian eps); E r_i^2 = (m/(m-1)) tr Sigma;
                RMS break-even m > 1/rho; certified-ball break-even rho > (3m+1)/(m(m-1))
  C2 quantile   quantile-ratio lemma: Q_{1-a}(||X||)/Q_{1-a}(||Y||) within [(1-eta)/(1+eta), (1+eta)/(1-eta)] x RMS ratio
  D1 two-point  coupling bound: if both laws covered at >= 1-alpha then P(diam >= ||delta||) >= 1-2alpha (and a small-volume
                two-ball set shows the bound is about diameter, not volume); D2 ball-family volume bound E vol(C) >= (1-a) vol(B_Delta)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "results"
RES: dict = {}


def check(name: str, ok: bool, detail: str) -> None:
    RES[name] = {"pass": bool(ok), "detail": detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def k_of(m: int, alpha: float) -> int:
    return int(np.ceil((1 - alpha) * (m + 1)))


# ---------- full conformal / LOO primitives (batched: E reps x m x G, Z reps x n x G) ----------
def full_cp_member(E, Z, k):
    m = E.shape[-2]
    mu = (E.sum(-2)[..., None, :] + Z) / (m + 1)                       # reps x n x G
    S_i = np.linalg.norm(E[..., None, :, :] - mu[..., :, None, :], axis=-1)  # reps x n x m
    S_t = np.linalg.norm(Z - mu, axis=-1)                               # reps x n
    return (S_i < S_t[..., None]).sum(-1) <= k - 1


def loo(E):
    m = E.shape[-2]
    B = (E.sum(-2)[..., None, :] - E) / (m - 1)                         # b_{-i}
    r = np.linalg.norm(E - B, axis=-1)                                  # r_i = ||e_i - b_{-i}||
    return B, r


def depth_member(E, Z, k):
    m = E.shape[-2]
    B, r = loo(E)
    inside = (np.linalg.norm(Z[..., :, None, :] - B[..., None, :, :], axis=-1) <= r[..., None, :]).sum(-1)
    return inside >= m + 1 - k


def draw(rng, law, shape):
    if law == "gauss":
        return rng.normal(size=shape)
    if law == "heavy":                                                   # scale mixture, heavy tails
        return rng.normal(size=shape) * np.exp(rng.normal(0, 0.7, shape[:-1] + (1,)))
    if law == "t3":
        return rng.standard_t(3, size=shape)
    if law == "ternary":                                                 # integers -> many exact ties
        return rng.integers(-1, 2, size=shape).astype(float)
    raise ValueError(law)


# ---------- A1 identity ----------
def check_a1(rng):
    worst = 0.0
    for _ in range(2000):
        m = int(rng.integers(2, 41)); G = int(rng.integers(1, 31))
        E = draw(rng, "heavy", (m, G)); z = draw(rng, "heavy", (1, G))
        mu = (E.sum(0) + z[0]) / (m + 1)
        lhs = np.linalg.norm(z[0] - mu) ** 2 - np.linalg.norm(E - mu, axis=1) ** 2
        B, r = loo(E)
        rhs = (m - 1) / (m + 1) * (np.linalg.norm(z[0] - B, axis=1) ** 2 - r ** 2)
        worst = max(worst, float(np.max(np.abs(lhs - rhs)) / (1 + np.max(np.abs(lhs)))))
    check("A1_identity", worst < 1e-10, f"max relative error {worst:.1e} over 2000 instances, m in [2,40], G in [1,30]")


# ---------- A2 membership equivalence ----------
def check_a2(rng):
    bad = 0; n_tot = 0
    for law in ("gauss", "heavy", "ternary"):
        for m in (2, 3, 5, 9, 10, 20, 50):
            for alpha in (0.1, 0.2, 0.5):
                k = k_of(m, alpha); G = 3 if law == "ternary" else int(rng.integers(1, 20))
                E = draw(rng, law, (200, m, G))
                Z = draw(rng, law, (200, 30, G)) * rng.uniform(0.2, 3, (200, 30, 1))
                a = full_cp_member(E, Z, k); b = depth_member(E, Z, k)
                bad += int((a != b).sum()); n_tot += a.size
    check("A2_fullCP_eq_depth", bad == 0, f"{bad}/{n_tot} membership mismatches (laws gauss/heavy/ternary(ties), m in {{2..50}}, alpha in {{.1,.2,.5}})")


# ---------- A3 coverage ----------
def check_a3(rng):
    rows = []; ok = True
    for law in ("gauss", "heavy", "t3", "ternary"):
        for m in (2, 5, 9, 10, 20, 50):
            for alpha in (0.1, 0.5):
                k = k_of(m, alpha); reps = 20000; G = 3 if law == "ternary" else 10
                E = draw(rng, law, (reps, m, G)); z = draw(rng, law, (reps, 1, G))
                cov = float(full_cp_member(E, z, k).mean()); se = float(np.sqrt(cov * (1 - cov) / reps))
                target = k / (m + 1)
                valid = cov >= 1 - alpha - 3 * max(se, 1e-9)
                exact = (law == "ternary") or abs(cov - target) <= 3.5 * max(se, 1e-9) + 1e-9
                ok &= valid and exact
                rows.append(dict(law=law, m=m, alpha=alpha, k=k, cov=round(cov, 4), se=round(se, 4), k_over_m1=round(target, 4), valid=valid, exact=exact))
    worst = min(r["cov"] - (1 - r["alpha"]) for r in rows)
    check("A3_coverage", ok, f"{len(rows)} configs; all cov >= 1-alpha-3SE and (continuous laws) |cov - k/(m+1)| <= 3.5SE; min(cov-(1-alpha)) = {worst:+.4f}")
    RES["A3_rows"] = rows


# ---------- A4 sandwich ----------
def check_a4(rng):
    inner_bad = outer_bad = union_bad = 0; n_in = n_mem = 0; max_ratio = 0.0; rk_mismatch = 0.0
    for _ in range(600):
        m = int(rng.integers(2, 41)); G = int(rng.integers(1, 12)); alpha = float(rng.choice([0.1, 0.2, 0.5]))
        k = k_of(m, alpha)
        if k > m:
            continue
        E = draw(rng, rng.choice(["gauss", "heavy"]), (m, G)); ebar = E.mean(0); d = E - ebar
        dn = np.linalg.norm(d, axis=1); dk = np.sort(dn)[k - 1]
        B, r = loo(E); Rk = np.sort(r)[k - 1]
        rk_mismatch = max(rk_mismatch, abs(Rk - m / (m - 1) * dk) / Rk)
        inner, outer = (m - 1) / m * Rk, (m + 1) / m * Rk
        dirs = rng.normal(size=(400, G)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        rad = outer * rng.uniform(0.0, 1.3, (400, 1)); U = ebar + dirs * rad
        # bias some probes to the ball centres' directions (where the outer bound is nearest to being attained)
        c = B - ebar; cn = np.linalg.norm(c, axis=1, keepdims=True) + 1e-12
        U2 = ebar + (c / cn)[rng.integers(0, m, 200)] * outer * rng.uniform(0.9, 1.05, (200, 1))
        U = np.vstack([U, U2]); un = np.linalg.norm(U - ebar, axis=1)
        mem = full_cp_member(E[None], U[None], k)[0]
        inner_bad += int((~mem[un <= inner]).sum()); n_in += int((un <= inner).sum())
        outer_bad += int(mem[un > outer + 1e-9].sum()); n_mem += int(mem.sum())
        if mem.any():
            max_ratio = max(max_ratio, float(un[mem].max() / Rk))
        dmin = np.linalg.norm(U[:, None, :] - B[None, :, :], axis=2).min(1)
        union_bad += int((mem & (dmin > Rk + 1e-9)).sum())
    check("A4a_inner_ball_subset_C", inner_bad == 0, f"{inner_bad}/{n_in} inner-ball probes outside C")
    check("A4b_C_subset_outer_ball", outer_bad == 0, f"{outer_bad}/{n_mem} members outside the outer ball; max ||u||/R_(k) seen among members = {max_ratio:.4f} (bound (m+1)/m)")
    check("A4c_C_subset_minmax_union", union_bad == 0, f"{union_bad} members outside union_i Ball(b_-i, R_(k))")
    check("A4d_Rk_eq_m_over_m-1_dk", rk_mismatch < 1e-12, f"max rel |R_(k) - (m/(m-1)) d_(k)| = {rk_mismatch:.1e}")
    # m = 2, alpha = 0.5: k = 2, need >= 1 ball; C = Ball(e1, D) u Ball(e2, D); both sandwich radii attained
    D = 1.0; E = np.array([[0.0, 0.0], [D, 0.0]]); ebar = E.mean(0); k = k_of(2, 0.5); Rk = np.sort(loo(E)[1])[k - 1]
    pts = np.array([ebar + [-1.5 * D, 0], ebar + [-1.5 * D - 1e-6, 0], ebar + [0, 0.5 * D], ebar + [0, 0.5 * D + 0.3 * D]])
    mem = full_cp_member(E[None], pts[None], k)[0]
    check("A4e_m2_sharp", bool(mem[0] and not mem[1] and mem[2] and Rk == D), f"m=2: outer radius 1.5 R_(2) attained ({mem[0]}), beyond it excluded ({not mem[1]}), inner-ball boundary inside ({mem[2]}); R_(2)={Rk}")
    # k = m+1 (m+1 < 1/alpha): C is everything
    E = draw(rng, "gauss", (5, 4)); Z = draw(rng, "gauss", (1, 50, 4)) * 50
    check("A4f_k_eq_m+1_trivial", bool(full_cp_member(E[None], Z, k_of(5, 0.1)).all()), f"m=5, alpha=0.1: k={k_of(5, 0.1)}=m+1, need >= 0 balls, all 50 far probes are members")


# ---------- B1 minmax union for non-linear symmetric adapters ----------
def check_b1(rng):
    rows = []; ok = True

    def adapters(name, E):                      # returns LOO offsets b_{-i}: reps x m x G
        reps, m, G = E.shape
        out = np.empty_like(E)
        for i in range(m):
            Ei = np.delete(E, i, axis=1)
            if name == "median":
                out[:, i] = np.median(Ei, axis=1)
            else:                               # 20% trimmed mean per coordinate
                s = np.sort(Ei, axis=1); t = int(0.2 * (m - 1))
                out[:, i] = s[:, t:m - 1 - t].mean(1)
        return out

    for name in ("median", "trimmed"):
        for law in ("gauss", "heavy"):
            for m in (10, 20):
                alpha = 0.1; k = k_of(m, alpha); reps = 6000; G = 8
                E = draw(rng, law, (reps, m, G)) + 0.5; z = draw(rng, law, (reps, G)) + 0.5
                B = adapters(name, E); r = np.linalg.norm(E - B, axis=-1); Rk = np.sort(r, axis=1)[:, k - 1]
                dist = np.linalg.norm(z[:, None, :] - B, axis=-1)
                cov_union = float((dist.min(1) <= Rk).mean())
                cov_depth = float(((dist <= r).sum(1) >= m + 1 - k).mean())
                se = np.sqrt(cov_union * (1 - cov_union) / reps)
                v = cov_union >= 1 - alpha - 3 * se; ok &= v
                rows.append(dict(adapter=name, law=law, m=m, cov_minmax_union=round(cov_union, 4), cov_jk_plus_depth=round(cov_depth, 4), valid=v))
    check("B1_minmax_union_nonlinear", ok, "; ".join(f"{r['adapter']}/{r['law']}/m={r['m']}: union {r['cov_minmax_union']} depth {r['cov_jk_plus_depth']}" for r in rows))
    RES["B1_rows"] = rows


# ---------- C1 moments ----------
def check_c1(rng):
    G = 40; A = rng.normal(size=(G, G)) * np.exp(rng.normal(0, 1, (1, G)))   # anisotropic, correlated
    Sigma = A @ A.T / G; L = np.linalg.cholesky(Sigma + 1e-9 * np.eye(G)); tr = float(np.trace(Sigma))
    rows = []; ok = True
    for m in (2, 5, 20, 50):
        reps = 20000                                                         # ~330 MB peak at m = 50
        eps = (rng.laplace(size=(reps, m + 1, G)) / np.sqrt(2)) @ L.T       # non-Gaussian, Cov = Sigma
        new, pil = eps[:, -1], eps[:, :-1]
        lhs = float(np.mean(np.linalg.norm(new - pil.mean(1), axis=1) ** 2))
        pred = (1 + 1 / m) * tr
        B, r = loo(pil); loo_e = float(np.mean(r ** 2)); loo_pred = m / (m - 1) * tr
        e1 = abs(lhs / pred - 1); e2 = abs(loo_e / loo_pred - 1); ok &= e1 < 0.02 and e2 < 0.02
        rows.append(dict(m=m, E_adapt_err2=round(lhs, 4), pred=round(pred, 4), rel_err=round(e1, 4), E_loo2=round(loo_e, 4), loo_pred=round(loo_pred, 4), loo_rel_err=round(e2, 4)))
    check("C1a_second_moment_identity", ok, "; ".join(f"m={r['m']}: adapt rel err {r['rel_err']:.3f}, LOO rel err {r['loo_rel_err']:.3f}" for r in rows))
    RES["C1_rows"] = rows
    # break-even algebra: RMS ratio^2 = (1+1/m)/(1+rho) < 1 iff m > 1/rho ; certified outer ball ((m+1)/m)^2 (m/(m-1)) /(1+rho) < 1 iff rho > (3m+1)/(m(m-1))
    ok2 = True; txt = []
    for m in (5, 10, 20, 50, 100):
        for rho in np.linspace(0.005, 0.6, 400):
            rms_gain = (1 + 1 / m) / (1 + rho) < 1
            cert_gain = ((m + 1) / m) ** 2 * (m / (m - 1)) / (1 + rho) < 1
            ok2 &= rms_gain == (m > 1 / rho); ok2 &= cert_gain == (rho > (3 * m + 1) / (m * (m - 1)))
        txt.append(f"m={m}: rho*_RMS={1 / m:.3f}, rho*_cert={(3 * m + 1) / (m * (m - 1)):.3f}")
    check("C1b_breakeven", ok2, "; ".join(txt))


# ---------- C2 quantile-ratio lemma ----------
def check_c2(rng):
    alpha = 0.1; rows = []; ok = True
    for name, Sig in (("iso_G200", np.eye(200)), ("lowdeff_G200", None)):
        G = 200
        if Sig is None:
            ev = np.exp(-np.arange(G) / 8.0); Sig = np.diag(ev)                  # effective dimension ~ 16
        L = np.sqrt(np.diag(Sig)); tr = float(np.trace(Sig)); deff = tr ** 2 / float((np.diag(Sig) ** 2).sum())
        for rho in (0.085, 0.24):
            m = 50; k = k_of(m, alpha); reps = 400                               # memory: keep every array < 50 MB
            delta = rng.normal(size=G); delta *= np.sqrt(rho * tr) / np.linalg.norm(delta)
            eps = rng.normal(size=(reps, m, G)) * L
            s_un, s_ad = [], []
            for b in range(reps):                                                # 200 test points per pilot draw
                new = rng.normal(size=(200, G)) * L
                s_un.append(np.linalg.norm(delta + new, axis=-1))                # unadapted test scores
                s_ad.append(np.linalg.norm(new - eps[b].mean(0), axis=-1))       # adapted (m pilots) test scores
            s_un, s_ad = np.concatenate(s_un), np.concatenate(s_ad)
            q_un, q_ad = np.quantile(s_un, 1 - alpha), np.quantile(s_ad, 1 - alpha)
            rms_ratio = np.sqrt((1 + 1 / m) / (1 + rho))
            obs = q_ad / q_un
            # eta at level beta = alpha/2 for both laws: P(| ||X|| - rms | > eta rms) <= beta
            def eta_of(s):
                rms = np.sqrt(np.mean(s ** 2)); dev = np.abs(s / rms - 1); return float(np.quantile(dev, 1 - alpha / 2))
            eta = max(eta_of(s_un), eta_of(s_ad))
            lo, hi = (1 - eta) / (1 + eta) * rms_ratio, (1 + eta) / (1 - eta) * rms_ratio
            # finite-sample certified outer ball vs pilot-only split radius (what transfer_matrix.md calls "exact gain")
            q_pilot = np.sort(np.linalg.norm(delta + eps, axis=-1), axis=1)[:, k - 1]
            Rk = np.sort(loo(eps)[1], axis=1)[:, k - 1]; q_out = (m + 1) / m * Rk
            gain_obs = float(np.mean(q_out) / np.mean(q_pilot) - 1)
            gain_pred = (m + 1) / m * np.sqrt((m / (m - 1)) / (1 + rho)) - 1
            inb = lo <= obs <= hi; ok &= inb
            rows.append(dict(cov=name, d_eff=round(deff, 1), rho=rho, m=m, quantile_ratio_obs=round(float(obs), 4), rms_ratio=round(float(rms_ratio), 4),
                             eta=round(eta, 4), bound=[round(float(lo), 4), round(float(hi), 4)], in_bound=inb,
                             certified_gain_obs=round(gain_obs, 4), certified_gain_pred=round(float(gain_pred), 4)))
    check("C2_quantile_lemma", ok, "; ".join(f"{r['cov']} rho={r['rho']}: obs {r['quantile_ratio_obs']} vs RMS {r['rms_ratio']} in [{r['bound'][0]},{r['bound'][1]}] (eta={r['eta']}, d_eff={r['d_eff']}); certified gain obs {r['certified_gain_obs']:+.3f} pred {r['certified_gain_pred']:+.3f}" for r in rows))
    RES["C2_rows"] = rows


# ---------- D two-point / ball-family impossibility ----------
def check_d(rng):
    alpha = 0.1; G = 2; reps = 200000
    eps = rng.normal(size=(reps, G)); delta = np.array([3.0, 0.0]); nd = np.linalg.norm(delta)
    ok = True; txt = []
    # (i) a ball of radius q around f (any data-based centre has the same law under both; take centre 0 for simplicity)
    n0 = np.linalg.norm(eps, axis=1); n1 = np.linalg.norm(eps + delta, axis=1)      # ||y-f|| under P_0 / P_delta (coupled on eps)
    n_valid = 0
    for q in (1.0, 1.5, 2.15, 3.0, 5.2, 6.0):
        c0 = float((n0 <= q).mean()); c1 = float((n1 <= q).mean()); both = float(((n0 <= q) & (n1 <= q)).mean())
        cond = (c0 >= 1 - alpha) and (c1 >= 1 - alpha)
        if cond:                      # theorem: P(both covered) >= 1-2alpha, and on that event diam >= ||delta||; a ball has diam = 2q
            n_valid += 1; ok &= (2 * q >= nd) and (both >= 1 - 2 * alpha - 0.01)
        txt.append(f"ball q={q}: cov0 {c0:.3f} cov_delta {c1:.3f} both {both:.3f} valid_both={cond} diam/||delta||={2 * q / nd:.2f}")
    ok &= n_valid >= 1
    # (ii) adversarial small-volume set S = Ball(f, r) u Ball(f+delta, r): valid under both laws, volume far below the enclosing
    #      ball's, but diameter ||delta|| + 2r >= ||delta|| -> the two-point bound is about diameter / enclosing radius, not volume
    r = 2.15
    in_S0 = (n0 <= r) | (np.linalg.norm(eps - delta, axis=1) <= r)      # P_0: y - f = eps
    in_S1 = (n1 <= r) | (n0 <= r)                                       # P_delta: y - f = delta + eps
    c0 = float(in_S0.mean()); c1 = float(in_S1.mean())
    vol_two = 2 * np.pi * r ** 2; vol_big = np.pi * (nd / 2 + r) ** 2
    txt.append(f"two-ball r={r}: cov0 {c0:.3f} cov_delta {c1:.3f}, diam {nd + 2 * r:.2f} >= ||delta||={nd}, vol {vol_two:.1f} < enclosing-ball vol {vol_big:.1f} -> bound is on diameter, not volume")
    ok &= c0 >= 1 - alpha and c1 >= 1 - alpha and (nd + 2 * r) >= nd
    check("D1_two_point_diameter", ok, " | ".join(txt))
    # (iii) ball family: delta ~ Unif(B_Delta); any set valid for all delta in B_Delta has E vol(C) >= (1-alpha) vol(B_Delta)
    Delta = 4.0; rs = np.linspace(2.0, 6.0, 41); found = None
    u = rng.normal(size=(reps, G)); u /= np.linalg.norm(u, axis=1, keepdims=True); dl = u * Delta * np.sqrt(rng.uniform(size=(reps, 1)))
    for rr in rs:                     # worst case over delta in B_Delta is the boundary; the uniform average is weaker, still implies the bound
        cov_worst = float((np.linalg.norm(eps + np.array([Delta, 0.0]), axis=1) <= rr).mean())
        if cov_worst >= 1 - alpha:
            found = rr; break
    volC, volB = np.pi * found ** 2, np.pi * Delta ** 2
    check("D2_ball_family_volume", volC >= (1 - alpha) * volB, f"smallest ball valid for every ||delta||<=Delta={Delta}: r={found:.1f}, vol {volC:.1f} >= (1-alpha) vol(B_Delta) = {(1 - alpha) * volB:.1f}")


if __name__ == "__main__":
    t0 = time.time(); rng = np.random.default_rng(20261002)
    check_a1(rng); check_a2(rng); check_a3(rng); check_a4(rng); check_b1(rng); check_c1(rng); check_c2(rng); check_d(rng)
    n_fail = sum(1 for v in RES.values() if isinstance(v, dict) and v.get("pass") is False)
    OUT.mkdir(exist_ok=True); (OUT / "theory_checks.json").write_text(json.dumps(RES, indent=1, default=float))
    print(f"THEORY_CHECKS_DONE fails={n_fail} elapsed={time.time() - t0:.0f}s", flush=True)
