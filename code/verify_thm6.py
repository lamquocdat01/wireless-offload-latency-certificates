"""Numerical check of Thm 6, Lemma 6.1, Thm 4', Prop 4.2(i) and Prop 7 (topic #19, P2). seed 42.

(1) Lemma 6.1: P_frame(N, r, p) nonincreasing in r (indep and GE), N in {35,100,300}, r <= N.
(2) Thm 6 domination: heterogeneous independent bad probabilities p_i <= p -> MC frequency of
    #bad > r <= P(Bin(N + r, p) > r) (10^6 frames per cell).
(3) Prop 7(a): exact GE dynamic programme == MC (10^6 frames per cell), incl. the fitted parameters
    of DET6G sessions (results/p2_fec_sensitivity.json) when available.
(4) Prop 7(b): GE MLE recovers known parameters on synthetic 10^6-packet sequences.
(5) Thm 4'(i): F(c) nonincreasing in c (random (K,a,b,Delta,T,c1<c2)); Alg 1' == brute force over
    (K, a, r) at fixed b = 1 + 2a.
(6) Prop 4.2(i): over a grid, min rho in F' is attained at a = 0.
Output: results/verify_thm6.json
"""
from __future__ import annotations

import json
import math

import numpy as np

import fec
import nc_bounds as nb
from paths import RESULTS

SEED = 42
TAU = 1000.0 / 30.0


def check_monotone() -> dict:
    viol, n = 0, 0
    for N in (35, 100, 300):
        for p in (5e-4, 2.3e-3, 1e-2, 5e-2):
            v = [fec.p_frame_indep(N, r, p) for r in range(0, 40)]
            viol += int(np.sum(np.diff(v) > 1e-15)); n += len(v) - 1
            g = fec.GE(p / 0.5 * 0.05, 0.05, 0.5)
            v = [fec.p_frame_ge(g, N, r) for r in range(0, 25)]
            viol += int(np.sum(np.diff(v) > 1e-15)); n += len(v) - 1
    return {"checks": n, "violations": viol}


def check_domination(rng) -> dict:
    out = []
    for N, r, p in ((35, 2, 3e-3), (100, 3, 2.3e-3), (100, 1, 2.3e-3), (300, 6, 2.3e-3)):
        M, F = N + r, 1_000_000
        pi = rng.uniform(0, p, M)                         # heterogeneous, all <= p
        cnt = np.zeros(F, np.int32)
        for i in range(M):
            cnt += rng.random(F) < pi[i]
        freq = float(np.mean(cnt > r))
        out.append({"N": N, "r": r, "p": p, "mc_freq": freq, "binomial_bound": fec.p_frame_indep(N, r, p),
                    "ok": freq <= fec.p_frame_indep(N, r, p) + 3 * math.sqrt(fec.p_frame_indep(N, r, p) / F)})
    return {"cells": out, "all_ok": all(c["ok"] for c in out)}


def check_dp_vs_mc(rng) -> dict:
    cells = [(fec.GE(2e-4, 0.1, 0.5), 100, 3), (fec.GE(1e-3, 0.3, 0.8), 35, 2), (fec.GE(5e-5, 0.02, 0.3), 300, 6)]
    try:
        fs = json.load(open(RESULTS / "p2_fec_sensitivity.json"))["sessions"]
        for s in ("s1-UL", "s6-UL", "s26-UL"):
            gb = fs[s]["eps"]["0.001"]["ge_bad"]
            cells.append((fec.GE(gb["p_gb"], gb["p_bg"], gb["h"]), 100, int(fs[s]["eps"]["0.001"]["r_star"]["100"]["indep"])))
    except FileNotFoundError:
        pass
    out = []
    for g, N, r in cells:
        dp = fec.p_frame_ge(g, N, r)
        mc = fec.simulate_ge_frames(g, N, r, 1_000_000, rng)
        se = math.sqrt(max(dp, 1e-7) * (1 - dp) / 1_000_000)
        out.append({"p_gb": g.p_gb, "p_bg": g.p_bg, "h": g.h, "N": N, "r": r, "dp": dp, "mc": mc,
                    "z": (mc - dp) / se, "ok": abs(mc - dp) <= 4 * se})
    return {"frames_per_cell": 1_000_000, "cells": out, "all_ok": all(c["ok"] for c in out)}


def simulate_ge_seq(g: fec.GE, n: int, rng) -> np.ndarray:
    u, v = rng.random(n), rng.random(n)
    st = np.empty(n, np.int8)
    s = int(rng.random() < g.pi[1])
    for t in range(n):
        if t:
            s = int(u[t] < g.p_gb) if s == 0 else int(u[t] >= g.p_bg)
        st[t] = s
    return ((st == 1) & (v < g.h)).astype(np.int8)


def check_fit(rng) -> dict:
    out = []
    for g in (fec.GE(2e-4, 0.1, 0.5), fec.GE(5e-4, 0.5, 0.9), fec.GE(1.2e-3, 0.99, 0.999)):
        x = simulate_ge_seq(g, 1_000_000, rng)
        f = fec.fit_ge(x)
        out.append({"true": [g.p_gb, g.p_bg, g.h], "fit": [f.p_gb, f.p_bg, f.h],
                    "true_mean_bad": g.mean_bad, "fit_mean_bad": f.mean_bad,
                    "P_frame_true_N100_r3": fec.p_frame_ge(g, 100, 3), "P_frame_fit_N100_r3": fec.p_frame_ge(f, 100, 3)})
    return {"cells": out}


def check_thm4prime(rng) -> dict:
    inc_viol, n = 0, 0
    for _ in range(300):
        K = int(rng.integers(1, 33)); a = float(rng.uniform(0, 0.5)); b = 1 + 2 * a + float(rng.uniform(0, 6))
        D = float(rng.choice([100, 200, 500, 1000])); T = float(rng.uniform(5, 20))
        c1 = float(rng.uniform(0.3, 3)); c2 = c1 * float(rng.uniform(1, 1.5))
        if nb.feasible(K, a, b, D, T, 20, c2, TAU) and not nb.feasible(K, a, b, D, T, 20, c1, TAU):
            inc_viol += 1
        n += 1
    mism, rows = 0, 0
    for D in (200.0, 500.0):
        for c in (0.6, 1.09, 1.5):
            for N, p in ((35, 2.3e-3), (100, 2.3e-3), (300, 1.4e-3)):
                for a_q in (0.05, 0.2):
                    rs = fec.r_star_indep(N, p, 1e-3)
                    plan = nb.alg1(D, 8.0, 20, fec.c_eff(c, N, rs), TAU, a_q, b_rule="min")
                    best = (math.inf, None)
                    for r in range(0, rs + 6):
                        if fec.p_frame_indep(N, r, p) > 1e-3:
                            continue
                        ce = fec.c_eff(c, N, r)
                        for K in range(1, nb.k_max(D, 8.0, 20, ce, TAU) + 3):
                            for a in np.round(np.arange(a_q, 0.5001, 0.05), 4):
                                if nb.feasible(K, float(a), 1 + 2 * float(a), D, 8.0, 20, ce, TAU):
                                    rho = min(1.0, 1 / K + a)
                                    if (rho, r) < (best[0], best[1] if best[1] is not None else 1e9):
                                        best = (rho, r)
                    ok = (not plan.admitted and best[1] is None) or (
                        plan.admitted and abs(plan.rho - best[0]) < 1e-12 and best[1] == rs)
                    mism += int(not ok); rows += 1
    return {"inclusion_checks": n, "inclusion_violations": inc_viol, "alg1prime_scenarios": rows,
            "alg1prime_mismatches": mism}


def check_prop42() -> dict:
    bad, n = 0, 0
    for D in (100.0, 200.0, 500.0, 1000.0):
        for c in (0.6, 1.09, 1.5, 2.5):
            best = (math.inf, None)
            for K in range(1, 40):
                for a in np.round(np.arange(0, 0.5001, 0.05), 4):
                    if nb.feasible(K, float(a), max(1.0, 1 + 2 * float(a)), D, 8.0, 20, c, TAU):
                        rho = min(1.0, 1 / K + a)
                        if rho < best[0] - 1e-12:
                            best = (rho, float(a))
            if best[1] is not None:
                n += 1
                bad += int(best[1] != 0.0)
    return {"scenarios_feasible": n, "min_not_at_a0": bad}


def main() -> None:
    rng = np.random.default_rng(SEED)
    res = {"seed": SEED}
    res["lemma61_monotone"] = check_monotone(); print(res["lemma61_monotone"], flush=True)
    res["thm6_domination"] = check_domination(rng); print(res["thm6_domination"]["all_ok"], flush=True)
    res["prop7_dp_vs_mc"] = check_dp_vs_mc(rng); print([round(c["z"], 2) for c in res["prop7_dp_vs_mc"]["cells"]], flush=True)
    res["prop7_fit_recovery"] = check_fit(rng); print(res["prop7_fit_recovery"], flush=True)
    res["thm4prime"] = check_thm4prime(rng); print(res["thm4prime"], flush=True)
    res["prop42"] = check_prop42(); print(res["prop42"], flush=True)
    json.dump(res, open(RESULTS / "verify_thm6.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
