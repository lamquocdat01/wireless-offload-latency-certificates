"""Numerical check of Thm 4 (dimensioning): closed-form bucket cap, Alg 1 optimality, Prop 4.1 table
(version A: fixed gate budget a_q). seed 42 (deterministic grids).

(1) b_max_closed <= b_max_exact (closed form is sufficient) and the gap, over a scenario grid.
(2) sigma_alpha(1/c) nondecreasing in b and nonincreasing in K (monotonicity used by Alg 1).
(3) Alg 1 == brute-force minimum of rho = 1/K + a over F intersected with {a >= a_q} (grid K = 1..K_max+12,
    a step 0.01, b = 1 + 2a), and Alg 1's output is in F.
(4) Prop 4.1 (version A): rho*_link - rho*_dev and admission over (Delta, T, c, L_srv, a_q), T from
    the Prop-5 certificate of the uplink sessions (results/p1_service_curve.json).
Output: results/verify_thm4.json
"""
from __future__ import annotations

import json
import math

import numpy as np

import nc_bounds as nb
from paths import RESULTS

TAU = 1000.0 / 30.0
DELTAS = (100.0, 200.0, 300.0, 500.0, 1000.0)
C_GRID = (0.6, 1.0, 1.07, 1.25, 1.5, 2.17, 3.0)
A_Q = (0.0, 0.05, 0.1, 0.2)


def t_grid() -> list[float]:
    sc = json.load(open(RESULTS / "p1_service_curve.json"))["sessions"]
    ul = sorted(v["eps"]["0.001"]["full_roundmax"]["T_ms"] for v in sc.values() if v["direction"] == "uplink")
    return [round(ul[0], 3), round(float(np.median(ul)), 3), round(ul[-1], 3), 50.0]


def check_bmax() -> dict:
    rows = []
    for D in DELTAS:
        for T in (7.2, 20.0, 50.0):
            for Ls in (0.0, 20.0):
                for c in (1.07, 1.25, 1.5, 2.17, 3.0):
                    for K in (2, 4, 8, 16, 32):
                        for a in (0.05, 0.1, 0.2, 0.3, 0.5):
                            if (1 / K + a) * c >= 1:
                                continue
                            bc = nb.b_max_closed(D, T, Ls, c, TAU, K, a)
                            be = nb.b_max_exact(D, T, Ls, c, TAU, K, a)
                            rows.append((D, T, Ls, c, K, a, bc, be))
    bad = [r for r in rows if math.isfinite(r[6]) and r[6] > r[7] + 1e-6 and r[6] >= 0]
    gaps = np.array([r[7] - max(r[6], 0.0) for r in rows if math.isfinite(r[7]) and r[7] >= 0 and r[6] >= 0])
    return {"scenarios": len(rows), "closed_exceeds_exact": len(bad),
            "gap_exact_minus_closed": {"n": int(len(gaps)), "median": float(np.median(gaps)),
                                       "q90": float(np.quantile(gaps, .9)), "max": float(gaps.max())},
            "examples": [dict(zip(("Delta", "T", "L_srv", "c", "K", "a", "b_closed", "b_exact"), r))
                         for r in rows[:: max(1, len(rows) // 12)]]}


def check_monotone(rng) -> dict:
    vb = vk = n = 0
    for _ in range(300):
        K = int(rng.integers(2, 33)); a = float(rng.uniform(0, 0.5)); c = float(rng.uniform(1.01, 3))
        if (1 / K + a) * c >= 1:
            continue
        b1, b2 = sorted(rng.uniform(1, 12, 2))
        vb += int(nb.sigma_tb(K, a, b1, 1 / c) > nb.sigma_tb(K, a, b2, 1 / c) + 1e-9)
        vk += int(nb.sigma_tb(K + 1, a, b1, 1 / c) > nb.sigma_tb(K, a, b1, 1 / c) + 1e-9)
        n += 1
    return {"checks": n, "violations_b": vb, "violations_K": vk}


def brute_min_rho(D, T, Ls, c, a_q):
    Km = nb.k_max(D, T, Ls, c, TAU)
    best = (math.inf, None)
    for K in range(1, Km + 13):
        for a in np.round(np.arange(a_q, 0.5001, 0.01), 4):
            if nb.feasible(K, float(a), nb.b_min_tight(float(a)), D, T, Ls, c, TAU):
                rho = min(1.0, 1 / K + a)
                if rho < best[0] - 1e-12:
                    best = (rho, (K, float(a)))
    return best


def check_alg1() -> dict:
    rows, mism, infeas = [], 0, 0
    for D in (100.0, 200.0, 500.0):
        for T in (7.2, 20.0):
            for Ls in (0.0, 20.0):
                for c in (0.6, 1.07, 1.5, 2.17):
                    for a_q in (0.0, 0.05, 0.1, 0.2):
                        p = nb.alg1(D, T, Ls, c, TAU, a_q, b_rule="min")
                        br = brute_min_rho(D, T, Ls, c, a_q)
                        alg = p.rho if p.admitted else math.inf
                        ok = (alg == br[0]) or abs(alg - br[0]) < 1e-12
                        mism += int(not ok)
                        if p.admitted and not nb.feasible(p.K, p.a, p.b, D, T, Ls, c, TAU):
                            infeas += 1
                        rows.append({"Delta": D, "T": T, "L_srv": Ls, "c": c, "a_q": a_q, "alg1_rho": alg,
                                     "brute_rho": br[0], "brute_arg": br[1], "alg1_K": p.K, "admitted": p.admitted,
                                     "reason": p.reason})
    return {"scenarios": len(rows), "mismatches": mism, "alg1_output_not_in_F": infeas, "detail": rows}


def prop41_table(Ts) -> dict:
    rows = []
    for D in DELTAS:
        for T in Ts:
            for Ls in (0.0, 20.0):
                Kd = nb.k_dev(D, Ls, TAU)
                for a_q in A_Q:
                    rho_dev = nb.rho_tb(Kd, a_q)
                    for c in C_GRID:
                        p = nb.alg1(D, T, Ls, c, TAU, a_q, b_rule="max")
                        pc = nb.alg1(D, T, Ls, c, TAU, a_q, b_rule="max", exact=False)
                        rows.append({"Delta": D, "T": T, "L_srv": Ls, "a_q": a_q, "c": c, "K_dev": Kd,
                                     "K_max": p.K_max, "rho_dev": rho_dev,
                                     "rho_link": p.rho if p.admitted else None,
                                     "price": (p.rho - rho_dev) if p.admitted else None,
                                     "admitted": p.admitted, "reason": p.reason,
                                     "b_max_exact": p.b_max, "admitted_closed_form": pc.admitted,
                                     "b_max_closed": pc.b_max})
    adm = lambda f: sum(1 for r in rows if f(r))
    summ = {}
    for c in C_GRID:
        rc = [r for r in rows if r["c"] == c]
        pr = [r["price"] for r in rc if r["admitted"]]
        summ[str(c)] = {"scenarios": len(rc), "admitted": sum(r["admitted"] for r in rc),
                        "admitted_closed_form": sum(r["admitted_closed_form"] for r in rc),
                        "rejected_burst": sum(r["reason"].startswith("burst") for r in rc),
                        "rejected_unstable": sum(r["reason"].startswith("unstable") for r in rc),
                        "rejected_deadline": sum(r["reason"].startswith("deadline") for r in rc),
                        "price_median": float(np.median(pr)) if pr else None,
                        "price_max": float(max(pr)) if pr else None}
    # admission region for a gated class (a_q > 0) vs a pure refresh class (a_q = 0)
    gated = [r for r in rows if r["a_q"] > 0]
    return {"T_grid_ms": Ts, "rows": rows, "summary_by_c": summ,
            "gated_admitted_c_le_1": adm(lambda r: r["a_q"] > 0 and r["c"] <= 1 and r["admitted"]),
            "gated_c_le_1": adm(lambda r: r["a_q"] > 0 and r["c"] <= 1),
            "gated_admitted_c_gt_1": adm(lambda r: r["a_q"] > 0 and r["c"] > 1 and r["admitted"]),
            "gated_c_gt_1": adm(lambda r: r["a_q"] > 0 and r["c"] > 1),
            "n_gated": len(gated)}


def main() -> None:
    rng = np.random.default_rng(42)
    res = {"seed": 42}
    res["bmax"] = check_bmax(); print({k: v for k, v in res["bmax"].items() if k != "examples"}, flush=True)
    res["monotone"] = check_monotone(rng); print(res["monotone"], flush=True)
    res["alg1"] = check_alg1(); print({k: v for k, v in res["alg1"].items() if k != "detail"}, flush=True)
    res["prop41"] = prop41_table(t_grid())
    print(json.dumps(res["prop41"]["summary_by_c"], indent=0))
    print({k: v for k, v in res["prop41"].items() if k not in ("rows", "summary_by_c")})
    json.dump(res, open(RESULTS / "verify_thm4.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
