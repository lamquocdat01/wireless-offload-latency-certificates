"""Numerical check of Thm 2 (max-form e2e detection-latency bound), its two-branch tightness and
Ext 2.2 (general service curve). seed 42.

(1) Exactness (brute force): for small (K, a, b, c) and horizon H, enumerate EVERY gate signal,
    refresh phase and event onset; the worst e2e detection latency (radio latency = T) must equal
    T + L_srv + tau_f max{K-1+c, Q(alpha*, c)} with alpha* the exact window supremum (Thm 2 +
    tightness), and Q(alpha, c) - Q(alpha*, c) in [0, c] in the tight regime of Thm 1'.
    Each branch is also reproduced by its explicit witness (proof of tightness).
(2) Upper bound, MC: long random gate signals (Bernoulli/Markov), random radio latencies in
    [0, T], FIFO constant-rate transmitter; every caught event's e2e <= closed-form bound.
(3) Ext 2.2: general service curves (rate-latency, TDD-like staircase, two-slope) realised by the
    minimal server D = A (x) beta (computed exactly on the arrival instants); every event's e2e <=
    L_srv + max{(K-1) tau_f + beta^-1(s), max_w [beta^-1(s(alpha(w)+1)) - w tau_f]}; for
    rate-latency the bound coincides with Thm 2.
Output: results/verify_thm2.json
"""
from __future__ import annotations

import itertools
import json
import math

import numpy as np

from nc_bounds import alpha_star_tb, alpha_tb, e2e_bound_ms, q_tb, q_term, token_bucket_opens
from paths import RESULTS

SEED = 42
TAU = 1000.0 / 30.0


def lindley_finish(t: np.ndarray, c: float) -> np.ndarray:
    """FIFO constant service c per frame, arrivals at slot instants t (sorted): finish times."""
    k = np.arange(len(t))
    return (k + 1) * c + np.maximum.accumulate(t - k * c)


def worst_e2e_slots(o: np.ndarray, c: float, onsets: np.ndarray) -> float:
    """max over onsets e of (u - e) + (F_u - t_u), u = first open >= e (slots, radio latency excluded)."""
    t = np.flatnonzero(o).astype(float)
    F = lindley_finish(t, c)
    pos = np.searchsorted(t, onsets)
    ok = pos < len(t)
    if not ok.any():
        return -math.inf
    j = pos[ok]
    return float(np.max(t[j] - onsets[ok] + F[j] - t[j]))


def brute_force(K: int, a: float, b: float, c: float, H: int) -> dict:
    best = -math.inf
    onsets = np.arange(0, H - K + 1, dtype=float)
    for phase in range(K):
        nonref = [t for t in range(H) if (t - phase) % K != 0]
        for bits in itertools.product((False, True), repeat=len(nonref)):
            fire = np.zeros(H, dtype=bool)
            fire[nonref] = bits
            o = token_bucket_opens(fire, K, a, b, phase)
            best = max(best, worst_e2e_slots(o, c, onsets))
    ast = alpha_star_tb(H, K, a, b)
    wmax = H - K
    q_star = q_term(ast[:wmax + 1].astype(float), c)
    formula = max(K - 1 + c, q_star)
    al = alpha_tb(np.arange(wmax + 1), K, a, b).astype(float)
    q_cl = q_term(al, c)
    # witnesses: branch 1 = refresh at e-1, gate silent, link idle; branch 2 = greedy burst from a
    # full bucket over [e - w*, e), gate silent from e on.
    w1 = K - 1 + c
    wstar = int(np.argmax((ast[:wmax + 1] + 1.0) * c - np.arange(wmax + 1)))
    w2 = -math.inf
    for phase in range(K):
        fire = np.zeros(H, dtype=bool)
        fire[:wstar] = True
        o = token_bucket_opens(fire, K, a, b, phase)
        if int(o[:wstar].sum()) == ast[wstar]:
            w2 = max(w2, worst_e2e_slots(o, c, np.array([float(wstar)])))
    return {"K": K, "a": a, "b": round(b, 4), "c": c, "H": H, "brute_sup_slots": best,
            "formula_alpha_star_slots": formula, "exact_match": abs(best - formula) < 1e-9,
            "Q_alpha_star": q_star, "Q_alpha_closed": q_cl, "Q_gap_closed_minus_star": q_cl - q_star,
            "witness_branch1_slots": w1, "witness_branch2_slots": w2,
            "witness_branch2_ge_Q_star": w2 >= q_star - 1e-9}


def check_exactness() -> dict:
    rows = []
    for K in (2, 3, 4):
        for a in (0.2, 0.3, 0.5):
            for b in (1 + 2 * a, 3.0):
                for c in (0.6, 1.5, 2.5):
                    if (1 / K + a) * c >= 1.0 and c > 1:
                        pass      # unstable in the long run; finite-horizon sup is still well defined
                    rows.append(brute_force(K, a, b, c, H=18))
    n_ok = sum(r["exact_match"] for r in rows)
    gaps = [r["Q_gap_closed_minus_star"] / r["c"] for r in rows]
    return {"configs": len(rows), "exact_matches": n_ok,
            "witness2_ge_Q_star": sum(r["witness_branch2_ge_Q_star"] for r in rows),
            "max_Q_gap_over_c": max(gaps), "min_Q_gap_over_c": min(gaps), "detail": rows}


def markov_fire(rng, n):
    if rng.random() < 0.4:
        return rng.random(n) < rng.uniform(0.02, 0.9)
    p_on, p_off = rng.uniform(0.005, 0.2), rng.uniform(0.01, 0.3)
    s, f, u = False, np.zeros(n, dtype=bool), rng.random(n)
    for t in range(n):
        s = (u[t] < p_on) if not s else (u[t] >= p_off)
        f[t] = s
    return f


def check_mc(rng: np.random.Generator) -> dict:
    T, Lsrv, n = 7.2, 5.0, 3000
    viol, ev, ratios, cells = 0, 0, [], 0
    for K in (2, 4, 8, 16, 32):
        for a in (0.05, 0.1, 0.2, 0.3, 0.5):
            for b in (1 + 2 * a, 4.0):
                for c in (0.4, 0.9, 1.3, 1.8):
                    if (1 / K + a) * c >= 1:
                        continue
                    bound = e2e_bound_ms(K, q_tb(K, a, b, c), c, T, Lsrv, TAU)
                    cells += 1
                    for _ in range(6):
                        o = token_bucket_opens(markov_fire(rng, n), K, a, b, int(rng.integers(K)),
                                               c0=float(rng.uniform(0, b)))
                        t = np.flatnonzero(o).astype(float)
                        F = lindley_finish(t, c)
                        radio = rng.uniform(0, T, len(t))
                        e = np.sort(rng.choice(n - K, 60, replace=False)).astype(float)
                        j = np.searchsorted(t, e)
                        lat = (F[j] - e) * TAU + radio[j] + Lsrv
                        viol += int(np.sum(lat > bound + 1e-9))
                        ev += len(e)
                        ratios.append(float(lat.max() / bound))
    return {"cells": cells, "events": ev, "violations": viol, "max_ratio": max(ratios),
            "median_ratio_of_max": float(np.median(ratios))}


# --------------------------------------------------------------------------- Ext 2.2

def server_delays(inst: np.ndarray, s_bits: float, beta, tol=1e-7) -> np.ndarray:
    """Delay of the last bit of each frame through the minimal server D = A (x) beta.
    A jumps by s at each instant (left-continuous). Exact: inf over s only at arrival instants."""
    n = len(inst)
    cumA_before = np.arange(n) * s_bits            # A(inst_k) (left limit)

    def D(t):
        m = inst <= t + 1e-12
        if not m.any():
            return 0.0
        return float(min(np.min(cumA_before[m] + beta(t - inst[m])),
                         (np.count_nonzero(inst < t)) * s_bits))   # s = t term: A(t) + beta(0)
    out = np.empty(n)
    for k in range(n):
        target = (k + 1) * s_bits
        lo, hi = 0.0, 1.0
        while D(inst[k] + hi) < target - 1e-9:
            hi *= 2
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            if D(inst[k] + mid) >= target - 1e-9:
                hi = mid
            else:
                lo = mid
        out[k] = hi
    return out


def check_ext22(rng: np.random.Generator) -> dict:
    s, Lsrv = 1.0, 0.0                 # bits normalised to one frame
    R = 1.0 / (1.2 * TAU)              # c = 1.2 for the base rate
    T = 7.0
    betas = {
        "rate_latency": (lambda t: np.maximum(0.0, R * (t - T)),
                         lambda y: np.where(y > 0, T + y / R, 0.0)),
        "tdd_staircase_P5ms": (lambda t: R * 5.0 * np.floor(np.maximum(0.0, t - T) / 5.0),
                               lambda y: np.where(y > 0, T + 5.0 * np.ceil(y / (R * 5.0) - 1e-12), 0.0)),
        "two_slope": (lambda t: np.minimum(np.maximum(0.0, 2 * R * (t - T)),
                                           np.maximum(0.0, 0.8 * R * (t - T)) + 1.5),
                      lambda y: np.where(y > 0, np.maximum(T + y / (2 * R), T + (y - 1.5) / (0.8 * R)), 0.0)),
    }
    res = {}
    for name, (beta, binv) in betas.items():
        viol, ev, worst_ratio, eq_thm2 = 0, 0, 0.0, []
        for K, a, b in [(4, 0.1, 1.2), (8, 0.05, 4.0), (16, 0.2, 2.0), (2, 0.1, 1.2)]:
            W = 400
            al = alpha_tb(np.arange(W + 1), K, a, b).astype(float)
            br2 = float(np.max(binv(s * (al + 1.0)) - np.arange(W + 1) * TAU))
            bound = Lsrv + max((K - 1) * TAU + float(binv(np.array([s]))[0]), br2)
            if name == "rate_latency":
                c = s / (R * TAU)
                eq_thm2.append(abs(bound - e2e_bound_ms(K, q_term(al, c), c, T, Lsrv, TAU)))
            for _ in range(3):
                n = 260
                o = token_bucket_opens(markov_fire(rng, n), K, a, b, int(rng.integers(K)))
                t_slots = np.flatnonzero(o).astype(float)
                d = server_delays(t_slots * TAU, s, beta)
                e = np.arange(0, n - K, 3, dtype=float)
                j = np.searchsorted(t_slots, e)
                lat = (t_slots[j] - e) * TAU + d[j] + Lsrv
                viol += int(np.sum(lat > bound + 1e-6))
                ev += len(e)
                worst_ratio = max(worst_ratio, float(lat.max() / bound))
        res[name] = {"events": ev, "violations": viol, "max_ratio": worst_ratio,
                     **({"max_abs_diff_vs_thm2_ms": max(eq_thm2)} if eq_thm2 else {})}
    return res


def main() -> None:
    rng = np.random.default_rng(SEED)
    res = {"seed": SEED}
    res["exactness"] = check_exactness()
    print({k: v for k, v in res["exactness"].items() if k != "detail"}, flush=True)
    res["mc_bound"] = check_mc(rng); print(res["mc_bound"], flush=True)
    res["ext22"] = check_ext22(rng); print(res["ext22"], flush=True)
    json.dump(res, open(RESULTS / "verify_thm2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
