"""Numerical check of Thm 1 / Thm 1' (arrival curve of token-bucket(K,a,b) and its tightness). seed 42.

(1) bound, MC: random gate signals (Bernoulli, Markov on/off), random phase and initial token
    level; max window counts <= alpha(w) for all w <= 2000.
(2) bound, corpus: token-bucket outputs on the real ASMAG gate scores (activation corpus, 121
    videos, 166 589 frames) for b in {1, 1+2a, 4, 8}; alpha checked on every window length.
(3) alpha_star_tb (greedy, full bucket, all phases) == exhaustive DP over all gate signals and
    initial token levels (small windows) -> alpha_star_tb is the exact supremum alpha*.
(4) Thm 1': a <= 1/2 and b >= 1 + 2a  =>  alpha(w) - alpha*(w) in {0, 1} (sweep, w <= 2000).
(5) Remarks (counter-examples to weaker hypotheses): b = 1 and 1 + a <= b < 1 + 2a give a gap
    growing linearly in w; a > 1/2 can give gap 2.
(6) bit form: random real intervals [x, x+t) contain <= alpha_bit(t)/s opens.
Output: results/verify_thm1.json
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from nc_bounds import alpha_bit, alpha_star_tb, alpha_tb, token_bucket_opens
from paths import DERIVED, RESULTS

SEED = 42
K_GRID = (2, 4, 8, 16, 32)
A_GRID = (0.05, 0.1, 0.2, 0.3, 0.5)


def max_window(o: np.ndarray, W: int) -> np.ndarray:
    cs = np.r_[0, np.cumsum(o, dtype=np.int64)]
    n = len(o)
    return np.array([0] + [int(np.max(cs[w:] - cs[:-w])) if w <= n else int(cs[-1])
                           for w in range(1, W + 1)])


def check_mc(rng: np.random.Generator) -> dict:
    W, n, viol, checks, seqs = 2000, 2000, 0, 0, 0
    for K in K_GRID:
        for a in A_GRID:
            for b in (1.0, 1 + a, 1 + 2 * a, 4.0):
                al = alpha_tb(np.arange(W + 1), K, a, b)
                for _ in range(40):
                    if rng.random() < 0.5:
                        fire = rng.random(n) < rng.uniform(0.05, 1.0)
                    else:
                        p_on, p_off = rng.uniform(0.01, 0.3), rng.uniform(0.01, 0.3)
                        s, fire = False, np.zeros(n, dtype=bool)
                        u = rng.random(n)
                        for t in range(n):
                            s = (u[t] < p_on) if not s else (u[t] >= p_off)
                            fire[t] = s
                    o = token_bucket_opens(fire, K, a, b, int(rng.integers(K)),
                                           c0=float(rng.uniform(0, b)))
                    mw = max_window(o, W)
                    viol += int(np.sum(mw > al))
                    checks += W
                    seqs += 1
    return {"sequences": seqs, "length": n, "window_checks": checks, "violations": viol}


def check_corpus() -> dict:
    burst = json.load(open(RESULTS / "p0_burstiness.json"))
    taus = {(c["K"], c["a"]): c["tau"] for c in burst["cells"]}
    W = 2000
    viol, checks, cells = 0, 0, 0
    for K in K_GRID:
        for a in A_GRID:
            act = pd.read_parquet(DERIVED / f"activations_K{K}_a{a}.parquet",
                                  columns=["dataset", "video", "score"])
            vids = [g.score.to_numpy() for _, g in act.groupby(["dataset", "video"], sort=False)]
            for b in (1.0, 1 + 2 * a, 4.0, 8.0):
                al = alpha_tb(np.arange(W + 1), K, a, b)
                for sc in vids:
                    o = token_bucket_opens(sc > taus[(K, a)], K, a, b)
                    mw = max_window(o, min(W, len(o)))
                    viol += int(np.sum(mw > al[:len(mw)]))
                    checks += len(mw) - 1
                cells += 1
    return {"cells_K_a_b": cells, "videos": len(vids), "window_checks": checks, "violations": viol}


def exact_sup_dp(W: int, K: int, a: float, b: float) -> np.ndarray:
    """max over phases, initial token levels (grid of 41 values in [0, b] + b) and ALL gate
    signals of the number of opens in the first w slots, w = 0..W (exhaustive DP on token level)."""
    best = np.zeros(W + 1, dtype=np.int64)
    inits = sorted(set(np.round(np.r_[np.linspace(0, b, 41), b], 9)))
    for phase in range(K):
        for c0 in inits:
            states = {round(float(c0), 9): 0}
            for t in range(W):
                ref = (t - phase) % K == 0
                new: dict[float, int] = {}
                for lev, cnt in states.items():
                    c = min(b, lev + a)
                    opts = [(c, cnt + (1 if ref else 0))]
                    if not ref and c >= 1.0 - 1e-9:
                        opts.append((c - 1.0, cnt + 1))
                    for l2, n2 in opts:
                        k = round(l2, 9)
                        if new.get(k, -1) < n2:
                            new[k] = n2
                states = new
                best[t + 1] = max(best[t + 1], max(states.values()))
    return best


def check_exact_sup() -> dict:
    out, mism = [], 0
    for K in (2, 3, 4):
        for a in (0.1, 0.25, 0.4, 0.5):
            for b in (1.0, 1 + a, 1 + 2 * a, 2.5):
                W = 24
                dp = exact_sup_dp(W, K, a, b)
                gr = alpha_star_tb(W, K, a, b)
                m = int(np.sum(dp != gr))
                mism += m
                out.append({"K": K, "a": a, "b": round(b, 3), "W": W, "mismatch_w": m})
    return {"configs": len(out), "total_mismatches": mism, "detail": out}


def check_tightness() -> dict:
    W = 2000
    rows, worst = [], 0
    for K in range(2, 9):
        for a in np.round(np.arange(0.02, 0.501, 0.02), 2):
            for b in (1 + 2 * a, 1 + 2 * a + 0.13, 1 + 2 * a + 0.5, 4.0, 8.0):
                gap = alpha_tb(np.arange(W + 1), K, a, b) - alpha_star_tb(W, K, a, b)
                assert gap.min() >= 0
                worst = max(worst, int(gap.max()))
                rows.append(int(gap.max()))
    # the P0/PROMPT grid K x a with b = 4 and b = 1 + 2a
    p0 = []
    for K in K_GRID:
        for a in A_GRID:
            for b in (1 + 2 * a, 4.0):
                gap = alpha_tb(np.arange(W + 1), K, a, b) - alpha_star_tb(W, K, a, b)
                p0.append({"K": K, "a": a, "b": b, "max_gap": int(gap.max()),
                           "share_w_exact": float(np.mean(gap[1:] == 0))})
    return {"sweep": {"K": "2..8", "a": "0.02..0.50 step 0.02", "b": "1+2a, 1+2a+0.13, 1+2a+0.5, 4, 8",
                      "W": W, "configs": len(rows), "max_gap": worst,
                      "configs_gap_0": int(sum(r == 0 for r in rows)),
                      "configs_gap_1": int(sum(r == 1 for r in rows))},
            "p0_grid": p0}


def counterexamples() -> dict:
    W = 3000
    out = []
    for K, a, b, why in [(4, 0.3, 1.0, "b = 1"), (2, 0.4, 1.4, "b = 1 + a"), (2, 0.4, 1.5, "1+a < b < 1+2a"),
                         (2, 0.45, 1.45, "b = 1 + a"), (3, 0.3, 1.3, "b = 1 + a"),
                         (2, 0.4, 1.8, "b = 1 + 2a (control)"), (3, 0.52, 2.04, "a > 1/2, b = 1+2a")]:
        gap = alpha_tb(np.arange(W + 1), K, a, b) - alpha_star_tb(W, K, a, b)
        out.append({"K": K, "a": a, "b": b, "case": why, "gap_at_w": {str(w): int(gap[w]) for w in
                                                                      (100, 500, 1000, 2000, 3000)},
                    "max_gap": int(gap.max()),
                    "slope_alpha": 1 / K + a,
                    "slope_alpha_star": float((alpha_star_tb(W, K, a, b)[W] - alpha_star_tb(W, K, a, b)[W // 2])
                                              / (W - W // 2))})
    return {"cases": out,
            "analysis_K2_a0.4_b1.5": "greedy cycle of 8 slots: 4 refresh + 3 gate opens, 0.2 token lost "
                                     "at the cap -> long-run rate 7/8 < 1/K + a = 0.9; gap ~ 0.025 w"}


def check_bit_form(rng: np.random.Generator) -> dict:
    tau, s = 1000 / 30, 1.0
    viol, n = 0, 0
    for K, a, b in [(2, 0.2, 1.4), (8, 0.1, 4.0), (16, 0.5, 2.0), (4, 0.3, 1.0)]:
        o = token_bucket_opens(np.ones(3000, dtype=bool), K, a, b, 0)
        inst = np.flatnonzero(o) * tau            # opening instants [ms]
        for _ in range(4000):
            x = rng.uniform(0, 2000 * tau)
            t = rng.uniform(0, 300 * tau) if rng.random() < 0.7 else rng.integers(1, 300) * tau
            cnt = int(np.sum((inst >= x) & (inst < x + t)))
            viol += int(cnt > alpha_bit(t, s, tau, K, a, b))
            n += 1
    return {"intervals": n, "violations": viol}


def main() -> None:
    rng = np.random.default_rng(SEED)
    res = {"seed": SEED}
    res["mc_bound"] = check_mc(rng); print("mc", res["mc_bound"], flush=True)
    res["corpus_bound"] = check_corpus(); print("corpus", res["corpus_bound"], flush=True)
    res["exact_sup_dp"] = check_exact_sup(); print("dp mismatches", res["exact_sup_dp"]["total_mismatches"], flush=True)
    res["tightness"] = check_tightness(); print("tight", res["tightness"]["sweep"], flush=True)
    res["counterexamples"] = counterexamples()
    for c in res["counterexamples"]["cases"]:
        print("cex", c["K"], c["a"], c["b"], c["case"], c["gap_at_w"], flush=True)
    res["bit_form"] = check_bit_form(rng); print("bit", res["bit_form"], flush=True)
    json.dump(res, open(RESULTS / "verify_thm1.json", "w"), indent=1)


if __name__ == "__main__":
    main()
