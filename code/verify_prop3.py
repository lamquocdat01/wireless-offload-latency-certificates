"""Numerical check of Prop 3 (burst-tax identity) and Cor 3.1 on the corpus. seed 42.

(1) Identity Q(alpha, c) = c (1 + sigma_alpha(1/c)) on random empirical envelopes and on the
    Thm-1 curves (float equality up to 1e-9); c <= 1 => Q = c for every envelope.
(2) Cor 3.1: periodic at rate rho <= 1/c has sigma(1/c) < 1; on the activation corpus (121 videos,
    25 (K, a) cells) we report sigma_hat(1/c) of gate+refresh (pure), token-bucket b = 4 and the
    same-rate periodic baseline for c in C_GRID (c = 1.07 and 2.17 are the measured 720p/1080p
    q75 values on 5G SA low-band median UL, results/p1_c_estimate.json).
Output: results/verify_prop3.json, results/p1_burst_tax.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from nc_bounds import alpha_tb, q_term, sigma, sigma_tb, q_tb
from paths import DERIVED, RESULTS

SEED = 42
K_GRID = (2, 4, 8, 16, 32)
A_GRID = (0.05, 0.1, 0.2, 0.3, 0.5)
C_GRID = (1.07, 1.25, 1.5, 2.17, 3.0)
POLICIES = ("gate_refresh", "token_bucket_b4", "periodic_same_rate")


def sigma_seq(opens_by_seq: list[np.ndarray], r: float) -> float:
    """sigma_hat(r) of the pooled empirical envelope = max over sequences of the max sub-array sum
    of (o_t - r), empty allowed (exact, no window grid)."""
    best = 0.0
    for o in opens_by_seq:
        cs = np.r_[0.0, np.cumsum(o - r)]
        best = max(best, float(np.max(cs - np.minimum.accumulate(cs))))
    return best


def check_identity(rng: np.random.Generator) -> dict:
    max_err, n, c_le1_bad = 0.0, 0, 0
    for _ in range(400):
        L = int(rng.integers(50, 600))
        o = (rng.random(L) < rng.uniform(0.05, 0.9)).astype(np.int8)
        cs = np.r_[0, np.cumsum(o)]
        al = np.array([0] + [int(np.max(cs[w:] - cs[:-w])) for w in range(1, L + 1)], float)
        for c in rng.uniform(0.2, 4.0, 6):
            lhs = q_term(al, c)
            rhs = c * (1.0 + sigma(al, 1.0 / c))
            max_err = max(max_err, abs(lhs - rhs))
            n += 1
            if c <= 1 and abs(lhs - c) > 1e-9:
                c_le1_bad += 1
    tb_err = 0.0
    for K in K_GRID:
        for a in A_GRID:
            for b in (1 + 2 * a, 4.0):
                for c in (0.5, 1.0, 1.2, 1.5):
                    if (1 / K + a) * c >= 1:
                        continue
                    W = 4000
                    al = alpha_tb(np.arange(W + 1), K, a, b).astype(float)
                    tb_err = max(tb_err, abs(q_term(al, c) - q_tb(K, a, b, c)))
    return {"random_envelopes_checks": n, "max_abs_err": max_err, "c_le_1_Q_ne_c": c_le1_bad,
            "thm1_curves_max_abs_err_vs_long_grid": tb_err}


def corpus_burst_tax() -> dict:
    cells = []
    for K in K_GRID:
        for a in A_GRID:
            act = pd.read_parquet(DERIVED / f"activations_K{K}_a{a}.parquet")
            vids = [g for _, g in act.groupby(["dataset", "video"], sort=False)]
            row = {"K": K, "a": a}
            for pol in POLICIES:
                ob = [v[f"open_{pol}"].to_numpy().astype(float) for v in vids]
                rate = float(sum(o.sum() for o in ob) / sum(len(o) for o in ob))
                row[pol] = {"rate": rate, "sigma_hat": {}, "Q_hat": {}}
                for c in C_GRID:
                    s = sigma_seq(ob, 1.0 / c)
                    row[pol]["sigma_hat"][str(c)] = s
                    row[pol]["Q_hat"][str(c)] = c * (1 + s)
                row[pol]["sigma_hat_own_rate"] = sigma_seq(ob, rate)
            cells.append(row)
    # Cor 3.1 check: periodic, rate <= 1/c  =>  sigma < 1
    cor_viol, cor_n = 0, 0
    for r in cells:
        for c in C_GRID:
            if r["periodic_same_rate"]["rate"] <= 1 / c:
                cor_n += 1
                cor_viol += int(r["periodic_same_rate"]["sigma_hat"][str(c)] >= 1.0)
    summ = {}
    for c in C_GRID:
        k = str(c)
        st = [r for r in cells if r["gate_refresh"]["rate"] < 1 / c]       # gate rate below 1/c
        if not st:
            continue
        g = [r["gate_refresh"]["sigma_hat"][k] for r in st]
        t = [r["token_bucket_b4"]["sigma_hat"][k] for r in st]
        p = [r["periodic_same_rate"]["sigma_hat"][k] for r in st]
        summ[k] = {"cells_gate_rate_lt_1_over_c": len(st),
                   "gate_pure_sigma_range": [min(g), max(g)],
                   "token_bucket_b4_sigma_range": [min(t), max(t)],
                   "periodic_sigma_range": [min(p), max(p)],
                   "gate_over_periodic_Q_ratio_range": [
                       min(r["gate_refresh"]["Q_hat"][k] / r["periodic_same_rate"]["Q_hat"][k] for r in st),
                       max(r["gate_refresh"]["Q_hat"][k] / r["periodic_same_rate"]["Q_hat"][k] for r in st)]}
    return {"n_videos": len(vids), "c_grid": list(C_GRID), "cells": cells,
            "cor31_periodic_cells_checked": cor_n, "cor31_violations": cor_viol, "summary": summ}


def main() -> None:
    rng = np.random.default_rng(SEED)
    idn = check_identity(rng)
    print(idn, flush=True)
    bt = corpus_burst_tax()
    json.dump(bt, open(RESULTS / "p1_burst_tax.json", "w"), indent=1)
    print(json.dumps(bt["summary"], indent=1), "cor3.1", bt["cor31_periodic_cells_checked"], bt["cor31_violations"])
    json.dump({"seed": SEED, "identity": idn,
               "cor31": {"checked": bt["cor31_periodic_cells_checked"], "violations": bt["cor31_violations"]},
               "burst_tax_summary": bt["summary"]}, open(RESULTS / "verify_prop3.json", "w"), indent=1)


if __name__ == "__main__":
    main()
