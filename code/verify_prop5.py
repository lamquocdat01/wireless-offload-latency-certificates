"""Numerical check of Prop 5 (Clopper-Pearson order-statistic certificate + loss budget). seed 42.

(1) Coverage: for known distributions (exponential, lognormal, Pareto, a discrete law with ties, and
    the empirical OWD law of DET6G s1-UL used as a KNOWN distribution), draw n i.i.d. samples,
    compute T_hat (service_curve.certified_T) and the TRUE exceedance P(X > T_hat). The fraction of
    repetitions with true exceedance > eps must be <= delta (up to MC error). The naive empirical
    (1-eps)-quantile is reported for contrast.
(2) Loss budget: Bernoulli(p) losses, CP upper bound at level 1 - delta: fraction with bound < p <= delta.
(3) Union: P(delay > T_hat or lost) <= eps_lat + eps_loss jointly with prob >= 1 - 2 delta.
Output: results/verify_prop5.json
"""
from __future__ import annotations

import json

import numpy as np
from scipy import stats

from paths import DERIVED, RESULTS
from service_curve import certified_T, cp_upper

SEED = 42
DELTA = 0.025


def laws(rng):
    owd = np.load(DERIVED / "owd_s1-UL.npy").astype(float)
    emp = np.sort(owd[~np.isnan(owd)])
    return {
        "exponential": (lambda n: rng.exponential(1.0, n), lambda t: np.exp(-t)),
        "lognormal_s1": (lambda n: rng.lognormal(0, 1, n), lambda t: stats.lognorm.sf(t, 1)),
        "pareto_2.5": (lambda n: rng.pareto(2.5, n) + 1, lambda t: np.where(t < 1, 1.0, t ** -2.5)),
        "discrete_ties": (lambda n: rng.integers(0, 2000, n) // 7,
                          lambda t: 1 - np.minimum(2000, 7 * (np.floor(t) + 1)) / 2000),
        "det6g_s1UL_empirical": (lambda n: emp[rng.integers(0, len(emp), n)],
                                 lambda t: 1 - np.searchsorted(emp, t, side="right") / len(emp)),
    }


def coverage(rng) -> dict:
    out = {}
    for name, (draw, sf) in laws(rng).items():
        for n, eps, reps in ((20000, 1e-2, 1000), (20000, 1e-3, 1000), (200000, 1e-3, 300)):
            bad = bad_naive = 0
            for _ in range(reps):
                x = draw(n)
                c = certified_T(x, eps, DELTA)
                bad += int(sf(c["T_ms"]) > eps)
                bad_naive += int(sf(np.quantile(x, 1 - eps)) > eps)
            ci = stats.beta.ppf(0.975, bad + 1, reps - bad) if bad < reps else 1.0
            out[f"{name}|n={n}|eps={eps}"] = {"reps": reps, "fail_rate": bad / reps,
                                              "fail_rate_cp97.5": float(ci), "delta": DELTA,
                                              "naive_quantile_fail_rate": bad_naive / reps}
    return out


def loss_coverage(rng) -> dict:
    out = {}
    for p, n in ((1.2e-3, 10**6), (1e-3, 2 * 10**5), (5e-3, 10**5)):
        reps = 4000
        m = rng.binomial(n, p, reps)
        ub = np.array([cp_upper(int(k), n, DELTA) for k in m])
        out[f"p={p}|n={n}"] = {"reps": reps, "fail_rate": float(np.mean(ub < p)), "delta": DELTA}
    return out


def union_check(rng) -> dict:
    """Joint: OWD ~ empirical s1-UL law, loss ~ Bernoulli(1.2e-3); total violation probability of a
    packet = p_loss + (1 - p_loss) P(OWD > T_hat) must be <= eps_lat + eps_loss in >= 1 - 2 delta reps."""
    draw, sf = laws(rng)["det6g_s1UL_empirical"]
    n, p, eps_lat, reps, bad = 200000, 1.2e-3, 1e-3, 300, 0
    for _ in range(reps):
        lost = rng.random(n) < p
        x = draw(n)[~lost]
        T = certified_T(x, eps_lat, DELTA)["T_ms"]
        el = cp_upper(int(lost.sum()), n, DELTA)
        true_total = p + (1 - p) * float(sf(T))
        bad += int(true_total > eps_lat + el)
    return {"reps": reps, "fail_rate": bad / reps, "bound_2delta": 2 * DELTA}


def main() -> None:
    rng = np.random.default_rng(SEED)
    res = {"seed": SEED, "coverage": coverage(rng), "loss": loss_coverage(rng), "union": union_check(rng)}
    for k, v in res["coverage"].items():
        print(f"{k:45s} fail={v['fail_rate']:.4f} (CP97.5 {v['fail_rate_cp97.5']:.4f}) naive={v['naive_quantile_fail_rate']:.3f}")
    print(res["loss"]); print(res["union"])
    json.dump(res, open(RESULTS / "verify_prop5.json", "w"), indent=1)


if __name__ == "__main__":
    main()
