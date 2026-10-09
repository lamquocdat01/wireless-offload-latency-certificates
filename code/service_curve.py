"""Statistical service-curve certificate from per-packet one-way delay (topic #19, Prop 5).

From owd_<session>.npy (ms, NaN = sent but not received; parse_det6g.py) and the send-grid loss
accounting of P0 (results/p0_owd_stats.json; the trace's `lost` column is constant and unused):

* T_hat_eps   = order statistic X_(n - m*) of the n received OWDs, m* = max{m : BinCDF(m; n, eps_lat)
                <= delta_lat}.  Then P( P(OWD > T_hat | T_hat) <= eps_lat ) >= 1 - delta_lat for i.i.d.
                OWD with ANY distribution (one-sided Clopper-Pearson / Wilks order statistic).
* eps_loss    = one-sided Clopper-Pearson upper bound (level 1 - delta_loss) of the packet loss
                probability from m_lost of n_sent packets (n_sent = send grid, P0 rule).
* eps         = eps_lat + eps_loss  (union bound; joint confidence >= 1 - delta_lat - delta_loss).
* hold-out    = first 60 % of the 30-min rounds calibrate T_hat, the remaining rounds test it:
                exceedance rate among received, loss rate, total violation rate (+ binomial p-values).

Output (main): results/p1_service_curve.json
"""
from __future__ import annotations

import json
import math

import numpy as np
from scipy import stats

from paths import DERIVED, RESULTS

SEED = 42
DELTA_LAT = 0.025
DELTA_LOSS = 0.025
EPS_LAT_GRID = (1e-2, 1e-3, 1e-4)
HOLDOUT_FRAC = 0.6


def cp_upper(m: int, n: int, delta: float) -> float:
    """One-sided Clopper-Pearson upper bound for a binomial proportion (level 1 - delta)."""
    if n <= 0:
        return 1.0
    if m >= n:
        return 1.0
    return float(stats.beta.ppf(1.0 - delta, m + 1, n - m))


def cert_exceedances(n: int, eps: float, delta: float) -> int:
    """m* = max{m >= 0 : BinCDF(m; n, eps) <= delta}; -1 if even m = 0 fails (n too small)."""
    if stats.binom.cdf(0, n, eps) > delta:
        return -1
    # BinCDF(m; n, eps) is increasing in m: bisection on [0, n]
    lo, hi = 0, max(1, int(n * eps + 10 * math.sqrt(n * eps + 1) + 10))
    while stats.binom.cdf(hi, n, eps) <= delta:
        hi *= 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if stats.binom.cdf(mid, n, eps) <= delta:
            lo = mid
        else:
            hi = mid
    return lo


def certified_T(owd_rx: np.ndarray, eps: float, delta: float = DELTA_LAT) -> dict:
    """T_hat with P(exceedance <= eps) >= 1 - delta (Prop 5). owd_rx: received OWDs only."""
    x = np.sort(np.asarray(owd_rx, float))
    n = len(x)
    m = cert_exceedances(n, eps, delta)
    if m < 0:
        return {"T_ms": math.inf, "m_star": m, "n": n, "eps": eps, "delta": delta}
    T = float(x[n - m - 1])                         # X_(n - m), 1-indexed
    return {"T_ms": T, "m_star": m, "n": n, "eps": eps, "delta": delta,
            "T_empirical_quantile_ms": float(np.quantile(x, 1 - eps)),
            "cp_upper_of_exceedance": cp_upper(m, n, delta)}


def session_rounds(session: str) -> list[dict]:
    """Per 30-min round: received OWDs, n_sent (grid), n_lost (NaN rows + grid-missing)."""
    owd = np.load(DERIVED / f"owd_{session}.npy").astype(float)
    meta = json.load(open(RESULTS / "p0_owd_stats.json"))["sessions"][session]
    out, i = [], 0
    for r in meta["rounds"]:
        seg = owd[i:i + r["n"]]
        i += r["n"]
        nan = int(np.isnan(seg).sum())
        n_sent = max(r["expected"], r["n"])
        out.append({"rx": seg[~np.isnan(seg)], "n_sent": n_sent,
                    "n_lost": nan + (n_sent - r["n"])})
    assert i == len(owd), session
    return out


def certificate(rounds: list[dict], eps_lat: float, delta_lat: float = DELTA_LAT,
                delta_loss: float = DELTA_LOSS) -> dict:
    rx = np.concatenate([r["rx"] for r in rounds])
    n_sent = sum(r["n_sent"] for r in rounds)
    n_lost = sum(r["n_lost"] for r in rounds)
    lat = certified_T(rx, eps_lat, delta_lat)
    eps_loss = cp_upper(n_lost, n_sent, delta_loss)
    return {**lat, "n_sent": int(n_sent), "n_lost": int(n_lost), "loss_hat": n_lost / n_sent,
            "eps_loss": eps_loss, "eps_total": eps_lat + eps_loss,
            "confidence": 1 - delta_lat - delta_loss}


def certificate_roundmax(rounds: list[dict], eps_lat: float, delta_lat: float = DELTA_LAT,
                         delta_loss: float = DELTA_LOSS) -> dict:
    """Robust variant (Remark 5.2): T_hat = max over rounds of the per-round certified T_hat
    (each round certified at level 1 - delta_lat / n_rounds, Bonferroni). Guards against a
    calibration period that is easier than the future, at the price of a larger T."""
    k = len(rounds)
    Ts = [certified_T(r["rx"], eps_lat, delta_lat / k)["T_ms"] for r in rounds]
    base = certificate(rounds, eps_lat, delta_lat, delta_loss)
    return {**base, "T_ms": float(max(Ts)), "T_per_round_ms": Ts, "T_pooled_ms": base["T_ms"]}


def holdout(rounds: list[dict], eps_lat: float, robust: bool = False) -> dict:
    n_cal = max(1, int(round(HOLDOUT_FRAC * len(rounds))))
    if n_cal >= len(rounds):
        n_cal = len(rounds) - 1
    cal, test = rounds[:n_cal], rounds[n_cal:]
    cert = (certificate_roundmax if robust else certificate)(cal, eps_lat)
    rx = np.concatenate([r["rx"] for r in test])
    n_sent = sum(r["n_sent"] for r in test)
    n_lost = sum(r["n_lost"] for r in test)
    exc = int(np.sum(rx > cert["T_ms"]))
    viol = exc + n_lost
    pval = lambda k, n, p: float(stats.binom.sf(k - 1, n, p)) if k > 0 else 1.0   # P(X >= k | p)
    return {"rounds_cal": n_cal, "rounds_test": len(test), "T_cal_ms": cert["T_ms"],
            "eps_loss_cal": cert["eps_loss"], "eps_total_cal": cert["eps_total"],
            "test_n_rx": int(len(rx)), "test_n_sent": int(n_sent),
            "test_exceed": exc, "test_exceed_rate": exc / max(len(rx), 1),
            "test_loss_rate": n_lost / n_sent, "test_viol_rate": viol / n_sent,
            "lat_ok": exc / max(len(rx), 1) <= eps_lat,
            "total_ok": viol / n_sent <= cert["eps_total"],
            "p_lat_H0_rate_le_eps": pval(exc, len(rx), eps_lat),
            "p_total_H0_rate_le_epstotal": pval(viol, n_sent, min(1.0, cert["eps_total"]))}


def main() -> None:
    sessions = json.load(open(RESULTS / "p0_owd_stats.json"))["sessions"]
    out = {"seed": SEED, "delta_lat": DELTA_LAT, "delta_loss": DELTA_LOSS,
           "holdout": f"first {HOLDOUT_FRAC:.0%} of 30-min rounds (rounded) calibrate, rest test",
           "loss_rule": "n_sent = send-time grid (P0); lost = NaN delay rows + grid-missing packets",
           "sessions": {}}
    for s, meta in sessions.items():
        rounds = session_rounds(s)
        row = {"direction": meta["direction"], "rounds": len(rounds),
               "packet_interval_ms": meta["packet_interval_ms"], "eps": {}}
        for e in EPS_LAT_GRID:
            row["eps"][str(e)] = {"full": certificate(rounds, e), "holdout": holdout(rounds, e),
                                  "full_roundmax": certificate_roundmax(rounds, e),
                                  "holdout_roundmax": holdout(rounds, e, robust=True)}
        # smallest total eps certifiable: eps_loss + the smallest eps_lat for which m* >= 0
        c = row["eps"]["0.001"]["full"]
        row["eps_loss_full"] = c["eps_loss"]
        row["eps1e-3_total_survives"] = bool(c["eps_loss"] < 1e-3)
        out["sessions"][s] = row
        h = row["eps"]["0.001"]["holdout"]
        print(f"{s:7s} T(1e-3)={c['T_ms']:.3f} ms m*={c['m_star']} n={c['n']} loss={c['loss_hat']:.2e} "
              f"eps_loss={c['eps_loss']:.2e} | holdout exc={h['test_exceed_rate']:.2e} "
              f"viol={h['test_viol_rate']:.2e} vs {h['eps_total_cal']:.2e} p={h['p_total_H0_rate_le_epstotal']:.3f}",
              flush=True)
    ul = [v for v in out["sessions"].values() if v["direction"] == "uplink"]
    out["summary"] = {}
    for e in EPS_LAT_GRID:
        k = str(e)
        hs = [v["eps"][k]["holdout"] for v in out["sessions"].values()]
        hr = [v["eps"][k]["holdout_roundmax"] for v in out["sessions"].values()]
        out["summary"][k] = {
            "sessions": len(hs),
            "roundmax_holdout_lat_ok": int(sum(h["lat_ok"] for h in hr)),
            "roundmax_holdout_lat_reject_5pct": int(sum(h["p_lat_H0_rate_le_eps"] < 0.05 for h in hr)),
            "roundmax_T_over_pooled_T_median": float(np.median(
                [v["eps"][k]["full_roundmax"]["T_ms"] / v["eps"][k]["full"]["T_ms"] for v in out["sessions"].values()])),
            "holdout_lat_ok": int(sum(h["lat_ok"] for h in hs)),
            "holdout_total_ok": int(sum(h["total_ok"] for h in hs)),
            "holdout_lat_reject_5pct": int(sum(h["p_lat_H0_rate_le_eps"] < 0.05 for h in hs)),
            "eps_loss_range": [min(v["eps"][k]["full"]["eps_loss"] for v in out["sessions"].values()),
                               max(v["eps"][k]["full"]["eps_loss"] for v in out["sessions"].values())]}
    out["summary"]["eps_total_1e-3_survives_UL_sessions"] = int(sum(v["eps1e-3_total_survives"] for v in ul))
    out["summary"]["UL_sessions"] = len(ul)
    json.dump(out, open(RESULTS / "p1_service_curve.json", "w"), indent=1)
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
