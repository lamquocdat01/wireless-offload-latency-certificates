"""P2 / Viec 1.2: FEC sensitivity under independence vs Gilbert-Elliott burst loss (Thm 6, Prop 7). seed 42.

Per DET6G session (18): grid OWD (p2_grid_owd.py), round-max certificate T_hat (Prop 5) at
eps_lat in {1e-3, 1e-4}, Clopper-Pearson loss bound, certified p = eps_lat + eps_loss.
  * loss runs vs geometric (independence) + inter-loss gap clustering;
  * GE MLE on the LOSS sequence and on the BAD sequence (lost or OWD > T_hat), LR test vs Bernoulli;
  * P_frame on N in {35, 100, 300} x r in 0..8 (+ r* search up to N) under:
      indep    : P(Bin(N + r, p) > r), p certified (Thm 6)
      ge_trace : GE fitted on the bad sequence, frame = N + r consecutive TRACE packets
      ge_time  : same GE, transitions rescaled to the frame's packet spacing
                 delta = c tau_f / N ms (c = 1.09, F = 30) -> whole frame ~ one GE state (pessimistic)
      empirical: non-overlapping blocks of N + r consecutive trace packets (model-free)
      union    : (N + r) p (no coding gain)
  * r* at eps_frame = 1e-3 for each model; flag r* <= 0.2 N.
Output: results/p2_fec_sensitivity.json
"""
from __future__ import annotations

import json
import math
from multiprocessing import Pool

import numpy as np

import fec
import service_curve as sc
from paths import DERIVED, RESULTS

N_GRID = (35, 100, 300)
R_GRID = tuple(range(0, 9))
EPS_LAT = (1e-3, 1e-4)
EPS_FRAME = 1e-3
C_TIME = 1.09
TAU_F = 1000.0 / 30.0


def session_job(s: str) -> dict:
    meta = json.load(open(RESULTS / "p2_grid_owd.json"))["sessions"][s]
    g = np.load(DERIVED / f"grid_{s}.npy").astype(float)
    iv = meta["rounds"][0]["interval_ms"]
    lost = np.isnan(g)
    # loss-run statistics vs geometric
    d = np.diff(np.r_[0, lost.astype(np.int8), 0])
    on, off = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    L = off - on
    p_loss = lost.mean()
    gaps = np.diff(on)
    out = {"session": s, "interval_ms": iv, "slots": int(len(g)), "lost": int(lost.sum()),
           "loss_runs": {"n_runs": int(len(L)), "max_len": int(L.max()), "n_len_ge2": int(np.sum(L >= 2)),
                         "expected_len_ge2_geometric": float(len(L) * p_loss),
                         "gap_lt10_obs": float(np.mean(gaps < 10)),
                         "gap_lt10_indep": float(1 - (1 - p_loss) ** 9),
                         "gap_lt100_obs": float(np.mean(gaps < 100)),
                         "gap_lt100_indep": float(1 - (1 - p_loss) ** 99)}}
    gl = fec.fit_ge(lost.astype(np.int8))
    out["ge_loss"] = {"p_gb": gl.p_gb, "p_bg": gl.p_bg, "h": gl.h, "mean_B_sojourn_pkts": 1 / gl.p_bg,
                      "loglik": gl.loglik, "loglik_bernoulli": fec.bernoulli_loglik(lost),
                      "LR": 2 * (gl.loglik - fec.bernoulli_loglik(lost))}
    rounds = sc.session_rounds(s)
    out["eps"] = {}
    for e in EPS_LAT:
        cert = sc.certificate_roundmax(rounds, e)
        T = cert["T_ms"]
        p = e + cert["eps_loss"]
        bad = (lost | (g > T)).astype(np.int8)
        gb = fec.fit_ge(bad)
        row = {"T_ms": T, "eps_loss": cert["eps_loss"], "p_certified": p, "bad_rate_empirical": float(bad.mean()),
               "ge_bad": {"p_gb": gb.p_gb, "p_bg": gb.p_bg, "h": gb.h, "mean_B_sojourn_pkts": 1 / gb.p_bg,
                          "pi_B": float(gb.pi[1]), "mean_bad": gb.mean_bad, "loglik": gb.loglik,
                          "LR_vs_bernoulli": 2 * (gb.loglik - fec.bernoulli_loglik(bad))},
               "grid": {}, "r_star": {}}
        for N in N_GRID:
            k_time = C_TIME * TAU_F / N / iv
            gt = gb.scaled(k_time)
            cells = {}
            for r in R_GRID:
                emp, nb = fec.empirical_block_rate(bad, N + r, r)
                cells[str(r)] = {"indep": fec.p_frame_indep(N, r, p), "ge_trace": fec.p_frame_ge(gb, N, r),
                                 "ge_time": fec.p_frame_ge(gt, N, r), "union": fec.union_bound(N, r, p),
                                 "empirical": emp, "empirical_blocks": nb}
            row["grid"][str(N)] = cells
            rmax = N
            rs = {"indep": fec.r_star_indep(N, p, EPS_FRAME, rmax), "ge_trace": fec.r_star_ge(gb, N, EPS_FRAME, rmax),
                  "ge_time": fec.r_star_ge(gt, N, EPS_FRAME, rmax)}
            rs["le_0.2N"] = {k: (v >= 0 and v <= 0.2 * N) for k, v in rs.items()}
            rs["c_eff_factor_indep"] = (N + rs["indep"]) / N if rs["indep"] >= 0 else None
            row["r_star"][str(N)] = rs
        out["eps"][str(e)] = row
    print(s, json.dumps({k: v["r_star"] for k, v in out["eps"].items()}), flush=True)
    return out


def main() -> None:
    sessions = list(json.load(open(RESULTS / "p2_grid_owd.json"))["sessions"])
    with Pool(3) as pool:
        res = pool.map(session_job, sessions)
    by = {r["session"]: r for r in res}
    ul = [s for s in sessions if s.endswith("UL")]
    summ = {}
    for e in map(str, EPS_LAT):
        for N in map(str, N_GRID):
            for m in ("indep", "ge_trace", "ge_time"):
                ok = sum(by[s]["eps"][e]["r_star"][N]["le_0.2N"][m] for s in ul)
                rs = [by[s]["eps"][e]["r_star"][N][m] for s in ul]
                summ[f"eps_lat={e}|N={N}|{m}"] = {"UL_sessions_r*_le_0.2N": ok, "of": len(ul),
                                                   "r*_UL": rs}
    # early-stop rule of the P2 prompt: under GE, r <= 0.2N must reach 1e-3 in >= 7/13 UL sessions
    stop = {f"N={N}|eps_lat={e}": summ[f"eps_lat={e}|N={N}|ge_trace"]["UL_sessions_r*_le_0.2N"] >= 7
            for e in map(str, EPS_LAT) for N in map(str, N_GRID)}
    out = {"seed": 42, "eps_frame": EPS_FRAME, "N_grid": N_GRID, "r_grid": R_GRID, "c_time_scaling": C_TIME,
           "sessions": by, "summary": summ, "headline_holds_under_ge_trace": stop}
    json.dump(out, open(RESULTS / "p2_fec_sensitivity.json", "w"), indent=1, default=float)
    print(json.dumps(summ, indent=0)); print(stop)


if __name__ == "__main__":
    main()
