"""P2 / Viec 2: full trace-driven evaluation, 18 DET6G sessions, FRAME-level SLA (topic #19). seed 42.

Per session (13 UL main, 5 DL secondary), temporal hold-out: the first 60 % of the 30-min rounds
calibrate, the remaining rounds are the test trace.
  calibration: T_hat_rmax(eps_lat) (Prop 5, round-max), eps_loss (Clopper-Pearson), p = eps_lat + eps_loss,
               r*(N) = min r with P(Bin(N+r, p) > r) <= eps_frame (Thm 6), c_eff = c (N + r*)/N
  Alg 1 (4 variables, Thm 4'): (K*, a_q, b*, r*) from (Delta, T_hat, L_srv, c_eff, a_q) or reject.
  policies   : token_bucket (K*, a_q, b* = b_max rounded down to 0.25, <= 64)
               gate_refresh (K*, a_q, b = inf)       -- certified only through the empirical envelope
               periodic     (same realised rate as token_bucket, random phase per video)
  replay     : corpus 3 170 events; FIFO transmitter with service c_eff per frame; the frame's N + r
               packets take N + r CONSECUTIVE test-grid OWD samples starting at the packet sent at the
               frame's transmission finish (P0 convention); the frame is decoded when its N-th packet
               arrives (all packets conservatively taken as sent at the finish time); fewer than N
               packets received -> undecodable.
               frame-level violation = undecodable or e2e > Delta.
Grid: Delta {100,200,500,1000} ms x c {0.6,1.09,1.5,2.5} x a_q {0.05,0.1,0.2,0.3} x eps_lat {1e-3,1e-4}
      x N {35,100,300}; L_srv = 20 ms; eps_frame = 1e-3.
r-model (P3, CP2 decision #2; env P3_RMODEL or --rmodel):
  indep    : r* = min r with P(Bin(N+r, p) > r) <= eps_frame (Thm 6)                    [P2 run, p2_main_*]
  ge_time  : HEADLINE. GE fitted (MLE, Prop 7) on the bad sequence of the CALIBRATION rounds only;
             transitions rescaled to the frame's packet spacing delta = c tau_f / N (frame-time scale,
             pessimistic); r* = max(r*_indep, r*_GE-time(c)), so it depends on c   [p3_main_getime_*]
  ge_trace : sensitivity column "if independent within the frame at trace spacing": r* = r*_GE-trace
                                                                                       [p3_main_getrace_*]
Outputs: results/p2_main_cells.csv (one row per session x cell x policy),
         THS_DATASETS/_derived/19/p2_main_pervideo.npz (per-video eval / violation counts, bootstrap),
         results/p2_main_crosssession.json (radio-level cross-session hold-out).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import fec
import nc_bounds as nb
import service_curve as sc
from paths import DERIVED, RESULTS
from policy import calibrate_tau, gate_refresh, periodic_rate, token_bucket

TAU_F = 1000.0 / 30.0
L_SRV = 20.0
EPS_FRAME = 1e-3
HOLDOUT = 0.6
DELTAS = (100.0, 200.0, 500.0, 1000.0)
C_GRID = (0.6, 1.09, 1.5, 2.5)
A_Q = (0.05, 0.1, 0.2, 0.3)
EPS_LAT = (1e-3, 1e-4)
N_GRID = (35, 100, 300)
B_CAP = 64.0
C_CROSS = 1.09            # frame-time scaling for the radio-level cross-session hold-out (no c there)
R_MODELS = ("indep", "ge_time", "ge_trace")
TAG = {"indep": "p2_main", "ge_time": "p3_main_getime", "ge_trace": "p3_main_getrace"}


def r_model() -> str:
    m = os.environ.get("P3_RMODEL", "indep")
    assert m in R_MODELS, m
    return m


def r_for(model: str, N: int, p: float, c: float, ge, iv_ms: float) -> tuple[int, int, int]:
    """(r used, r*_indep, r*_GE of the model or -1). -1 for 'r used' = no r <= N reaches eps_frame."""
    ri = fec.r_star_indep(N, p, EPS_FRAME)
    if model == "indep":
        return ri, ri, -1
    if model == "ge_trace":
        rg = fec.r_star_ge(ge, N, EPS_FRAME)
        return rg, ri, rg
    rg = fec.r_star_ge(ge.scaled(c * TAU_F / N / iv_ms), N, EPS_FRAME)
    return (max(ri, rg) if rg >= 0 else -1), ri, rg

_CORPUS = None


def corpus():
    global _CORPUS
    if _CORPUS is None:
        act = pd.read_parquet(DERIVED / "activations_K2_a0.05.parquet",
                              columns=["dataset", "video", "frame", "g_t", "score"])
        vids = [g.reset_index(drop=True) for _, g in act.groupby(["dataset", "video"], sort=False)]
        ev = []
        for v in vids:
            d = np.diff(np.r_[0, v.g_t.to_numpy(), 0])
            ev.append((np.flatnonzero(d == 1), np.flatnonzero(d == -1)))
        _CORPUS = {"act": act, "vids": vids, "ev": ev, "tau": {}, "opens": {}}
    return _CORPUS


def tau_for(K: int, a: float) -> float:
    C = corpus()
    if (K, a) not in C["tau"]:          # K = 1: every frame is a refresh, the gate is idle
        C["tau"][(K, a)] = math.inf if K == 1 else calibrate_tau(C["act"], K, a)
    return C["tau"][(K, a)]


def opens_for(policy: str, K: int, a: float, b: float, rate: float | None = None) -> list[np.ndarray]:
    C = corpus()
    key = (policy, K, a, b, None if rate is None else round(rate, 9))
    if key not in C["opens"]:
        sc_ = [v.score.to_numpy() for v in C["vids"]]
        if policy == "token_bucket":
            t = tau_for(K, a)
            o = [token_bucket(s, K, t, a, b) for s in sc_]
        elif policy == "gate_refresh":
            t = tau_for(K, a)
            o = [gate_refresh(s, K, t) for s in sc_]
        else:   # periodic at the given pooled rate, deterministic phase per video
            rng = np.random.default_rng([42, K, int(round(a * 1000)), int(round(rate * 1e6))])
            o = [periodic_rate(len(s), rate, float(rng.random())) for s in sc_]
        C["opens"][key] = o
        if len(C["opens"]) > 400:                       # bounded cache
            C["opens"].pop(next(iter(C["opens"])))
    return C["opens"][key]


def replay_frames(opens, c_eff, test_grid, M, N, iv_ms, rng):
    """Per video: (dev slots, e2e ms or inf if undecodable) of caught events."""
    C = corpus()
    L = len(test_grid) - M - 1
    out = []
    for o, (on, end) in zip(opens, C["ev"]):
        idx = np.flatnonzero(o)
        if len(idx) == 0 or len(on) == 0:
            out.append((np.array([]), np.array([])))
            continue
        k = np.arange(len(idx))
        finish = (k + 1) * c_eff + np.maximum.accumulate(idx - k * c_eff)
        pos = np.searchsorted(idx, on)
        ok = pos < len(idx)
        ok[ok] = idx[pos[ok]] < end[ok]
        j = pos[ok]
        if len(j) == 0:
            out.append((np.array([]), np.array([])))
            continue
        off = int(rng.integers(L))
        start = (off + np.floor(finish[j] * TAU_F / iv_ms).astype(np.int64)) % L
        blk = test_grid[start[:, None] + np.arange(M)[None, :]]
        blk = np.where(np.isnan(blk), np.inf, blk)
        radio = np.partition(blk, N - 1, axis=1)[:, N - 1]          # N-th arrival (inf = undecodable)
        dev = idx[j] - on[ok]
        e2e = dev * TAU_F + (finish[j] - idx[j]) * TAU_F + radio + L_SRV
        out.append((dev, e2e))
    return out


MIN_FREE_GB = 6.0


def _ram_guard(tag: str) -> None:
    import psutil
    free = psutil.virtual_memory().available / 1e9
    if free < MIN_FREE_GB:
        raise MemoryError(f"{tag}: free RAM {free:.1f} GB < {MIN_FREE_GB} GB, stopping (machine rule)")


def session_job(args):
    s, s_idx = args
    _ram_guard(s)
    meta = json.load(open(RESULTS / "p2_grid_owd.json"))["sessions"][s]
    g = np.load(DERIVED / f"grid_{s}.npy").astype(float)
    iv = meta["rounds"][0]["interval_ms"]
    n_r = len(meta["rounds"])
    n_cal = min(max(1, int(round(HOLDOUT * n_r))), n_r - 1)
    cal_slots = sum(r["slots"] for r in meta["rounds"][:n_cal])
    test = g[cal_slots:]
    rounds = sc.session_rounds(s)[:n_cal]
    model = r_model()
    rows, pv_eval, pv_viol, ge_info = [], [], [], {}
    t0 = time.time()
    for e in EPS_LAT:
        cert = sc.certificate_roundmax(rounds, e)
        T, p = cert["T_ms"], e + cert["eps_loss"]
        ge = None
        if model != "indep":                      # GE on the calibration rounds only (no test leakage)
            cal = g[:cal_slots]
            ge = fec.fit_ge((np.isnan(cal) | (cal > T)).astype(np.int8))
            ge_info[str(e)] = {"p_gb": ge.p_gb, "p_bg": ge.p_bg, "h": ge.h, "mean_bad": ge.mean_bad,
                               "cal_slots": int(cal_slots)}
        for N in N_GRID:
            rc = {c: r_for(model, N, p, c, ge, iv) for c in C_GRID}
            if ge is not None:
                ge_info[str(e)][f"r|N={N}"] = {str(c): list(v) for c, v in rc.items()}
            for D in DELTAS:
                for c in C_GRID:
                    r, r_ind, r_ge = rc[c]
                    if r < 0:                     # no r <= N reaches eps_frame: reject every policy
                        for a_q in A_Q:
                            for pol in ("token_bucket", "gate_refresh", "periodic"):
                                rows.append({"session": s, "dir": s[-2:], "eps_lat": e, "N": N, "r": r,
                                             "r_indep": r_ind, "r_model": r_ge, "rmodel": model, "p": p,
                                             "T_ms": T, "Delta": D, "c": c, "a_q": a_q, "policy": pol,
                                             "alg1_admitted": False, "alg1_reason": "r_infeasible",
                                             "certified": False, "n_events": 3170})
                                pv_eval.append(np.zeros(121, np.int32)); pv_viol.append(np.zeros(121, np.int32))
                        continue
                    M = N + r
                    ce = fec.c_eff(c, N, r)
                    for a_q in A_Q:
                        plan = nb.alg1(D, T, L_SRV, ce, TAU_F, a_q, b_rule="max")
                        K = plan.K if plan.K >= 1 else None
                        base = {"session": s, "dir": s[-2:], "eps_lat": e, "N": N, "r": r, "r_indep": r_ind,
                                "r_model": r_ge, "rmodel": model, "p": p, "T_ms": T,
                                "Delta": D, "c": c, "c_eff": ce, "a_q": a_q, "K": plan.K,
                                "alg1_admitted": plan.admitted, "alg1_reason": plan.reason,
                                "b_max": plan.b_max}
                        if K is None:
                            for pol in ("token_bucket", "gate_refresh", "periodic"):
                                rows.append({**base, "policy": pol, "certified": False, "n_events": 3170})
                                pv_eval.append(np.zeros(121, np.int32)); pv_viol.append(np.zeros(121, np.int32))
                            continue
                        bm = plan.b_max if np.isfinite(plan.b_max) else B_CAP
                        b_tb = min(B_CAP, math.floor(max(bm, 0) * 4) / 4)
                        b_tb = max(b_tb, nb.b_min_tight(a_q))   # rejected cells still replayed at b_min
                        o_tb = opens_for("token_bucket", K, a_q, b_tb)
                        rate_tb = sum(o.sum() for o in o_tb) / sum(len(o) for o in o_tb)
                        pols = {"token_bucket": (o_tb, b_tb), "gate_refresh": (opens_for("gate_refresh", K, a_q, math.inf), math.inf),
                                "periodic": (opens_for("periodic", K, a_q, 0.0, rate_tb), 0.0)}
                        for pol, (op, b) in pols.items():
                            rate = sum(o.sum() for o in op) / sum(len(o) for o in op)
                            if pol == "periodic":
                                Kd = int(max(np.max(np.diff(np.flatnonzero(o))) for o in op if o.sum() > 1))
                            else:
                                Kd = K
                            if pol == "token_bucket" and plan.admitted:
                                Q = nb.q_tb(K, a_q, b, ce) if ce > 1 else ce
                                qsrc = "thm1"
                            else:
                                Q = nb.q_from_sequences(op, ce) if rate * ce < 1 else math.inf
                                qsrc = "empirical"
                            bound = nb.e2e_bound_ms(Kd, Q, ce, T, L_SRV, TAU_F) if math.isfinite(Q) else math.inf
                            rng = np.random.default_rng([42, s_idx, len(rows)])
                            rep = replay_frames(op, ce, test, M, N, iv, rng)
                            ev_v = np.array([len(x[1]) for x in rep], np.int32)
                            vi_v = np.array([int(np.sum(x[1] > D)) for x in rep], np.int32)
                            e2e = np.concatenate([x[1] for x in rep]) if ev_v.sum() else np.array([])
                            dev = np.concatenate([x[0] for x in rep]) if ev_v.sum() else np.array([])
                            fin = e2e[np.isfinite(e2e)]
                            rows.append({**base, "policy": pol, "b": b, "rate": rate, "K_dev": Kd, "Q": Q,
                                         "Q_source": qsrc, "bound_ms": bound,
                                         "certified": bool(bound <= D) and (pol != "token_bucket" or plan.admitted),
                                         "n_events": 3170, "n_caught": int(ev_v.sum()),
                                         "n_undecodable": int(np.sum(~np.isfinite(e2e))),
                                         "n_viol": int(vi_v.sum()),
                                         "viol_rate": float(vi_v.sum() / max(ev_v.sum(), 1)),
                                         "max_e2e_decoded": float(fin.max()) if len(fin) else None,
                                         "tightness": float(fin.max() / bound) if len(fin) and math.isfinite(bound) else None,
                                         "n_over_bound_decoded": int(np.sum(fin > bound)) if math.isfinite(bound) else None,
                                         "mean_e2e": float(fin.mean()) if len(fin) else None,
                                         "mean_device_ms": float(dev.mean() * TAU_F) if len(dev) else None,
                                         "miss_rate": 1 - ev_v.sum() / 3170})
                            pv_eval.append(ev_v); pv_viol.append(vi_v)
        print(f"{s} eps_lat={e} done {time.time() - t0:.0f}s", flush=True)
        _ram_guard(s)
    return s, rows, np.array(pv_eval), np.array(pv_viol), ge_info


def cross_session() -> dict:
    """Radio-level cross-session hold-out: calibrate (T_hat_rmax, eps_loss, r*) on ALL rounds of session
    i, test on session j (same direction): fraction of non-overlapping blocks of N + r* consecutive
    packets whose N-th smallest OWD exceeds T_hat_i or that have < N received packets."""
    meta = json.load(open(RESULTS / "p2_grid_owd.json"))["sessions"]
    model = r_model()
    out = {}
    for e in EPS_LAT:
        cal = {s: sc.certificate_roundmax(sc.session_rounds(s), e) for s in meta}
        ges = {}
        if model != "indep":
            for s in meta:
                gs = np.load(DERIVED / f"grid_{s}.npy").astype(float)
                ges[s] = fec.fit_ge((np.isnan(gs) | (gs > cal[s]["T_ms"])).astype(np.int8))
        for N in N_GRID:
            tab = {}
            for d in ("UL", "DL"):
                ss = [s for s in meta if s.endswith(d)]
                grids = {s: np.load(DERIVED / f"grid_{s}.npy").astype(float) for s in ss}
                for i in ss:
                    p = e + cal[i]["eps_loss"]
                    r = r_for(model, N, p, C_CROSS, ges.get(i), meta[i]["rounds"][0]["interval_ms"])[0]
                    if r < 0:
                        for j in ss:
                            tab[f"{i}->{j}"] = {"r": r, "blocks": 0, "bad_blocks": 0, "rate": 1.0, "cp_upper_95": 1.0}
                        continue
                    M = N + r
                    for j in ss:
                        gj = grids[j]
                        nb_ = len(gj) // M
                        blk = gj[:nb_ * M].reshape(nb_, M)
                        blk = np.where(np.isnan(blk), np.inf, blk)
                        nth = np.partition(blk, N - 1, axis=1)[:, N - 1]
                        k = int(np.sum(nth > cal[i]["T_ms"]))
                        tab[f"{i}->{j}"] = {"r": r, "blocks": nb_, "bad_blocks": k, "rate": k / nb_,
                                            "cp_upper_95": sc.cp_upper(k, nb_, 0.05)}
            out[f"eps_lat={e}|N={N}"] = tab
            print("cross", e, N, flush=True)
    return out


def main() -> None:
    sessions = list(json.load(open(RESULTS / "p2_grid_owd.json"))["sessions"])
    args = sys.argv[1:]
    if "--rmodel" in args:                   # set before the Pool is created: spawned workers inherit it
        i = args.index("--rmodel")
        os.environ["P3_RMODEL"] = args[i + 1]
        del args[i:i + 2]
    model = r_model()
    only = args                              # optional subset for pilots
    jobs = [(s, i) for i, s in enumerate(sessions) if not only or s in only]
    t0 = time.time()
    with Pool(2) as pool:
        res = pool.map(session_job, jobs)
    rows = [r for _, rr, _, _, _ in res for r in rr]
    df = pd.DataFrame(rows)
    tag = "" if not only else "_pilot"
    base = TAG[model]
    df.to_csv(RESULTS / f"{base}_cells{tag}.csv", index=False)
    np.savez_compressed(DERIVED / f"{base}_pervideo{tag}.npz",
                        eval=np.concatenate([e for _, _, e, _, _ in res]),
                        viol=np.concatenate([v for _, _, _, v, _ in res]))
    if model != "indep":
        json.dump({s_: gi for s_, _, _, _, gi in res}, open(RESULTS / f"{base}_ge_fit{tag}.json", "w"),
                  indent=1, default=float)
    print(f"[{model}] cells {len(df)} in {time.time() - t0:.0f}s", flush=True)
    if not only:
        json.dump(cross_session(), open(RESULTS / f"{base}_crosssession.json", "w"), indent=1)


if __name__ == "__main__":
    main()
