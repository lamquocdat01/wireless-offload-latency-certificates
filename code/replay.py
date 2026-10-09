"""Trace replay of the gated-camera class over a measured 5G link (topic #19, P1 harness).

Module version of the P0 sanity (b) logic (p0_sanity.sanity_b), split into reusable parts:

    replay(session, c, policy, K, a, b, Delta, L_srv, ...) -> per-event e2e latencies [ms],
        deadline violations, the Thm-2 bound and the bound check.

Link model (A4): one FIFO constant-rate transmitter (c slots per frame, c = s F / R) followed by a
per-frame radio latency drawn from the session's measured one-way-delay (OWD) sequence (NaN = lost
packet, re-inserted at random positions for the packets missing from the send grid, as in P0).
The OWD of a frame is the OWD of the packet sent at the frame's transmission-finish time
(index = offset + floor(finish * tau_f / packet_interval)), offset uniform per video (seed).

Events: maximal runs of g_t = 1 (detector reports >= 1 box). Caught at the first open frame
inside the run. Latency = device wait + queue + transmission + radio (+ L_srv).

p1_reproduce_p0.py re-runs the 147 stable P0 cells with this module and checks equality with
results/p0_sanity_b_cells.csv (tolerance 0).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

import nc_bounds as nb
from paths import DERIVED, RESULTS

FPS = 30.0
TAU_F = 1000.0 / FPS
P0_POLICIES = ("gate_refresh", "token_bucket_b1", "token_bucket_b4", "periodic_same_rate")


# ----------------------------------------------------------------------------- inputs

def session_meta(session: str) -> dict:
    return json.load(open(RESULTS / "p0_owd_stats.json"))["sessions"][session]


def load_owd(session: str, rng: np.random.Generator | None) -> tuple[np.ndarray, np.ndarray, dict]:
    """(owd_with_losses, owd_received, meta). Grid-missing packets are inserted as NaN at
    rng-drawn positions (P0 convention); rng=None appends nothing (losses = NaN rows only)."""
    owd_all = np.load(DERIVED / f"owd_{session}.npy").astype(float)
    meta = session_meta(session)
    rx = owd_all[~np.isnan(owd_all)]
    owd = owd_all.copy()
    n_miss = meta["n_missing_from_grid"]
    if n_miss and rng is not None:
        owd = np.insert(owd, rng.integers(0, len(owd), n_miss), np.nan)
    return owd, rx, meta


def load_activations(K: int, a: float) -> list[pd.DataFrame]:
    """Per-video frames of the activation corpus. Uses the stored P0 file for (K, a) when it exists;
    otherwise the corpus frames (score, g_t) of any stored file, without P0 open_* columns."""
    f = DERIVED / f"activations_K{K}_a{a}.parquet"
    if not f.exists():
        f = DERIVED / "activations_K2_a0.05.parquet"
        act = pd.read_parquet(f, columns=["dataset", "video", "frame", "g_t", "score"])
    else:
        act = pd.read_parquet(f)
    return [g for _, g in act.groupby(["dataset", "video"], sort=False)]


def policy_opens(vids: list[pd.DataFrame], policy: str, K: int, a: float, b: float | None
                 ) -> list[np.ndarray]:
    """Open sequences: stored P0 columns, or token_bucket with an arbitrary depth b (P0 semantics,
    policy.token_bucket, global tau from p0_burstiness.json)."""
    col = f"open_{policy}"
    if col in vids[0].columns:
        return [v[col].to_numpy() for v in vids]
    if policy != "token_bucket":
        raise ValueError(policy)
    from policy import calibrate_tau, token_bucket      # P0 policy (raw float semantics)
    tau = {(c["K"], c["a"]): c["tau"] for c in json.load(open(RESULTS / "p0_burstiness.json"))["cells"]}
    t = tau.get((K, a))
    if t is None:                                          # same global calibration rule as P0
        t = calibrate_tau(pd.concat(vids, ignore_index=True), K, a)
    return [token_bucket(v.score.to_numpy(), K, t, a, float(b)) for v in vids]


# ----------------------------------------------------------------------------- core

def replay_video(opens: np.ndarray, g: np.ndarray, owd: np.ndarray, c: float, off: int,
                 iv_ms: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Per caught event: device latency [slots] and e2e latency [ms] (NaN if the frame is lost).
    Identical arithmetic to p0_sanity.replay_video."""
    idx = np.flatnonzero(opens)
    d = np.diff(np.r_[0, g, 0])
    on, end = np.flatnonzero(d == 1), np.flatnonzero(d == -1)           # event = [on, end)
    if len(idx) == 0:
        return np.array([]), np.array([]), len(on)
    k = np.arange(len(idx))
    finish = (k + 1) * c + np.maximum.accumulate(idx - k * c)          # FIFO, constant service c
    radio = owd[(off + np.floor(finish * TAU_F / iv_ms).astype(np.int64)) % len(owd)]
    delay_ms = (finish - idx) * TAU_F + radio                           # queue + tx + radio
    pos = np.searchsorted(idx, on)                                       # first open >= onset
    ok = pos < len(idx)
    ok[ok] = idx[pos[ok]] < end[ok]
    j = pos[ok]
    dev = idx[j] - on[ok]
    return dev, dev * TAU_F + delay_ms[j], len(on)


def device_k(policy: str, K: int, opens: list[np.ndarray]) -> int:
    """Device worst case (refresh period): K, or the largest gap for the periodic baseline."""
    if policy == "periodic_same_rate":
        return int(max(np.max(np.diff(np.flatnonzero(o))) for o in opens if o.sum() > 1))
    return K


@dataclass
class CellResult:
    dev: np.ndarray          # device latency of caught events [slots]
    e2e: np.ndarray          # e2e latency with real OWD [ms] (NaN = lost frame)
    e2e_clip: np.ndarray     # e2e with OWD clipped at T (A4 holds) [ms]
    n_events: int
    rate: float
    Q: float
    Q_grid: float
    K_dev: int


def replay_cell(vids, opens, owd, T_ms, c, iv_ms, rng, policy, K, alpha_grid_fn=None) -> CellResult:
    owd_clip = np.minimum(owd, T_ms)
    rate = float(sum(o.sum() for o in opens) / sum(len(o) for o in opens))
    Kd = device_k(policy, K, opens)
    Q = nb.q_from_sequences(opens, c)
    Q_grid = alpha_grid_fn(opens, c) if alpha_grid_fn else float("nan")
    devs, e2es, dets, n_ev = [], [], [], 0
    for v, o in zip(vids, opens):
        off = int(rng.integers(len(owd)))
        dv, e2, ne = replay_video(o, v.g_t.to_numpy(), owd, c, off, iv_ms)
        _, e2d, _ = replay_video(o, v.g_t.to_numpy(), owd_clip, c, off, iv_ms)
        devs.append(dv); e2es.append(e2); dets.append(e2d); n_ev += ne
    return CellResult(np.concatenate(devs), np.concatenate(e2es), np.concatenate(dets), n_ev,
                      rate, Q, Q_grid, Kd)


def summarize(r: CellResult, T_ms: float, c: float, L_srv_ms: float = 0.0, eps: float = 1e-3,
              delta_ms: float | None = None) -> dict:
    """P0 field set (+ deadline fields when delta_ms is given). Bounds in ms."""
    Kd, Q = r.K_dev, r.Q
    L_sum = (Kd - 1) * TAU_F + T_ms + Q * TAU_F + L_srv_ms
    L_max = T_ms + L_srv_ms + TAU_F * max(Kd - 1 + c, Q)
    cov = r.dev <= Kd - 1
    lost = np.isnan(r.e2e)
    rc = r.e2e[cov & ~lost] + L_srv_ms
    rd = r.e2e_clip[cov & ~lost] + L_srv_ms
    out = {
        "rate": r.rate, "util": r.rate * c, "stable": bool(r.rate * c < 1.0),
        "Q_slots": Q, "device_K": Kd,
        "n_viol_deterministic_clipT": int(np.sum(rd > L_max + 1e-9)),
        "bound_sum_ms": L_sum, "bound_max_ms": L_max,
        "n_events": int(r.n_events), "n_caught": int(len(r.e2e)), "n_lost_frames": int(lost.sum()),
        "n_device_covered": int(cov.sum()), "n_eval": int(len(rc)),
        "replay_max_ms": float(rc.max()) if len(rc) else None,
        "replay_q999_ms": float(np.quantile(rc, 1 - eps)) if len(rc) else None,
        "n_viol_max_bound": int(np.sum(rc > L_max)),
        "n_viol_sum_bound": int(np.sum(rc > L_sum)),
        "viol_rate_max_bound": float(np.mean(rc > L_max)) if len(rc) else None,
        "max_over_bound_ratio": float(rc.max() / L_max) if len(rc) else None,
        "tightness_sum": float(rc.max() / L_sum) if len(rc) else None,
        "mean_e2e_ms": float(np.nanmean(r.e2e) + L_srv_ms) if len(r.e2e) else None,
        "mean_device_ms": float(r.dev.mean() * TAU_F) if len(r.dev) else None,
        "miss_rate": 1 - len(r.e2e) / max(r.n_events, 1)}
    if delta_ms is not None:
        e = r.e2e + L_srv_ms
        out.update({"Delta_ms": delta_ms, "certificate_holds": bool(L_max <= delta_ms),
                    "n_deadline_viol_received": int(np.sum(e[~lost] > delta_ms)),
                    "n_deadline_viol_incl_lost": int(np.sum(e[~lost] > delta_ms) + lost.sum()),
                    "deadline_viol_rate_incl_lost": float((np.sum(e[~lost] > delta_ms) + lost.sum())
                                                          / max(len(e), 1))})
    return out


def replay(session: str, c: float, policy: str, K: int, a: float, b: float | None, delta_ms: float,
           L_srv_ms: float, T_ms: float, seed: int = 42, eps: float = 1e-3) -> dict:
    """Public entry point: one (session, c, policy, K, a, b, Delta, L_srv) cell, T given (use
    service_curve.certified_T for the Prop-5 certificate). Returns summary + per-event arrays."""
    rng = np.random.default_rng(seed)
    owd, _, meta = load_owd(session, rng)
    vids = load_activations(K, a)
    opens = policy_opens(vids, policy, K, a, b)
    r = replay_cell(vids, opens, owd, T_ms, c, meta["packet_interval_ms"], rng, policy, K)
    out = summarize(r, T_ms, c, L_srv_ms, eps, delta_ms)
    out.update({"session": session, "c": c, "policy": policy, "K": K, "a": a, "b": b, "T_ms": T_ms,
                "L_srv_ms": L_srv_ms, "seed": seed})
    return {"summary": out, "dev_slots": r.dev, "e2e_ms": r.e2e + L_srv_ms}
