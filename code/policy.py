"""Gate+refresh activation policies and their empirical burstiness (topic #19, P0 / T03).

Self-contained re-implementation (no code imported from other repositories) of the on-device
activation policy studied in

    Deep3: "certifiable worst-case missed-event latency for frame-skipping pipelines",
    IEEE Internet of Things Journal, 2026, DOI 10.1109/JIOT.2026.3740931.

Policies (one camera, frame index t = 0, 1, ...; "open" = frame is sent to the edge server):

* periodic(P)          open iff (t - phi) mod P == 0                      (rate 1/P, burst <= 1)
* periodic_same_rate   evenly spaced opens at the POOLED rate rho of gate+refresh(K, a), fractional
                       period 1/rho, random phase per video (seed 42)
* gate+refresh(K, a)   open iff t mod K == phi  OR  score_t > tau_a        (Deep3 style)
                       tau_a is ONE global threshold (pooled over the corpus) chosen so that the
                       gate fires on a fraction a of the non-refresh frames.
* token-bucket(K,a,b)  refresh as above; the gate may open a non-refresh frame only if a token
                       is available. Bucket depth b, refill a token/frame:
                           c_t = min(b, c_{t-1} + a);  open if score_t > tau_a and c_t >= 1
                           then c_t -= 1.     c_{-1} = b.
                       => in every window of w frames: #open <= ceil(w/K) + floor(b + a*w)  (T1).

Inputs
* g_t  : 1 if the detector reports >= 1 box on frame t. Built from the yolo26s detection dump in
         THS_DATASETS/_derived/detections_cpu_yolo26s. LASIESTA dumps index frames by the
         LEXICOGRAPHIC position of the file name ({vid}-1, -10, -100, ...); we re-key them to the
         true temporal index (file number - 1) before building runs (topic 17 note B0, 27-09-2026).
* score_t : ASMAG motion-gate score (data file published with Deep3). For LASIESTA we use the
         time-ordered re-run, because the original gate pass received lexicographically keyed
         detections through its temporal-persistence feedback.

Outputs
* THS_DATASETS/_derived/19/activations_K{K}_a{a}.parquet  (per frame: g_t, score, refresh, open_*)
* results/p0_burstiness.json  (empirical rate and minimal token-bucket burst sigma_hat per policy)
* results/p0_events_check.json (event counts vs topic-17 table: BMC 261, CDnet 2516, LASIESTA 393)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from paths import (DERIVED, DETECTIONS, GATE_SCORES, GATE_SCORES_LASIESTA_FIXED, LASIESTA_RAW,
                   RESULTS)

DATASETS_USED = ("BMC", "CDnet2014", "LASIESTA")
K_GRID = (2, 4, 8, 16, 32)
A_GRID = (0.05, 0.1, 0.2, 0.3, 0.5)
BUCKET_DEPTHS = (1, 4)
SEED = 42
EXPECTED_EVENTS = {"BMC": 261, "CDnet2014": 2516, "LASIESTA": 393}   # topic 17, B0 (27-09-2026)


# ---------------------------------------------------------------- inputs

def _lasiesta_numbers(vid: str) -> list[int]:
    """File numbers of {vid}-{n}.bmp in LEXICOGRAPHIC file-name order (= dump index order)."""
    d = LASIESTA_RAW / vid
    files = sorted(p.name for p in d.glob(f"{vid}-*.bmp"))
    return [int(re.search(r"-(\d+)\.bmp$", f).group(1)) for f in files]


def load_detections(ds: str) -> dict[str, np.ndarray]:
    """video -> g_t (int8) indexed by TRUE temporal frame index (offset by the first frame)."""
    out = {}
    for jp in sorted((DETECTIONS / ds).glob("*.json")):
        frames = json.load(open(jp))["frames"]
        idx = np.array([f["frame"] for f in frames])
        has = np.array([len(f.get("boxes", [])) > 0 for f in frames], dtype=np.int8)
        video = jp.stem.split("__", 1)[1]
        if ds == "LASIESTA":
            nums = _lasiesta_numbers(video)
            assert list(idx) == list(range(len(frames))), jp.name
            assert sorted(nums) == list(range(1, len(nums) + 1)), jp.name
            idx = np.array(nums)[idx] - 1                     # true temporal index, 0-based
        order = np.argsort(idx, kind="stable")
        idx, has = idx[order], has[order]
        assert np.all(np.diff(idx) == 1), f"non-contiguous frames in {jp.name}"
        out[video] = (int(idx[0]), has)
    return out


def load_scores() -> pd.DataFrame:
    pub = pd.read_parquet(GATE_SCORES)
    fix = pd.read_parquet(GATE_SCORES_LASIESTA_FIXED)
    las = fix[fix.dataset == "LASIESTA"]
    other = pub[pub.dataset.isin([d for d in DATASETS_USED if d != "LASIESTA"])]
    return pd.concat([other, las], ignore_index=True)


def build_corpus() -> tuple[pd.DataFrame, dict]:
    """Per-frame table (dataset, video, frame, g_t, score) on frames covered by both sources."""
    scores = load_scores()
    rows, check = [], {}
    for ds in DATASETS_USED:
        dets = load_detections(ds)
        n_ev = sum(int(np.sum(np.diff(np.r_[0, g]) == 1)) for _, g in dets.values())
        sc_ds = scores[scores.dataset == ds]
        n_match = n_tot = 0
        for video, (f0, g) in dets.items():
            s = sc_ds[sc_ds.video == video].sort_values("frame")
            if s.empty:
                continue
            fr = s.frame.to_numpy()
            ok = (fr >= f0) & (fr < f0 + len(g))
            s = s[ok]
            gt = g[s.frame.to_numpy() - f0]
            n_match += int(np.sum(gt == s.g_t.to_numpy()))
            n_tot += len(s)
            rows.append(pd.DataFrame({"dataset": ds, "video": video,
                                      "frame": s.frame.to_numpy().astype(np.int32),
                                      "g_t": gt.astype(np.int8),
                                      "score": s.score.to_numpy().astype(np.float32)}))
        check[ds] = {"videos": len(dets), "events_full_dump": n_ev,
                     "events_expected_topic17": EXPECTED_EVENTS[ds],
                     "events_match": n_ev == EXPECTED_EVENTS[ds],
                     "frames_with_score": n_tot,
                     "g_t_agreement_with_score_file": n_match / max(n_tot, 1)}
    df = pd.concat(rows, ignore_index=True)
    # keep only contiguous score segments (needed for window counts)
    df = df.sort_values(["dataset", "video", "frame"], ignore_index=True)
    return df, check


# ---------------------------------------------------------------- policies

def periodic(n: int, P: int, phase: int = 0) -> np.ndarray:
    o = np.zeros(n, dtype=np.int8)
    o[phase % P::P] = 1
    return o


def periodic_rate(n: int, rho: float, phi: float = 0.0) -> np.ndarray:
    """Evenly spaced opens at (possibly fractional) rate rho: gaps in {floor(1/rho), ceil(1/rho)}."""
    t = np.arange(n + 1)
    return np.diff(np.floor(t * rho + phi)).astype(np.int8)


def refresh_mask(n: int, K: int, phase: int = 0) -> np.ndarray:
    return periodic(n, K, phase)


def gate_refresh(score: np.ndarray, K: int, tau: float, phase: int = 0) -> np.ndarray:
    return np.maximum(refresh_mask(len(score), K, phase), (score > tau).astype(np.int8))


def token_bucket(score: np.ndarray, K: int, tau: float, a: float, b: float,
                 phase: int = 0) -> np.ndarray:
    ref = refresh_mask(len(score), K, phase)
    fire = score > tau
    o = ref.copy()
    c = float(b)
    for t in range(len(score)):
        c = min(b, c + a)
        if not ref[t] and fire[t] and c >= 1.0:
            o[t] = 1
            c -= 1.0
    return o


def calibrate_tau(df: pd.DataFrame, K: int, a: float) -> float:
    """Global tau such that the gate fires on a fraction a of the non-refresh frames (pooled)."""
    nonref = []
    for _, g in df.groupby(["dataset", "video"], sort=False):
        ref = refresh_mask(len(g), K)
        nonref.append(g.score.to_numpy()[ref == 0])
    s = np.concatenate(nonref)
    return float(np.quantile(s, 1.0 - a))


# ---------------------------------------------------------------- burstiness

def min_burst(opens: np.ndarray, r: float) -> float:
    """sigma(r) = max over windows [s, t) of (N(t) - N(s)) - r (t - s): minimal token-bucket burst."""
    x = np.concatenate([[0.0], np.cumsum(opens, dtype=float)]) - r * np.arange(len(opens) + 1)
    return float(np.max(x - np.minimum.accumulate(x)))


def max_window_counts(opens: np.ndarray, ws: np.ndarray) -> np.ndarray:
    c = np.concatenate([[0], np.cumsum(opens, dtype=np.int64)])
    return np.array([int(np.max(c[w:] - c[:-w])) if w <= len(opens) else int(c[-1]) for w in ws])


def t1_alpha(ws: np.ndarray, K: int, a: float, b: float) -> np.ndarray:
    return np.minimum(ws, np.ceil(ws / K) + np.floor(b + a * ws + 1e-12))


def main() -> None:
    df, check = build_corpus()
    json.dump(check, open(RESULTS / "p0_events_check.json", "w"), indent=1)
    print(json.dumps(check, indent=1))
    groups = [(k, g.reset_index(drop=True)) for k, g in df.groupby(["dataset", "video"], sort=False)]
    ws = np.unique(np.round(np.logspace(0, np.log10(6300), 44)).astype(int))
    burst = {"seed": SEED, "n_videos": len(groups), "n_frames": int(len(df)),
             "datasets": list(DATASETS_USED), "window_grid_frames": ws.tolist(),
             "note": "rates and bursts in frames (s = 1 frame, F = 1 frame/slot); sigma_hat(r) = "
                     "minimal token-bucket burst at rate r, max over videos",
             "cells": []}
    for K in K_GRID:
        for a in A_GRID:
            tau = calibrate_tau(df, K, a)
            rho = sum(int(gate_refresh(g.score.to_numpy(), K, tau).sum()) for _, g in groups) / len(df)
            rng = np.random.default_rng(SEED)
            out, per = [], {"gate_refresh": [], "periodic_same_rate": []}
            per.update({f"token_bucket_b{b}": [] for b in BUCKET_DEPTHS})
            wc = {k: np.zeros(len(ws), dtype=np.int64) for k in per}
            viol_t1 = {f"token_bucket_b{b}": 0 for b in BUCKET_DEPTHS}
            viol_t1_gr = {b: 0 for b in BUCKET_DEPTHS}
            n_open = {k: 0 for k in per}
            for (ds, vid), g in groups:
                sc = g.score.to_numpy()
                n = len(sc)
                o_gr = gate_refresh(sc, K, tau)
                pol = {"gate_refresh": o_gr,
                       "periodic_same_rate": periodic_rate(n, rho, float(rng.random()))}
                for b in BUCKET_DEPTHS:
                    pol[f"token_bucket_b{b}"] = token_bucket(sc, K, tau, a, b)
                for name, o in pol.items():
                    r = float(o.mean())
                    per[name].append((r, min_burst(o, r), min_burst(o, 1.0 / K + a)))
                    mw = max_window_counts(o, ws)
                    wc[name] = np.maximum(wc[name], mw)
                    n_open[name] += int(o.sum())
                    if name.startswith("token_bucket"):
                        b = float(name.rsplit("_b", 1)[1])
                        viol_t1[name] += int(np.sum(mw > t1_alpha(ws, K, a, b)))
                    if name == "gate_refresh":
                        for b in BUCKET_DEPTHS:
                            viol_t1_gr[b] += int(np.sum(mw > t1_alpha(ws, K, a, b)))
                out.append(pd.DataFrame({
                    "dataset": ds, "video": vid, "frame": g.frame.to_numpy(), "g_t": g.g_t.to_numpy(),
                    "score": sc, "refresh": refresh_mask(n, K),
                    **{f"open_{k}": v for k, v in pol.items()}}))
            pd.concat(out, ignore_index=True).to_parquet(
                DERIVED / f"activations_K{K}_a{a}.parquet", index=False)
            cell = {"K": K, "a": a, "tau": tau, "rho_gate_refresh_pooled": rho, "policies": {}}
            for name, lst in per.items():
                arr = np.array(lst)
                cell["policies"][name] = {
                    "rate_pooled": n_open[name] / len(df),
                    "rate_video_median": float(np.median(arr[:, 0])),
                    "sigma_hat_own_rate_max": float(arr[:, 1].max()),
                    "sigma_hat_own_rate_median": float(np.median(arr[:, 1])),
                    "sigma_hat_at_T1_rate_max": float(arr[:, 2].max()),
                    "max_window_count": wc[name].tolist()}
                if name in viol_t1:
                    cell["policies"][name]["T1_violations_window_video"] = viol_t1[name]
            cell["policies"]["gate_refresh"]["T1_violations_if_bucket_b"] = {
                str(b): v for b, v in viol_t1_gr.items()}
            burst["cells"].append(cell)
            p = cell["policies"]
            print(f"K={K:2d} a={a:.2f} tau={tau:.3f} | GR rate={p['gate_refresh']['rate_pooled']:.3f} "
                  f"sig={p['gate_refresh']['sigma_hat_own_rate_max']:.1f} | PER sig="
                  f"{p['periodic_same_rate']['sigma_hat_own_rate_max']:.2f} | TB4 rate="
                  f"{p['token_bucket_b4']['rate_pooled']:.3f} sig={p['token_bucket_b4']['sigma_hat_own_rate_max']:.1f}"
                  f" T1viol={p['token_bucket_b4']['T1_violations_window_video']}", flush=True)
    json.dump(burst, open(RESULTS / "p0_burstiness.json", "w"), indent=1)


if __name__ == "__main__":
    main()
