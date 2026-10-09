"""P3 / Dot 3a: frozen final numbers, numbers.json and the six paper figures (topic #19). seed 42.

Headline (CP2 decision #2): r* on the Gilbert-Elliott FRAME-TIME scale (run_full.py --rmodel ge_time).
Sensitivity columns: independence (P2 run, p2_main_*) and GE trace-index scale (--rmodel ge_trace).
Bootstrap B = 1000, seed 42 (sessions resampled; videos resampled), 95 % percentile CIs.

Inputs : results/{p3_main_getime,p3_main_getrace,p2_main}_cells.csv, *_crosssession.json, *_ge_fit.json,
         THS_DATASETS/_derived/19/p3_main_getime_pervideo.npz, results/p1_*.json, p2_fec_sensitivity.json,
         p2_grid_owd.json, p0_owd_stats.json, p1_quality_grid.csv
Outputs: results/p3_main_summary.json, results/numbers.json (key -> {value, tex, pdf}),
         results/fig_F1..F6_*.pdf (IEEE one column, vector, fonts >= 7 pt)
"""
from __future__ import annotations

import json
import math
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.figure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

_savefig = matplotlib.figure.Figure.savefig


def _savefig_reproducible(self, *a, **k):
    """No creation date in the PDF, so figure hashes are identical across reruns (C7, C8b)."""
    k.setdefault("metadata", {"CreationDate": None})
    return _savefig(self, *a, **k)


matplotlib.figure.Figure.savefig = _savefig_reproducible
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import fec  # noqa: E402
import nc_bounds as nb  # noqa: E402
import service_curve as sc  # noqa: E402
from paths import DERIVED, RESULTS  # noqa: E402
from run_full import r_for  # noqa: E402

B, SEED = 1000, 42
EPS_FRAME = 1e-3
TAU_F = 1000.0 / 30.0
HEAD = "p3_main_getime"
SENS = {"indep": "p2_main", "ge_trace": "p3_main_getrace", "ge_time": "p3_main_getime"}
N_GRID = (35, 100, 300)

# dataviz reference palette, light mode, first three categorical slots (validated all-pairs);
# every series also carries a marker / line style (aqua is below 3:1 contrast -> secondary encoding)
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = {"token_bucket": S1, "gate_refresh": S2, "periodic": S3}
LBL = {"token_bucket": "token bucket (Alg. 1')", "gate_refresh": "gate+refresh (b = ∞)",
       "periodic": "periodic, same rate"}
MOD = {"indep": ("independent", S1, "o", "-"), "ge_trace": ("GE, trace scale", S3, "s", "--"),
       "ge_time": ("GE, frame-time scale", S2, "^", "-")}
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.5, "axes.spines.top": False,
                     "axes.spines.right": False, "lines.linewidth": 1.4, "font.family": "DejaVu Sans",
                     "pdf.fonttype": 42, "mathtext.default": "regular"})
W1 = 3.45   # IEEE one-column width [in] (3.5 in minus a small margin)


# ------------------------------------------------------------------------------------------- numbers
NUM: dict[str, dict] = {}


def _sci(x: float, sig: int = 2) -> tuple[str, str]:
    if x == 0:
        return "0", "0"
    e = int(math.floor(math.log10(abs(x))))
    m = round(x / 10 ** e, sig - 1)
    if abs(m) >= 10:
        m /= 10; e += 1
    ms = f"{m:.{sig - 1}f}"
    if e == 0:
        return ms, ms
    return rf"\ensuremath{{{ms}\times10^{{{e}}}}}", f"{ms}×10{'−' if e < 0 else ''}{abs(e)}"


def _int(n: int) -> tuple[str, str]:
    s = f"{int(n):,}"
    return s.replace(",", r"\,"), s.replace(",", "")


def put(key: str, value, kind: str = "auto", d: int = 2) -> None:
    """Register a number printed in the paper. kind: sci | int | fix | frac | str."""
    assert re.fullmatch(r"[A-Za-z][A-Za-z0-9.]*", key), key
    if isinstance(value, float) and not math.isfinite(value):
        kind, value = "str", "n/a"
    if kind == "auto":
        kind = "int" if isinstance(value, (int, np.integer)) else "sci"
    if kind == "sci":
        tex, pdf = _sci(float(value), d)
    elif kind == "int":
        tex, pdf = _int(int(value))
    elif kind == "fix":
        tex = pdf = f"{float(value):.{d}f}"
        if tex.startswith("-"):                       # typographic minus, not a hyphen
            tex, pdf = rf"\ensuremath{{-{tex[1:]}}}", "−" + pdf[1:]
    else:
        tex = pdf = str(value)
    NUM[key] = {"value": value if not isinstance(value, np.generic) else value.item(), "tex": tex,
                "pdf": pdf.replace(" ", "")}


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def boot_rate_sessions(g, rng):
    s = g.groupby("session")[["n_viol", "n_caught"]].sum()
    v, n = s.n_viol.to_numpy(), s.n_caught.to_numpy()
    idx = rng.integers(0, len(s), (B, len(s)))
    return ci(v[idx].sum(1) / np.maximum(n[idx].sum(1), 1))


def boot_rate_videos(ev, vi, rng):
    e, v = ev.sum(0), vi.sum(0)
    idx = rng.integers(0, len(e), (B, len(e)))
    return ci(v[idx].sum(1) / np.maximum(e[idx].sum(1), 1))


def load(base):
    df = pd.read_csv(RESULTS / f"{base}_cells.csv")
    df["row"] = np.arange(len(df))
    df["certified"] = df.certified.astype(bool)
    return df


def holdout(df, rng, EV=None, VI=None):
    cert = df[(df.policy == "token_bucket") & df.certified & (df.n_caught > 0)]
    tab, per = {}, {}
    for (d, e, N), g in cert.groupby(["dir", "eps_lat", "N"]):
        ps = g.groupby("session")[["n_viol", "n_caught"]].sum()
        prate = ps.n_viol / ps.n_caught
        k = f"{d}|eps_lat={e:g}|N={N}"
        tab[k] = {"cells": int(len(g)), "sessions": int(g.session.nunique()), "caught": int(g.n_caught.sum()),
                  "viol": int(g.n_viol.sum()), "undecodable": int(g.n_undecodable.sum()),
                  "rate": float(g.n_viol.sum() / g.n_caught.sum()),
                  "cp_upper_95": sc.cp_upper(int(g.n_viol.sum()), int(g.n_caught.sum()), 0.05),
                  "ci_sessions": boot_rate_sessions(g, rng),
                  "sessions_rate_le_eps": int(np.sum(prate <= EPS_FRAME)),
                  "sessions_over_eps": sorted(prate[prate > EPS_FRAME].index.tolist()),
                  "max_session_rate": float(prate.max())}
        if EV is not None:
            tab[k]["ci_videos"] = boot_rate_videos(EV[g.row], VI[g.row], rng)
        per[k] = {s: {"caught": int(r.n_caught), "viol": int(r.n_viol), "rate": float(r.n_viol / r.n_caught),
                      "cp_upper_95": sc.cp_upper(int(r.n_viol), int(r.n_caught), 0.05)} for s, r in ps.iterrows()}
    return cert, tab, per


def ge_from(d):
    return fec.GE(d["p_gb"], d["p_bg"], d["h"])


def main() -> None:
    rng = np.random.default_rng(SEED)
    grid = json.load(open(RESULTS / "p2_grid_owd.json"))["sessions"]
    iv = {s: v["rounds"][0]["interval_ms"] for s, v in grid.items()}
    UL = [s for s in grid if s.endswith("UL")]
    S = {"seed": SEED, "B": B, "headline": "r* Gilbert-Elliott frame-time scale (CP2 #2)", "inputs": SENS}

    df = load(HEAD)
    pv = np.load(DERIVED / f"{HEAD}_pervideo.npz")
    EV, VI = pv["eval"], pv["viol"]
    assert len(EV) == len(df)
    S["cells"] = int(len(df))
    cert, tab, per = holdout(df, rng, EV, VI)
    S["holdout_temporal_certified_tb"] = tab
    S["holdout_per_session"] = per

    # ---- stop rule: >= 4/13 UL sessions above 1e-3 at the headline (eps_lat = 1e-3)
    stop = {N: len(tab[f"UL|eps_lat=0.001|N={N}"]["sessions_over_eps"]) for N in N_GRID}
    S["stop_rule"] = {"UL_sessions_over_1e-3": stop, "triggered": any(v >= 4 for v in stop.values())}

    # ---- sensitivity columns (independence = P2, GE trace)
    sens = {}
    for m, base in SENS.items():
        if m == "ge_time":
            sens[m] = {k: {kk: v[kk] for kk in ("cells", "caught", "viol", "rate", "cp_upper_95",
                                                  "sessions_rate_le_eps", "sessions")} for k, v in tab.items()}
            continue
        d2 = load(base)
        _, t2, _ = holdout(d2, np.random.default_rng(SEED))
        sens[m] = {k: {kk: v[kk] for kk in ("cells", "caught", "viol", "rate", "cp_upper_95",
                                              "sessions_rate_le_eps", "sessions")} for k, v in t2.items()}
    S["sensitivity_r_model"] = sens

    # ---- r* tables
    rst = {}
    gefit = {m: json.load(open(RESULTS / f"{SENS[m]}_ge_fit.json")) for m in ("ge_time", "ge_trace")}
    for e in ("0.001", "0.0001"):
        for N in N_GRID:
            for c in ("0.6", "1.09", "1.5", "2.5"):
                rr = {s: gefit["ge_time"][s][e][f"r|N={N}"][c] for s in UL}
                used = [v[0] for v in rr.values()]; ind = [v[1] for v in rr.values()]
                rt = [gefit["ge_trace"][s][e][f"r|N={N}"][c][0] for s in UL]
                rst[f"eps_lat={e}|N={N}|c={c}"] = {
                    "indep": [min(ind), max(ind)], "ge_trace": [min(rt), max(rt)],
                    "ge_time": [min(used), max(used)], "ge_time_median": float(np.median(used)),
                    "ge_time_le_0.2N": int(sum(0 <= u <= 0.2 * N for u in used)),
                    "ge_time_gt_2x_indep": int(sum(u > 2 * i for u, i in zip(used, ind))),
                    "sessions_over_0.2N": [s for s in UL if not (0 <= rr[s][0] <= 0.2 * N)],
                    "per_session": {s: rr[s][0] for s in UL}}
    S["r_star_UL"] = rst

    # ---- cross-session (radio level)
    cs = json.load(open(RESULTS / f"{HEAD}_crosssession.json"))
    cst = {}
    for k, t in cs.items():
        for d in ("UL", "DL"):
            off = {kk: v for kk, v in t.items() if kk.split("->")[0].endswith(d)
                   and kk.split("->")[0] != kk.split("->")[1]}
            r = np.array([v["rate"] for v in off.values()])
            worst = max(off, key=lambda kk: off[kk]["rate"])
            bad_cal = pd.Series([kk.split("->")[0] for kk, v in off.items() if v["rate"] > EPS_FRAME]).value_counts()
            bad_tst = pd.Series([kk.split("->")[1] for kk, v in off.items() if v["rate"] > EPS_FRAME]).value_counts()
            cst[f"{k}|{d}"] = {"pairs": len(off), "pairs_le_eps": int(np.sum(r <= EPS_FRAME)),
                               "median_rate": float(np.median(r)), "max_rate": float(r.max()), "worst_pair": worst,
                               "worst_calibrators": bad_cal.head(3).to_dict(), "worst_tested": bad_tst.head(4).to_dict()}
    S["holdout_cross_session_radio"] = cst

    # ---- tightness
    tt = cert.dropna(subset=["tightness"])
    tight = {}
    for (d, N), g in tt.groupby(["dir", "N"]):
        med_s = g.groupby("session").tightness.median().to_numpy()
        idx = rng.integers(0, len(med_s), (B, len(med_s)))
        tight[f"{d}|N={N}"] = {"cells": int(len(g)), "median": float(g.tightness.median()),
                               "q10": float(g.tightness.quantile(.1)), "max": float(g.tightness.max()),
                               "ci_median_sessions": ci(np.median(med_s[idx], 1)),
                               "decoded_over_bound": int(g.n_over_bound_decoded.sum())}
    S["tightness_certified_tb"] = tight

    # ---- admission map + gating vs periodic (UL, eps_lat = 1e-3, N = 100)
    m = df[(df.eps_lat == 1e-3) & (df.N == 100) & (df.dir == "UL")]
    surv = m.groupby(["policy", "Delta", "c"]).certified.mean().unstack("c")
    S["admission_map_UL_eps1e-3_N100"] = {p: {f"{D:g}": {f"{c:g}": float(v) for c, v in row.items()}
                                              for D, row in surv.loc[p].iterrows()} for p in COL}
    piv = m.pivot_table(index=["session", "Delta", "c", "a_q"], columns="policy",
                        values=["certified", "miss_rate", "mean_e2e"])
    both = piv.dropna()
    gp = {"cells": int(len(both))}
    ses = both.index.get_level_values("session").to_numpy(); us = np.unique(ses)
    idx = rng.integers(0, len(us), (B, len(us)))
    for x, y in (("token_bucket", "periodic"), ("gate_refresh", "periodic"), ("token_bucket", "gate_refresh")):
        cx, cy = both["certified"][x].astype(bool), both["certified"][y].astype(bool)
        gp[f"{x}_cert_{y}_not"] = int((cx & ~cy).sum()); gp[f"{y}_cert_{x}_not"] = int((cy & ~cx).sum())
        gp[f"both_cert_{x}_{y}"] = int((cx & cy).sum())
        dm = (both["miss_rate"][x] - both["miss_rate"][y]).to_numpy()
        pm = np.array([dm[ses == u].mean() for u in us])
        gp[f"miss_{x}_minus_{y}"] = {"mean": float(dm.mean()), "ci_sessions": ci(pm[idx].mean(1))}
        de = (both["mean_e2e"][x] / both["mean_e2e"][y]).to_numpy()
        pe = np.array([np.median(de[ses == u]) for u in us])
        gp[f"e2e_ratio_{x}_over_{y}"] = {"median": float(np.median(de)), "ci_sessions": ci(np.median(pe[idx], 1))}
    gp["certified_per_policy"] = {p: int(both["certified"][p].astype(bool).sum()) for p in COL}
    S["gating_vs_periodic_UL_eps1e-3_N100"] = gp

    # ---- price of the link (version B), r from the headline GE frame-time model at N = 100
    q = pd.read_csv(RESULTS / "p1_quality_grid.csv")
    m_levels = [float(x) for x in np.round(np.quantile(q.miss, [0.1, 0.25, 0.5]), 4)]
    sT = df[(df.dir == "UL") & (df.eps_lat == 1e-3)].groupby("session").T_ms.first()
    sp = df[(df.dir == "UL") & (df.eps_lat == 1e-3)].groupby("session").p.first()
    Dp, Ls = 500.0, 20.0
    Kd = nb.k_dev(Dp, Ls, TAU_F)
    C_PRICE = (0.6, 0.9, 1.0, 1.09, 1.25, 1.5, 2.0, 2.5)
    price = {}
    for mq in m_levels:
        okq = q[q.miss <= mq]
        rho_dev = okq[okq.K <= Kd].rate.min()
        for c in C_PRICE:
            vals = []
            for s in sT.index:
                T = float(sT[s])
                r = r_for("ge_time", 100, float(sp[s]), c, ge_from(gefit["ge_time"][s]["0.001"]), iv[s])[0]
                ce = fec.c_eff(c, 100, r)
                Km = nb.k_max(Dp, T, Ls, ce, TAU_F)
                nob = okq[(okq.K <= Km) & (okq.rate * ce < 1)]

                def certified(row):
                    if row.K > Km:
                        return False
                    if ce <= 1:
                        return True
                    if row.kind == "refresh":
                        return nb.sigma_tb(int(row.K), 0.0, 0.0, 1 / ce) <= nb.burst_budget(Dp, T, Ls, ce, TAU_F) + 1e-9
                    if not math.isfinite(row.b):
                        return False
                    return nb.feasible(int(row.K), float(row.a), float(row.b), Dp, T, Ls, ce, TAU_F)
                cer = okq[[certified(rw) for rw in okq.itertuples()]]
                rn = nob.rate.min() if len(nob) else math.nan
                rl = cer.rate.min() if len(cer) else math.nan
                vals.append((rl - rn, rn - rho_dev, ce))
            v = np.array(vals, float)
            fin = v[np.isfinite(v[:, 0]), 0]
            ii = rng.integers(0, len(fin), (B, len(fin))) if len(fin) else None
            price[f"m_q={mq}|c={c}"] = {"sessions": int(len(v)), "sessions_certifiable": int(len(fin)),
                                       "price_burst_mean": float(fin.mean()) if len(fin) else None,
                                       "price_burst_max": float(fin.max()) if len(fin) else None,
                                       "ci_sessions": ci(fin[ii].mean(1)) if len(fin) else None,
                                       "price_shift_mean": float(np.nanmean(v[:, 1])),
                                       "c_eff_range": [float(v[:, 2].min()), float(v[:, 2].max())]}
    S["price_of_link_vB_Delta500_N100"] = {"m_q_levels": m_levels, "c": C_PRICE, "table": price}

    # ---- fixed measurement facts carried from P0-P2 (unchanged inputs)
    owd = json.load(open(RESULTS / "p0_owd_stats.json"))
    svc = json.load(open(RESULTS / "p1_service_curve.json"))
    fs = json.load(open(RESULTS / "p2_fec_sensitivity.json"))
    cest = json.load(open(RESULTS / "p1_c_estimate.json"))
    bt = json.load(open(RESULTS / "p1_burst_tax.json"))
    S["inputs_fixed"] = {"fec_sensitivity_summary": fs["summary"]}
    json.dump(S, open(RESULTS / "p3_main_summary.json", "w", encoding="utf-8"), indent=1, default=float)

    # ======================================================================== numbers.json
    put("nSessions", len(grid)); put("nUL", len(UL)); put("nDL", len(grid) - len(UL))
    put("nOWD", int(sum(v["n"] for v in owd["sessions"].values())))
    put("nGrid", int(sum(v["slots"] for v in grid.values())))
    put("durationH", sum(v["duration_h"] for v in owd["sessions"].values()), "fix", 1)
    put("nEvents", 3170); put("nVideos", 121); put("nFrames", 166589)
    put("nCells", len(df)); put("Bboot", B); put("seed", SEED)
    loss = [v["eps_loss_full"] for v in svc["sessions"].values()]
    put("lossFloorMin", min(loss), "sci", 3); put("lossFloorMax", max(loss), "sci", 3)
    tcal = df[(df.eps_lat == 1e-3)].groupby("session").T_ms.first()
    put("ThatULmin", tcal[UL].min(), "fix", 1); put("ThatULmax", tcal[UL].max(), "fix", 1)
    dl = [s for s in grid if s.endswith("DL")]
    put("ThatDLmin", tcal[dl].min(), "fix", 1); put("ThatDLmax", tcal[dl].max(), "fix", 1)
    lr_loss = [v["ge_loss"]["LR"] for v in fs["sessions"].values()]
    lr_bad = [v["eps"]["0.001"]["ge_bad"]["LR_vs_bernoulli"] for v in fs["sessions"].values()]
    put("LRlossMin", round(min(lr_loss)), "int"); put("LRlossMax", round(max(lr_loss)), "int")
    put("LRbadMin", round(min(lr_bad)), "int"); put("LRbadMax", round(max(lr_bad)), "int")
    put("lossRunsGeTwoExp", sum(v["loss_runs"]["expected_len_ge2_geometric"] for v in fs["sessions"].values()), "fix", 1)
    put("lossRunsGeTwoObs", int(sum(v["loss_runs"]["n_len_ge2"] for v in fs["sessions"].values())))
    put("cHDmedian", 1.09, "fix"); put("cSAlow", 1.07, "fix")
    put("RallMedian", cest["ul_scenarios"]["walk_all"]["q50"], "fix", 1)
    put("RsaMedian", cest["ul_scenarios"]["walk_TMobile_SA-lowband(n71)"]["q50"], "fix", 1)
    put("nRsec", cest["ul_scenarios"]["walk_all"]["n"])
    p0 = pd.read_csv(RESULTS / "p0_sanity_b_cells.csv")              # sum vs max composition (P0 replay, s1-UL)
    p0 = p0[p0.stable & (p0.n_eval > 0) & p0.policy.str.startswith("token") & (p0.c_slots == 1.5)]
    put("sumCells", len(p0)); put("sumTight", p0.tightness_sum.median(), "fix", 3)
    put("maxTight", p0.max_over_bound_ratio.median(), "fix", 3)
    put("sumGapMs", (p0.bound_sum_ms - p0.bound_max_ms).median(), "fix", 0)
    rg = json.load(open(RESULTS / "p2_asmag_regen_check.json"))        # LASIESTA gate-score regeneration
    put("lasFrames", rg["rows_regen"]); put("lasDiff", rg["score_n_diff_gt_1e-6"])
    put("lasMaxDiff", rg["score_max_abs_diff"], "fix", 2); put("lasEvents", rg["events_regen"])
    put("lasSpearman", rg["diagnosis"]["spearman"], "fix", 3)
    s0a = json.load(open(RESULTS / "p3_seed0_check.json"))                         # process 1: process, video, video
    s0b = json.load(open(RESULTS / "p3_seed0_check_fresh_video_video_repeat.json"))  # process 2: video, video
    assert s0a["video"]["vs_run_corr_bitwise_identical"] and s0b["video_repeat"]["vs_run_corr_bitwise_identical"]
    assert not s0b["video"]["vs_run_corr_bitwise_identical"]
    put("lasFirstPassDiff", s0b["video"]["vs_run_corr_n_diff_gt_1e-6"])
    put("lasSecondPassDiff", s0b["video_repeat"]["vs_run_corr_n_diff_gt_1e-6"])
    put("lasThirdPassDiff", s0a["video_repeat"]["vs_run_corr_n_diff_gt_1e-6"])
    put("lasProcesses", 2)
    sa, sb = bt["summary"]["1.07"], bt["summary"]["2.17"]
    put("sigGateLoA", sa["gate_pure_sigma_range"][0], "fix", 1); put("sigGateHiA", sa["gate_pure_sigma_range"][1], "fix", 1)
    put("sigPerLoA", sa["periodic_sigma_range"][0], "fix", 2); put("sigPerHiA", sa["periodic_sigma_range"][1], "fix", 2)
    put("sigGateLoB", sb["gate_pure_sigma_range"][0], "fix", 0); put("sigGateHiB", sb["gate_pure_sigma_range"][1], "fix", 0)
    put("sigPerLoB", sb["periodic_sigma_range"][0], "fix", 2); put("sigPerHiB", sb["periodic_sigma_range"][1], "fix", 2)
    put("cellsSigA", sa["cells_gate_rate_lt_1_over_c"]); put("cellsSigB", sb["cells_gate_rate_lt_1_over_c"])

    NW = {35: "a", 100: "b", 300: "c"}
    for (d, e) in (("UL", "0.001"), ("UL", "0.0001"), ("DL", "0.001"), ("DL", "0.0001")):
        tag = d + ("A" if e == "0.001" else "B")
        for N in N_GRID:
            t = tab.get(f"{d}|eps_lat={float(e):g}|N={N}")
            if t is None:
                continue
            k = f"ho{tag}{NW[N]}"
            put(k + "cells", t["cells"]); put(k + "caught", t["caught"]); put(k + "viol", t["viol"])
            put(k + "rate", t["rate"]); put(k + "cp", t["cp_upper_95"])
            put(k + "ciLo", t["ci_sessions"][0]); put(k + "ciHi", t["ci_sessions"][1])
            put(k + "vLo", t["ci_videos"][0]); put(k + "vHi", t["ci_videos"][1])
            put(k + "ok", f"{t['sessions_rate_le_eps']}/{t['sessions']}", "frac")
            put(k + "maxs", t["max_session_rate"])
            for mname, mk in (("indep", "I"), ("ge_trace", "T")):
                ts = sens[mname].get(f"{d}|eps_lat={float(e):g}|N={N}")
                if ts:
                    put(k + mk + "rate", ts["rate"]); put(k + mk + "cp", ts["cp_upper_95"])
                    put(k + mk + "ok", f"{ts['sessions_rate_le_eps']}/{ts['sessions']}", "frac")
    for e, et in (("0.01", "X"), ("0.001", "A"), ("0.0001", "B")):   # supplement: packet-level SLA (CP1 #1)
        sm = svc["summary"][e]
        tot = [svc["sessions"][s]["eps"][e]["full"]["eps_total"] for s in grid]
        put(f"pk{et}rmOK", f"{sm['roundmax_holdout_lat_ok']}/{sm['sessions']}", "frac")
        put(f"pk{et}poolOK", f"{sm['holdout_lat_ok']}/{sm['sessions']}", "frac")
        put(f"pk{et}Tratio", sm["roundmax_T_over_pooled_T_median"], "fix", 2)
        put(f"pk{et}totLo", min(tot)); put(f"pk{et}totHi", max(tot))
        for lvl, lk in ((1e-3, "a"), (3e-3, "b"), (1e-2, "c")):
            put(f"pk{et}le{lk}", f"{sum(t <= lvl for t in tot)}/{len(tot)}", "frac")
    for s in grid:                                        # supplement: GE fit on calibration rounds
        g_ = gefit["ge_time"][s]["0.001"]
        ks = "ge" + s.replace("-", "")
        put(ks + "pgb", g_["p_gb"]); put(ks + "pbg", g_["p_bg"], "fix", 3); put(ks + "h", g_["h"], "fix", 3)
        put(ks + "mb", g_["mean_bad"])
        for N in N_GRID:
            put(ks + NW[N], gefit["ge_time"][s]["0.001"][f"r|N={N}"]["1.09"][0])
            put(ks + NW[N] + "t", gefit["ge_trace"][s]["0.001"][f"r|N={N}"]["1.09"][0])
    for s in grid:                                        # Table III: per session, eps_lat = 1e-3
        ks = "ps" + s.replace("-", "")
        put(ks + "T", tcal[s], "fix", 1)
        rs = gefit["ge_time"][s]["0.001"]["r|N=100"]["1.09"]
        put(ks + "r", rs[0]); put(ks + "ri", rs[1])
        for N in N_GRID:
            v = per.get(f"{s[-2:]}|eps_lat=0.001|N={N}", {}).get(s)
            if v is not None:
                put(ks + NW[N] + "v", v["viol"]); put(ks + NW[N] + "n", v["caught"])
                put(ks + NW[N] + "cp", v["cp_upper_95"])
    for e, et in (("0.001", "A"), ("0.0001", "B")):
        for N in N_GRID:
            for c in ("0.6", "1.09", "1.5", "2.5"):
                rr = rst[f"eps_lat={e}|N={N}|c={c}"]
                k = f"r{et}{NW[N]}c{c.replace('.', '')}"
                put(k + "lo", rr["ge_time"][0]); put(k + "hi", rr["ge_time"][1])
                put(k + "iLo", rr["indep"][0]); put(k + "iHi", rr["indep"][1])
                put(k + "tLo", rr["ge_trace"][0]); put(k + "tHi", rr["ge_trace"][1])
                put(k + "ok", f"{rr['ge_time_le_0.2N']}/{len(UL)}", "frac")
    for k, v in cst.items():
        mm = re.match(r"eps_lat=([0-9.e-]+)\|N=(\d+)\|(UL|DL)", k)
        e, N, d = float(mm.group(1)), int(mm.group(2)), mm.group(3)
        kk = f"cs{d}{'A' if e == 1e-3 else 'B'}{NW[N]}"
        put(kk + "ok", f"{v['pairs_le_eps']}/{v['pairs']}", "frac"); put(kk + "med", v["median_rate"])
        put(kk + "max", v["max_rate"]); put(kk + "worst", v["worst_pair"].replace("->", "→"), "str")
    for mname, mk in (("indep", "I"), ("ge_trace", "T")):     # cross-session, sensitivity columns
        csm = json.load(open(RESULTS / f"{SENS[mname]}_crosssession.json"))
        for k, t in csm.items():
            e, N = float(k.split("|")[0].split("=")[1]), int(k.split("|")[1].split("=")[1])
            off = {kk: v for kk, v in t.items() if kk.split("->")[0].endswith("UL")
                   and kk.split("->")[0] != kk.split("->")[1]}
            r = np.array([v["rate"] for v in off.values()]); w = max(off, key=lambda kk: off[kk]["rate"])
            kk = f"cs{mk}{'A' if e == 1e-3 else 'B'}{NW[N]}"
            put(kk + "ok", f"{int(np.sum(r <= EPS_FRAME))}/{len(r)}", "frac"); put(kk + "max", float(r.max()))
            put(kk + "worst", w.replace("->", "→"), "str")
    for k, v in tight.items():
        d, N = k.split("|N=")
        kk = f"ti{d}{NW[int(N)]}"
        put(kk + "med", v["median"], "fix", 3); put(kk + "q", v["q10"], "fix", 2); put(kk + "max", v["max"], "fix", 2)
        put(kk + "lo", v["ci_median_sessions"][0], "fix", 3); put(kk + "hi", v["ci_median_sessions"][1], "fix", 3)
        put(kk + "cells", v["cells"])
    put("gpCells", gp["cells"])
    put("gpPerNotTB", gp["periodic_cert_token_bucket_not"]); put("gpTBNotPer", gp["token_bucket_cert_periodic_not"])
    put("gpPerNotGR", gp["periodic_cert_gate_refresh_not"]); put("gpGRNotPer", gp["gate_refresh_cert_periodic_not"])
    for p_, kk in (("token_bucket", "TB"), ("gate_refresh", "GR"), ("periodic", "Per")):
        put("gpCert" + kk, gp["certified_per_policy"][p_])
    for x, kk in (("token_bucket", "TB"), ("gate_refresh", "GR")):
        mmz = gp[f"miss_{x}_minus_periodic"]
        put(f"gpMiss{kk}", 100 * mmz["mean"], "fix", 2 if kk == "TB" else 1)
        put(f"gpMiss{kk}lo", 100 * mmz["ci_sessions"][0], "fix", 2 if kk == "TB" else 1)
        put(f"gpMiss{kk}hi", 100 * mmz["ci_sessions"][1], "fix", 2 if kk == "TB" else 1)
        ee = gp[f"e2e_ratio_{x}_over_periodic"]
        put(f"gpLat{kk}", ee["median"], "fix", 2); put(f"gpLat{kk}lo", ee["ci_sessions"][0], "fix", 2)
        put(f"gpLat{kk}hi", ee["ci_sessions"][1], "fix", 2)
    for D, row in S["admission_map_UL_eps1e-3_N100"]["token_bucket"].items():
        for c, v in row.items():
            for p_, kk in (("token_bucket", "TB"), ("gate_refresh", "GR"), ("periodic", "Per")):
                put(f"am{kk}D{D}c{c.replace('.', '')}", S["admission_map_UL_eps1e-3_N100"][p_][D][c], "fix", 2)
    for i, mq in enumerate(m_levels):
        put(f"mq{'abc'[i]}", mq, "fix", 2)
        for c in C_PRICE:
            t = price[f"m_q={mq}|c={c}"]
            kk = f"pr{'abc'[i]}c{str(c).replace('.', '')}"
            put(kk, t["price_burst_mean"] if t["price_burst_mean"] is not None else "n/a",
                "fix" if t["price_burst_mean"] is not None else "str", 4)
            put(kk + "n", f"{t['sessions_certifiable']}/{t['sessions']}", "frac")
            put(kk + "shift", t["price_shift_mean"], "fix", 4)
    json.dump(NUM, open(RESULTS / "numbers.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)

    # ======================================================================== figures (vector PDF)
    figs(df, grid, tcal, surv, gefit, rst, price, m_levels, C_PRICE, tt, UL)
    print(json.dumps({"stop_rule": S["stop_rule"],
                      "headline": {k: {kk: v[kk] for kk in ("rate", "cp_upper_95", "sessions_rate_le_eps",
                                                           "sessions_over_eps")}
                                   for k, v in tab.items() if k.startswith("UL")},
                      "r_le_0.2N": {k: (v["ge_time_le_0.2N"], v["sessions_over_0.2N"]) for k, v in rst.items()},
                      "n_numbers": len(NUM)}, indent=1, default=float))


def _box(ax, x, y, w, h, title, sub):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.04",
                                fc="#f2f1ed", ec=INK2, lw=0.7))
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=7.5, color=INK, weight="bold")
    ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=7, color=INK2)


def _arrow(ax, x0, x1, y, lab=None, ly=None):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=7, lw=0.8, color=INK2))
    if lab:
        ax.text((x0 + x1) / 2, ly if ly is not None else y + 0.06, lab, ha="center", va="bottom", fontsize=7, color=INK2)


def figs(df, grid, tcal, surv, gefit, rst, price, m_levels, C_PRICE, tt, UL):
    # F1 architecture + model
    # labels use no sub/superscripts, so every glyph is >= 7 pt at print size (C8)
    fig, ax = plt.subplots(figsize=(W1, 0.95))
    ax.set_xlim(0, 3.45); ax.set_ylim(0, 0.95); ax.axis("off")
    w, h, y = 0.72, 0.78, 0.08
    xs = [0.01, 0.84, 1.67, 2.72]
    for x0, ttl, sub in zip(xs, ("camera", "shaper", "FEC", "edge"),
                            ("gate,\nrefresh K", "bucket\na, b", "N data,\nr repair", "detector")):
        ax.add_patch(FancyBboxPatch((x0, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.04",
                                    fc="#f2f1ed", ec=INK2, lw=0.7))
        ax.text(x0 + w / 2, y + h - 0.16, ttl, ha="center", va="center", fontsize=7.5, color=INK, weight="bold")
        ax.text(x0 + w / 2, y + 0.27, sub, ha="center", va="center", fontsize=7, color=INK2, linespacing=1.1)
    for i in range(2):
        _arrow(ax, xs[i] + w, xs[i + 1], y + h / 2)
    xm = (xs[2] + w + xs[3]) / 2
    ax.add_patch(FancyArrowPatch((xs[2] + w + 0.01, y + h / 2), (xs[3] - 0.01, y + h / 2), arrowstyle="-|>",
                                 mutation_scale=7, lw=1.8, color=S1))
    ax.text(xm, y + h / 2 + 0.07, "5G", ha="center", va="bottom", fontsize=7, color=S1, weight="bold")
    ax.text(xm, y + h / 2 - 0.07, "c, T̂", ha="center", va="top", fontsize=7, color=INK)
    fig.savefig(RESULTS / "fig_F1_architecture.pdf", bbox_inches="tight", pad_inches=0.01); plt.close(fig)

    # F2 CCDF + T_hat (needs the per-packet DET6G arrays: skipped in a level-1 reproduction without them)
    if not all((DERIVED / f"grid_{s}.npy").exists() for s in grid):
        print("F2 skipped: grid_<session>.npy not found (run parse_det6g.py and p2_grid_owd.py first)")
        figs_rest(df, grid, tcal, surv, gefit, rst, price, m_levels, C_PRICE, tt, UL)
        return
    fig, ax = plt.subplots(figsize=(W1, 2.15))
    for s in grid:
        g = np.load(DERIVED / f"grid_{s}.npy").astype(float)
        x = np.sort(g[~np.isnan(g)])
        cc = 1 - np.arange(len(x)) / len(x)
        up = s.endswith("UL")
        col, ls = (S1, "-") if up else (S2, "--")
        ax.loglog(x[::40], cc[::40], color=col, lw=0.7, ls=ls, alpha=0.85)
        ax.plot([tcal[s]], [1e-3], marker="o" if up else "s", ms=3.5, color=col, mec="white", mew=0.4)
    ax.axhline(1e-3, color=INK2, lw=0.6, ls=":")
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
    ax.xaxis.set_major_locator(FixedLocator([5, 10, 20, 30])); ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["5", "10", "20", "30"]); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("one-way delay x [ms]"); ax.set_ylabel("P(OWD > x)")
    ax.plot([], [], color=S1, label=f"uplink ({len(UL)} sessions)")
    ax.plot([], [], color=S2, ls="--", label=f"downlink ({len(grid) - len(UL)} sessions)")
    ax.plot([], [], "o", color=INK2, ms=3.5, label="certificate at ε = 0.001")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout(pad=0.2); fig.savefig(RESULTS / "fig_F2_ccdf.pdf"); plt.close(fig)
    figs_rest(df, grid, tcal, surv, gefit, rst, price, m_levels, C_PRICE, tt, UL)


def figs_rest(df, grid, tcal, surv, gefit, rst, price, m_levels, C_PRICE, tt, UL):
    # F3 admission map (Δ × c) per policy, small multiples sharing one sequential scale
    fig, axs = plt.subplots(1, 3, figsize=(W1, 1.75), sharey=True)
    for ax, p in zip(axs, COL):
        M = surv.loc[p]
        im = ax.imshow(M.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto", origin="lower")
        ax.set_xticks(range(M.shape[1])); ax.set_xticklabels([f"{c:g}" for c in M.columns])
        ax.set_yticks(range(M.shape[0])); ax.set_yticklabels([f"{d:g}" for d in M.index])
        ax.set_title({"token_bucket": "token bucket", "gate_refresh": "gate+refresh",
                      "periodic": "periodic"}[p]); ax.set_xlabel("c"); ax.grid(False)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                val = M.to_numpy()[i, j]
                ax.text(j, i, "1" if val == 1 else ("0" if val == 0 else f"{val:.2f}".lstrip("0")),
                        ha="center", va="center", fontsize=7, color="white" if val > .55 else INK)
    axs[0].set_ylabel("Δ [ms]")
    fig.tight_layout(pad=0.2, w_pad=0.3); fig.savefig(RESULTS / "fig_F3_admission.pdf"); plt.close(fig)

    # F4 r* vs N under independence / GE trace / GE frame time (13 UL, eps_lat = 1e-3, c = 1.09; bars = range)
    fig, ax = plt.subplots(figsize=(W1, 2.0))
    off = {"indep": -0.06, "ge_trace": 0.0, "ge_time": 0.06}
    for mk, (lab, col, mkr, ls) in MOD.items():
        lo = [rst[f"eps_lat=0.001|N={N}|c=1.09"][mk][0] for N in N_GRID]
        hi = [rst[f"eps_lat=0.001|N={N}|c=1.09"][mk][1] for N in N_GRID]
        x = np.log10(N_GRID) + off[mk]
        mid = [np.sqrt(a * b) for a, b in zip(lo, hi)]
        ax.errorbar(x, mid, yerr=[np.subtract(mid, lo), np.subtract(hi, mid)], fmt=mkr, ls=ls, color=col,
                    ms=4, capsize=2, lw=1.0, label=lab)
    ax.plot(np.log10(N_GRID), [0.2 * N for N in N_GRID], color=INK2, lw=0.8, ls=":", label="0.2 N")
    ax.set_yscale("log"); ax.set_xticks(np.log10(N_GRID)); ax.set_xticklabels([str(N) for N in N_GRID])
    from matplotlib.ticker import FixedLocator, NullLocator
    ax.yaxis.set_major_locator(FixedLocator([2, 5, 10, 20, 50])); ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels(["2", "5", "10", "20", "50"])
    ax.set_xlabel("data packets per frame N"); ax.set_ylabel("repair packets r* (range, 13 UL)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(pad=0.2); fig.savefig(RESULTS / "fig_F4_rstar.pdf"); plt.close(fig)

    # F5 price of the link Π_burst vs c at three m_q
    fig, ax = plt.subplots(figsize=(W1, 1.95))
    for mq, col, mkr in zip(m_levels, (S1, S2, S3), ("o", "s", "^")):
        y = [price[f"m_q={mq}|c={c}"]["price_burst_mean"] for c in C_PRICE]
        y = [np.nan if v is None else v for v in y]
        ax.plot(C_PRICE, y, marker=mkr, ms=4, color=col, label=f"miss target {mq:.2f}")
    ax.axvline(1.0, color=INK2, lw=0.6, ls=":")
    ax.text(1.03, 0.012, "c = 1", fontsize=7, color=INK2, ha="left")
    ax.set_xlabel("link load factor c"); ax.set_ylabel("burst price [activations/slot]")
    ax.legend(frameon=False, loc="center right")
    fig.tight_layout(pad=0.2); fig.savefig(RESULTS / "fig_F5_price.pdf"); plt.close(fig)

    # F6 tightness
    fig, ax = plt.subplots(figsize=(W1, 1.75))
    pos, data, cols = [], [], []
    for i, N in enumerate(N_GRID):
        for j, (d, col) in enumerate((("UL", S1), ("DL", S2))):
            data.append(tt[(tt.dir == d) & (tt.N == N)].tightness.to_numpy()); pos.append(i + (j - 0.5) * 0.3)
            cols.append(col)
    bp = ax.boxplot(data, positions=pos, widths=0.24, showfliers=False, patch_artist=True)
    for b_, col in zip(bp["boxes"], cols):
        b_.set(facecolor="white", edgecolor=col, lw=1.0)
    for med, col in zip(bp["medians"], cols):
        med.set(color=col, lw=1.4)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"N = {N}" for N in N_GRID])
    ax.set_ylabel("max replayed latency / bound"); ax.set_ylim(0.6, 1.05)
    ax.plot([], [], color=S1, label="uplink"); ax.plot([], [], color=S2, label="downlink")
    ax.legend(frameon=False, loc="lower right", ncol=2)
    fig.tight_layout(pad=0.2); fig.savefig(RESULTS / "fig_F6_tightness.pdf"); plt.close(fig)


if __name__ == "__main__":
    main()
