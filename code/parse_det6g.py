"""Parse DETERMINISTIC6G COTS 5G latency measurements (Zenodo 10.5281/zenodo.10390211, CC BY 4.0).

Layout: THS_DATASETS/DET6G_COTS5G/COTS5G measurements/s<k>-<UL|DL>/<ip>_<port>_<date>_<time>.parquet,
one file per ~30-min round, one folder per session (direction x UE x packet interval x payload,
see "COTS5G measurement campaign.xlsx"). Per packet: irtt send/receive wall clocks (PTP-synced),
`delay.send` = one-way delay [ns].

Loss: the `lost` column is the constant string "true" in every file we checked, so it carries no
information. Received packets are stored with contiguous seqno re-indexing, therefore loss is
estimated from the send-time grid: expected = round(span / interval) + 1 per round,
loss = 1 - received / expected, and the number of send gaps > 1.5 intervals is reported.

Outputs: results/p0_owd_stats.json, results/p0_owd_ccdf.png,
         THS_DATASETS/_derived/19/owd_<session>.npy (OWD sequence in ms, float32, for replay).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from paths import DERIVED, DET6G, RESULTS  # noqa: E402

ROOT = DET6G / "COTS5G measurements"
QS = (0.5, 0.9, 0.99, 0.999)
WIN_NS = 30 * 60 * 10**9   # stationarity window: 30 min (= one round)
COLS = ["timestamps.client.send.wall", "delay.send", "packet_length", "packet_interval", "trip"]


def campaign() -> pd.DataFrame:
    x = pd.read_excel(ROOT / "COTS5G measurement campaign.xlsx", sheet_name="EP5G industrial control")
    x["folder"] = x["Session #"].astype(str).str.replace(r"^(s\d+)$", r"\1", regex=True)
    return x


def load_session(folder: Path) -> tuple[pd.DataFrame, list[dict]]:
    parts, rounds = [], []
    for i, f in enumerate(sorted(folder.glob("*.parquet"))):
        d = pd.read_parquet(f, columns=COLS)
        t = d["timestamps.client.send.wall"].to_numpy(np.int64)
        iv = int(d["packet_interval"].iloc[0])
        exp = int(round((t[-1] - t[0]) / iv)) + 1
        gaps = int(np.sum(np.diff(t) > 1.5 * iv))
        rounds.append({"file": f.name, "n": len(d), "expected": exp,
                       "loss": 1.0 - len(d) / exp, "send_gaps": gaps})
        d["round"] = i
        parts.append(d)
    return pd.concat(parts, ignore_index=True), rounds


def qdict(x: np.ndarray) -> dict:
    q = np.quantile(x, QS)
    return {**{f"q{p}": float(v) for p, v in zip(QS, q)}, "max": float(x.max()),
            "min": float(x.min()), "mean": float(x.mean())}


def main() -> None:
    meta = campaign().set_index("Session #")
    out = {"source": "DETERMINISTIC6G COTS 5G latency measurements, Zenodo 10.5281/zenodo.10390211 "
                     "(CC BY 4.0)", "unit": "ms", "stationarity_window_min": 30,
           "loss_method": "1 - received/expected from send-time grid (column `lost` is constant)",
           "sessions": {}}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    folders = sorted([p for p in ROOT.iterdir() if p.is_dir()],
                     key=lambda p: int(re.match(r"s(\d+)", p.name).group(1)))
    for folder in folders:
        sid = folder.name.split("-")[0]
        d, rounds = load_session(folder)
        owd_all = d["delay.send"].to_numpy() / 1e6          # NaN = sent but not received
        t_all = d["timestamps.client.send.wall"].to_numpy(np.int64)
        np.save(DERIVED / f"owd_{folder.name}.npy", owd_all.astype(np.float32))
        rx = ~np.isnan(owd_all)
        owd, t = owd_all[rx], t_all[rx]
        # non-overlapping 30-min windows: quantiles per window (received packets)
        wid = (t - t_all[0]) // WIN_NS
        win = []
        for w in np.unique(wid):
            x = owd[wid == w]
            if len(x) >= 1000:
                win.append({"window": int(w), "n": int(len(x)),
                            **{f"q{p}": float(v) for p, v in zip(QS, np.quantile(x, QS))}})
        wq = pd.DataFrame(win)
        stat = {f"q{p}_range_over_windows": [float(wq[f"q{p}"].min()), float(wq[f"q{p}"].max())]
                for p in QS}
        stat["q0.999_rel_spread"] = float((wq["q0.999"].max() - wq["q0.999"].min()) / wq["q0.999"].median())
        stat["q0.5_rel_spread"] = float((wq["q0.5"].max() - wq["q0.5"].min()) / wq["q0.5"].median())
        stat["n_windows"] = int(len(wq))
        key = folder.name if folder.name in meta.index else sid
        m = meta.loc[key] if key in meta.index else None
        n_exp = sum(r["expected"] for r in rounds)
        n_nan = int((~rx).sum())
        n_grid_missing = max(0, n_exp - len(owd_all))
        out["sessions"][folder.name] = {
            "direction": str(d["trip"].iloc[0]), "ue": None if m is None else str(m["network"]).split(",")[0],
            "packet_interval_ms": int(d["packet_interval"].iloc[0]) / 1e6,
            "payload_B_file": int(d["packet_length"].iloc[0]),
            "payload_B_campaign_sheet": None if m is None else int(m["Payload size (B)"]),
            "network": None if m is None else str(m["network"]),
            "n": int(len(owd)), "n_rows": int(len(owd_all)), "n_nan_delay": n_nan,
            "n_expected_grid": int(n_exp), "n_missing_from_grid": int(n_grid_missing),
            "loss": (n_nan + n_grid_missing) / max(n_exp, len(owd_all)),
            "send_gaps_total": int(sum(r["send_gaps"] for r in rounds)),
            "n_rounds": len(rounds), "duration_h": float((t_all[-1] - t_all[0]) / 3.6e12),
            "owd": qdict(owd), "stationarity": stat, "windows": win, "rounds": rounds}
        s = out["sessions"][folder.name]
        print(f"{folder.name:7s} {s['direction']:8s} {str(s['ue']):6s} iv={s['packet_interval_ms']:4.0f}ms "
              f"B={s['payload_B_file']:4d} n={s['n']:8d} loss={s['loss']:.2e} q50={s['owd']['q0.5']:.2f} "
              f"q99={s['owd']['q0.99']:.2f} q999={s['owd']['q0.999']:.2f} max={s['owd']['max']:.1f} "
              f"q999 spread={stat['q0.999_rel_spread']:.2f}", flush=True)
        xs = np.sort(owd)
        cc = 1.0 - np.arange(len(xs)) / len(xs)
        ax = axes[0] if s["direction"] == "uplink" else axes[1]
        ax.loglog(xs[::50], cc[::50], lw=1, label=folder.name)
    for ax, ttl in zip(axes, ("Uplink", "Downlink")):
        ax.set_title(f"{ttl} one-way delay CCDF (DET6G COTS 5G)")
        ax.set_xlabel("OWD [ms]")
        ax.grid(True, which="both", alpha=.3)
        ax.legend(fontsize=6, ncol=2)
        ax.axhline(1e-3, color="k", ls=":", lw=.8)
    axes[0].set_ylabel("P(OWD > x)")
    fig.tight_layout()
    fig.savefig(RESULTS / "p0_owd_ccdf.png", dpi=150)
    out["n_sessions"] = len(out["sessions"])
    out["n_packets_total"] = int(sum(s["n"] for s in out["sessions"].values()))
    json.dump(out, open(RESULTS / "p0_owd_stats.json", "w"), indent=1)


if __name__ == "__main__":
    main()
