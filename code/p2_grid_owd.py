"""P2: per-session OWD sequence ON THE SEND GRID (true positions of lost packets). topic #19.

For each DET6G session and 30-min round: slot = round((t_send - t_first) / interval); the grid has
round((t_last - t_first)/interval) + 1 slots (the P0 loss rule). Received packets fill their slot
with the OWD [ms]; empty slots and rows with NaN delay are lost (NaN). Duplicate / off-grid sends
are counted and reported (expected 0).

Output: THS_DATASETS/_derived/19/grid_<session>.npy (float32, rounds concatenated)
        results/p2_grid_owd.json (per round: slots, received, lost, offgrid, duplicates)
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from paths import DERIVED, DET6G, RESULTS

ROOT = DET6G / "COTS5G measurements"
COLS = ["timestamps.client.send.wall", "delay.send", "packet_interval"]


def main() -> None:
    out = {"rule": "slot = round((t - t_first)/interval); lost = empty slot or NaN delay", "sessions": {}}
    folders = sorted([p for p in ROOT.iterdir() if p.is_dir()],
                     key=lambda p: int(re.match(r"s(\d+)", p.name).group(1)))
    for folder in folders:
        parts, rounds = [], []
        for f in sorted(folder.glob("*.parquet")):
            d = pd.read_parquet(f, columns=COLS)
            t = d["timestamps.client.send.wall"].to_numpy(np.int64)
            iv = int(d["packet_interval"].iloc[0])
            pos = np.round((t - t[0]) / iv).astype(np.int64)
            off = np.abs((t - t[0]) - pos * iv) > iv / 4
            n = int(pos[-1]) + 1
            g = np.full(n, np.nan, dtype=np.float32)
            dup = int(len(pos) - len(np.unique(pos)))
            g[pos] = (d["delay.send"].to_numpy() / 1e6).astype(np.float32)
            parts.append(g)
            rounds.append({"file": f.name, "slots": n, "rows": len(d), "received": int(np.sum(~np.isnan(g))),
                           "lost": int(np.sum(np.isnan(g))), "offgrid": int(off.sum()), "duplicates": dup,
                           "interval_ms": iv / 1e6})
        g = np.concatenate(parts)
        np.save(DERIVED / f"grid_{folder.name}.npy", g)
        out["sessions"][folder.name] = {"slots": int(len(g)), "lost": int(np.isnan(g).sum()),
                                        "loss": float(np.isnan(g).mean()), "rounds": rounds}
        print(folder.name, len(g), int(np.isnan(g).sum()), f"{np.isnan(g).mean():.2e}",
              "offgrid", sum(r["offgrid"] for r in rounds), "dup", sum(r["duplicates"] for r in rounds), flush=True)
    json.dump(out, open(RESULTS / "p2_grid_owd.json", "w"), indent=1)


if __name__ == "__main__":
    main()
