"""P3: does the published gate pass with numpy/random seeded to 0 reproduce run_corr (LASIESTA)? (topic #19)

CP2 decision #4 asks the paper to say the LASIESTA scores are "reproducible with seed 0". P2 only showed
that two seeded runs agree with each other. This script tests the stronger statement against the file
the paper actually uses (run_corr, read-only): gate pass of the Deep3 repo unchanged, detections re-keyed
in temporal order (as p2_regen_lasiesta_scores.py), seeds set (a) once per process, (b) before each video.
Output: results/p3_seed0_check.json; scores -> THS_DATASETS/_derived/19/scores_lasiesta_seed0_*.parquet
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np
import pandas as pd

from p2_regen_lasiesta_scores import rekey
from paths import DEEP3_REPO, DERIVED, DETECTIONS, GATE_SCORES_LASIESTA_FIXED, RESULTS


def regen(mode: str, gate_parts) -> pd.DataFrame:
    _load, Gate, gate_cfg, guard_cfg, resolver, root, run_gate_on_video = gate_parts
    if mode == "process":
        np.random.seed(0); random.seed(0)
    rows = []
    for jp in sorted((DETECTIONS / "LASIESTA").glob("*.json")):
        key = jp.stem
        vid = key.split("__", 1)[1]
        det = json.load(open(jp, encoding="utf-8"))
        if not det["frames"]:
            continue
        det = rekey(det, vid)
        start = det["frames"][0]["frame"]
        if mode == "video":
            np.random.seed(0); random.seed(0)
        sc, *_ = run_gate_on_video("LASIESTA", key, det, Gate, gate_cfg, guard_cfg, resolver, root)
        rows += [(vid, start + i, float(s)) for i, s in enumerate(sc)]
    df = pd.DataFrame(rows, columns=["video", "frame", "score"])
    df.to_parquet(DERIVED / f"scores_lasiesta_seed0_{mode}{'_fresh' if len(sys.argv) > 1 else ''}.parquet", index=False)
    return df.sort_values(["video", "frame"]).reset_index(drop=True)


def main() -> None:
    sys.path.insert(0, str(DEEP3_REPO))
    cwd = os.getcwd(); os.chdir(DEEP3_REPO)
    try:
        from scripts.generate_gate_scores import (_load_cfg, _load_asmag_gate_cfg, _import_gate,
                                                  _get_frame_resolver, run_gate_on_video)
        cfg = _load_cfg(str(DEEP3_REPO / "configs" / "default.yaml"))
        gate_cfg, guard_cfg = _load_asmag_gate_cfg(cfg)
        Gate = _import_gate(cfg)
        resolver, root = _get_frame_resolver("LASIESTA", cfg)
    finally:
        os.chdir(cwd)
    parts = (None, Gate, gate_cfg, guard_cfg, resolver, root, run_gate_on_video)
    ref = pd.read_parquet(GATE_SCORES_LASIESTA_FIXED)
    ref = ref[ref.dataset == "LASIESTA"][["video", "frame", "score"]].sort_values(["video", "frame"]).reset_index(drop=True)
    out = {}
    runs = {}
    modes = tuple(sys.argv[1:]) or ("process", "video", "video_repeat")   # e.g. "video" alone = fresh process
    for mode in modes:
        a = regen("video" if mode == "video_repeat" else mode, parts)
        runs[mode] = a
        ok = len(a) == len(ref) and bool((a[["video", "frame"]].values == ref[["video", "frame"]].values).all())
        d = np.abs(a.score.to_numpy(float) - ref.score.to_numpy(float)) if ok else None
        out[mode] = {"keys_identical": ok,
                     "vs_run_corr_max_abs_diff": float(d.max()) if ok else None,
                     "vs_run_corr_n_diff_gt_1e-6": int((d > 1e-6).sum()) if ok else None,
                     "vs_run_corr_bitwise_identical": bool(ok and (d == 0).all())}
        print(mode, out[mode], flush=True)
    if "video" in runs and "video_repeat" in runs:
        rr = np.abs(runs["video"].score.to_numpy(float) - runs["video_repeat"].score.to_numpy(float))
        out["seeded_run_to_run_max_abs_diff"] = float(rr.max())
    out["modes_in_this_process"] = list(modes)
    tag = "" if not sys.argv[1:] else "_fresh_" + "_".join(modes)
    json.dump(out, open(RESULTS / f"p3_seed0_check{tag}.json", "w"), indent=1)
    print(out)


if __name__ == "__main__":
    main()
