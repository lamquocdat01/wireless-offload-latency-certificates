"""P2 / Viec 3: regenerate the ASMAG gate scores for LASIESTA in TRUE temporal order (topic #19).

Uses the gate pass of the published Deep3 repository UNCHANGED (read-only import of
scripts/generate_gate_scores.py from DEEP3_REPO; the gate class is imported from the ASMAG source
named in that repo's configs/default.yaml). No code of topic 17 is imported.

Detections: THS_DATASETS/_derived/detections_cpu_yolo26s/LASIESTA (dump indexed by lexicographic
file position) re-keyed to the true temporal index (file number - 1), exactly as policy.py does.

Output: THS_DATASETS/_derived/19/scores_lasiesta_regen/frame_scores_lasiesta.parquet
        results/p2_asmag_regen_check.json   (comparison with topic-17 run_corr, read-only)
Requirement (CP0 #4): 100 % agreement of keys, g_t, score and the 393 events.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

from paths import DEEP3_REPO, DERIVED, DETECTIONS, GATE_SCORES_LASIESTA_FIXED, RESULTS
from policy import _lasiesta_numbers

OUT = DERIVED / "scores_lasiesta_regen"
OUT.mkdir(parents=True, exist_ok=True)


def rekey(det: dict, vid: str) -> dict:
    nums = _lasiesta_numbers(vid)
    fr = det["frames"]
    assert [f["frame"] for f in fr] == list(range(len(fr))), vid
    assert sorted(nums) == list(range(1, len(nums) + 1)), vid
    new = sorted(({"frame": nums[f["frame"]] - 1, "boxes": f.get("boxes", [])} for f in fr),
                 key=lambda f: f["frame"])
    return {**det, "frames": new}


def main() -> None:
    sys.path.insert(0, str(DEEP3_REPO))
    cwd = os.getcwd()
    os.chdir(DEEP3_REPO)                       # Deep3 resolves configs relative to its root
    try:
        from scripts.generate_gate_scores import (_load_cfg, _load_asmag_gate_cfg, _import_gate,
                                                  _get_frame_resolver, run_gate_on_video)
        cfg = _load_cfg(str(DEEP3_REPO / "configs" / "default.yaml"))
        gate_cfg, guard_cfg = _load_asmag_gate_cfg(cfg)
        Gate = _import_gate(cfg)
        resolver, root = _get_frame_resolver("LASIESTA", cfg)
    finally:
        os.chdir(cwd)
    rows, t0 = [], time.time()
    for jp in sorted((DETECTIONS / "LASIESTA").glob("*.json")):
        key = jp.stem
        vid = key.split("__", 1)[1]
        det = json.load(open(jp, encoding="utf-8"))
        if not det["frames"]:
            continue
        det = rekey(det, vid)
        fl = det["frames"]
        start = fl[0]["frame"]
        has = {f["frame"]: len(f.get("boxes", [])) > 0 for f in fl}
        sc, *_ = run_gate_on_video("LASIESTA", key, det, Gate, gate_cfg, guard_cfg, resolver, root)
        for pos in range(len(sc)):
            rows.append(("LASIESTA", vid, start + pos, float(sc[pos]), int(has.get(start + pos, False))))
        print(f"{vid}: {len(sc)} frames ({time.time() - t0:.0f}s)", flush=True)
    new = pd.DataFrame(rows, columns=["dataset", "video", "frame", "score", "g_t"])
    new["frame"] = new.frame.astype("int32"); new["score"] = new.score.astype("float32")
    new["g_t"] = new.g_t.astype("int8")
    new.to_parquet(OUT / "frame_scores_lasiesta.parquet", index=False)

    ref = pd.read_parquet(GATE_SCORES_LASIESTA_FIXED)
    ref = ref[ref.dataset == "LASIESTA"][["video", "frame", "score", "g_t"]]
    a = new.sort_values(["video", "frame"]).reset_index(drop=True)
    b = ref.sort_values(["video", "frame"]).reset_index(drop=True)
    res = {"rows_regen": len(a), "rows_run_corr": len(b)}
    keys_ok = len(a) == len(b) and bool((a[["video", "frame"]].values == b[["video", "frame"]].values).all())
    res["keys_identical"] = keys_ok
    if keys_ok:
        d = np.abs(a.score.to_numpy(float) - b.score.to_numpy(float))
        res.update({"g_t_identical": bool((a.g_t.values == b.g_t.values).all()),
                    "score_max_abs_diff": float(d.max()), "score_n_diff_gt_0": int((d > 0).sum()),
                    "score_n_diff_gt_1e-6": int((d > 1e-6).sum()), "score_bitwise_identical": bool((d == 0).all())})
    ev = lambda df: int(sum(int(np.sum(np.diff(np.r_[0, g.g_t.to_numpy()]) == 1))
                            for _, g in df.groupby("video")))
    res["events_regen"], res["events_run_corr"] = ev(a), ev(b)
    res["match_100pct"] = bool(keys_ok and res.get("g_t_identical") and res.get("score_n_diff_gt_1e-6", 1) == 0
                               and res["events_regen"] == res["events_run_corr"] == 393)
    json.dump(res, open(RESULTS / "p2_asmag_regen_check.json", "w"), indent=1)
    print(res)


if __name__ == "__main__":
    main()
