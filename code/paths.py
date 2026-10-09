"""Path resolution for the artifact (repo-relative; override with environment variables).

WOLC_DERIVED  derived inputs (default: <repo>/data/derived) -- gate scores of the corpus, per-video counts,
              and, after running parse_det6g.py / p2_grid_owd.py, the per-packet OWD arrays
DET6G_DIR     the unzipped DETERMINISTIC6G COTS5G measurements (Zenodo 10.5281/zenodo.10390211),
              only needed for the full re-run (default: <repo>/data/DET6G_COTS5G)
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
DERIVED = Path(os.environ.get("WOLC_DERIVED", REPO / "data" / "derived"))
DET6G = Path(os.environ.get("DET6G_DIR", REPO / "data" / "DET6G_COTS5G"))

# Not used by the released pipeline (scores ship as data/derived/activations_K2_a0.05.parquet); kept so that
# policy.py imports unchanged.
DETECTIONS = DERIVED / "detections"
LASIESTA_RAW = DERIVED / "LASIESTA"
DEEP3_REPO = DERIVED / "deep3"
GATE_SCORES = DERIVED / "frame_scores.parquet"
GATE_SCORES_LASIESTA_FIXED = DERIVED / "frame_scores_lasiesta.parquet"

for _p in (RESULTS, DERIVED):
    _p.mkdir(parents=True, exist_ok=True)
