# wireless-offload-latency-certificates

Code and derived data for the manuscript *Admitting Content-Gated Camera Traffic onto 5G Uplinks:
Frame-Level Detection-Latency Certificates with Shaping and Erasure Coding* (Dat Lam Quoc).

Every number printed in the paper and its supplement is a key of `results/numbers.json`, written by
`code/p3_analysis.py`. The LaTeX sources print only macros generated from that file.

## Contents

| Path | What it is |
|---|---|
| `code/` | certificates (`nc_bounds.py`, `fec.py`, `service_curve.py`), policies (`policy.py`), trace replay (`run_full.py`), analysis and figures (`p3_analysis.py`), machine checks of the theorems (`verify_*.py`), trace parsing (`parse_det6g.py`, `p2_grid_owd.py`) |
| `tests/` | unit tests (`python -m pytest -q tests`) |
| `data/derived/activations_K2_a0.05.parquet` | the activation corpus: 121 videos (BMC, CDnet2014, LASIESTA), per-frame object indicator `g_t` and motion-gate `score` |
| `data/derived/p3_main_getime_pervideo.npz` | per-video detection / violation counts of the headline run (video bootstrap) |
| `results/` | derived results: grid cells of the three runs, cross-session tables, Gilbert–Elliott fits, theorem checks, `numbers.json`, figures |
| `MANIFEST.sha256` | SHA-256 of every file in this repository |

**Not included (cited, not re-hosted):** the DETERMINISTIC6G COTS-5G one-way-delay traces
(Zenodo, DOI [10.5281/zenodo.10390211](https://doi.org/10.5281/zenodo.10390211), CC BY 4.0), the uplink
throughput traces of Narayanan et al., SIGCOMM 2021 (DOI
[10.1145/3452296.3472923](https://doi.org/10.1145/3452296.3472923)), and the raw videos. Per-packet arrays
derived from DET6G are regenerated locally by level 2 below.

**Gate scores.** BMC and CDnet2014 scores are the published scores of the motion gate (DOI
[10.1109/JIOT.2026.3740931](https://doi.org/10.1109/JIOT.2026.3740931)); LASIESTA scores come from a
fixed run of the same gate on time-ordered detections. From the supplement, Sec. S8: "The published gate keeps internal state across passes over the corpus. Running it twice on the same time-ordered detections in one process, with the numpy and random generators seeded to 0 before each video, the second pass reproduces all 18 425 released scores bit for bit (0 differences; checked in 2 independent processes). A single pass in a fresh process reproduces the frame keys, the object indicators and all 393 events exactly, but 13 041 scores differ (maximum 0.15, Spearman 0.990); a third pass drifts again (13 052 differences). All experiments read the released file." Evidence: `results/p3_seed0_check*.json`.

## Reproduce

Python 3.11; `python -m pip install -r requirements.txt`. All runs use seed 42.

### Level 1 — numbers and figures from the released derived results (≈ 5 min)

```bash
python -m pytest -q tests
cp results/numbers.json /tmp/numbers_released.json
python code/p3_analysis.py
python code/compare_numbers.py results/numbers.json /tmp/numbers_released.json   # must print 0 differences
git status --short results/                                                     # numbers.json unchanged
```

Without the DET6G arrays, `p3_analysis.py` skips Fig. 2 (the OWD CCDF); every number is still produced.

### Level 2 — full re-run from the public traces

1. Download `COTS5G measurements.zip` from Zenodo record 10390211 (541 015 900 B, SHA-256
   `4189189ac65dca2d5c396002bb2952137c0d192a62e0d1fc7c2d3a54a2607ec0`) and unzip it into
   `data/DET6G_COTS5G/` (or set `DET6G_DIR`).
2. Run, in order:

```bash
python code/parse_det6g.py          # per-packet OWD arrays + results/p0_owd_stats.json
python code/p2_grid_owd.py          # send-grid arrays + results/p2_grid_owd.json
python code/service_curve.py        # radio certificates -> results/p1_service_curve.json
python code/p2_fec_sensitivity.py   # Gilbert-Elliott fits -> results/p2_fec_sensitivity.json
python code/run_full.py --rmodel indep      # independence column  -> results/p2_main_*
python code/run_full.py --rmodel ge_time    # headline             -> results/p3_main_getime_*
python code/run_full.py --rmodel ge_trace   # GE trace-scale column -> results/p3_main_getrace_*
python code/p3_analysis.py
python code/compare_numbers.py results/numbers.json /tmp/numbers_released.json
```

`run_full.py` uses two worker processes and stops itself if free RAM drops below 6 GB.

Measured from a clean clone of this tag on an i7-1185G7 laptop (32 GB), with the CPU shared with other
jobs (so these are upper figures; the original headline grid took 647 s on a less loaded machine):

| Step | Time |
|---|---|
| Level 1 (tests + `p3_analysis.py` + comparison) | 26 s |
| `parse_det6g.py` / `p2_grid_owd.py` / `service_curve.py` / `p2_fec_sensitivity.py` | 12 s / 2 s / 32 s / 42 s |
| `run_full.py` indep / ge_time / ge_trace | 1 445 s / 953 s / 861 s |
| `p3_analysis.py` | 38 s |
| **Level 2 total** (after the download) | **3 385 s (56 min)** |

Both levels reproduced `results/numbers.json` key by key (1 107 of 1 107 keys), and level 2 regenerated
every other file in `results/` byte for byte.

Inputs that level 2 does not regenerate (they need data that cannot be redistributed or were produced
upstream): `p1_c_estimate.json` (frame sizes × SIGCOMM'21 throughput), `p1_burst_tax.json`,
`p1_quality_grid.csv`, `p0_sanity_b_cells.csv`, and the activation corpus.

## Data provenance & licenses

| Material in this repository | Source | License |
|---|---|---|
| Code (`code/`, `tests/`) | this work | MIT (`LICENSE`) |
| Rows with `dataset == "LASIESTA"` in `data/derived/activations_K2_a0.05.parquet` (gate scores and object indicators derived from LASIESTA) | LASIESTA database, GTI, Universidad Politécnica de Madrid, http://www.gti.ssr.upm.es/data/lasiesta | **CC BY-SA 4.0** (`LICENSE-DATA`) |
| Rows with `dataset` = `CDnet2014` or `BMC` in the same file | CDnet2014 (changedetection.net); BMC (Background Models Challenge) | cite-only, no license published; cite the original datasets |
| Results derived from the DETERMINISTIC6G traces (`results/`, per-session statistics, grids, certificates) | Zenodo, DOI 10.5281/zenodo.10390211 | CC BY 4.0, inherited (`LICENSE-DATA`) |
| Throughput quantiles in `results/p1_c_estimate.json` | Narayanan et al., SIGCOMM 2021, DOI 10.1145/3452296.3472923 | summary statistics only; the traces are not re-hosted |

No video frame, image or raw trace of any dataset is re-hosted here.
