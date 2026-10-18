# Reproduction order (run from the project root; Python 3.13, versions in requirements.txt)

Every script exits with a non-zero status if a primary check fails.

1. Thesis pipeline, which produces `data/train_enriched.parquet` from the Kaggle files (see README):
   `python3 src/01_load_data.py && python3 src/02_preprocessing.py && python3 src/03_iseql_features.py`
   (`models/column_meta.json`, the feature lists, is shipped.)
2. Revision-4 inputs (frozen training-fitted device map, corrected P2, elapsed-bucket P4, per-split files):
   `mkdir -p data4 out4 out5 && python3 analysis/prepare_v4.py`
3. Checks:
   `python3 analysis/conformance_v4.py` (reference comparison; asserts 0 mismatches and 0 exclusions)
   `python3 analysis/prefix_v4.py` (7 recorded cuts from raw-valued fields; asserts primary statistics unchanged)
   `python3 analysis/coverage_tests.py && python3 analysis/spec_fixtures_v4.py` (synthetic; need no data)
4. Models (each fit about 1 min; writes `out4/pred_*.npz`, `models/v4_*.pkl` and `models/cols_v4_*.json`):
   `python3 analysis/train_v4.py C D Dbatch D_minus_P1 D_minus_P2 D_minus_P3 D_minus_P4 D_minus_P5 D_minus_P6`
   `python3 analysis/make_drop_files_v4.py`
   `python3 analysis/train_v4.py C@drop1 D@drop1 C@drop2 D@drop2 C@drop3 D@drop3 C@drop4 D@drop4`
5. Statistics, rules, diagnostics and SHAP:
   `python3 analysis/stats_v4.py` (about 40 min; 1,000 group-bootstrap replicates per comparison)
   `python3 analysis/rules_v4.py && python3 analysis/diagnostics_v4.py`
   `python3 analysis/shap_v4.py` (asserts the feature list equals the stored model's names and order)
6. Figures and traces:
   `FIGDIR=figures python3 analysis/make_fig_v4.py`
   `python3 analysis/make_traces_v4.py && python3 analysis/replay_trace.py out4/traces/*.json`
7. Grouping sensitivity (Supplement S8; protocol `analysis/results/PROTOCOL_uid.md`, written after the original `card1` diagnostics and before any composite-key result):
   `python3 analysis/stats_by_key.py card1 out5/stats_card1_selfcheck.parquet` (must equal revision 4)
   `python3 analysis/stats_by_key.py uid out5/stats_uid.parquet && python3 analysis/conformance_uid.py`
   `python3 analysis/train_uid.py Duid Duid@drop1 Duid@drop2 Duid@drop3 Duid@drop4 && python3 analysis/eval_uid.py`
   `python3 analysis/grouping_sensitivity.py`

8. Interval-join evaluation (Supplement S9; needs only `numpy`, `pandas`, `pyarrow`;
   writes `engine/out/`). Steps 8a and 8g need no data; the others need `data/train_enriched.parquet` from step 1:
   `python3 engine/test_ops.py` (8a: operator properties and the paper's worked cases)
   `python3 engine/ref_pandas.py` (8b: rebuilds the pandas reference vectors; asserts they match the stored columns)
   `python3 engine/run_eval.py` (8c: sweep and materializing evaluators vs the manual vectors on all 590,540 rows)
   `python3 engine/run_alerts.py` (8d: witness alerts and the composed burst constraint)
   `python3 engine/run_p1_readings.py` (8e: P1 under the rank and wall-clock readings)
   `python3 engine/run_p3.py` (8f: P3's conceptual escalation-span predicate, both readings)
   `python3 engine/test_inexpressibility.py` (8g: the pairwise-profile proposition, exact rational arithmetic)
   Override paths with `ISEQL_DATA` and `ISEQL_OUT` if your layout differs.
   Recorded outputs are `engine/results_*.txt`; see `engine/RESULTS_engine.md`.

Expected: LightGBM training is deterministic with these versions; best iterations are listed in Supplement S6.
The clean-directory run log is `analysis/results/clean_run.log`.
