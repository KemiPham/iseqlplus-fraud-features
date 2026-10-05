# Reproduction order (run from the project root; Python 3.13, see requirements.txt)

1. Thesis pipeline, which produces `data/train_enriched.parquet` and `models/column_meta.json`:
   `python3 src/01_load_data.py && python3 src/02_preprocessing.py && python3 src/03_iseql_features.py && python3 src/04_train_models.py`
2. Revision-4 inputs (frozen training-fitted device map, corrected P2, elapsed-bucket P4, per-split files):
   `mkdir -p data4 out4 && python3 analysis/prepare_v4.py`
3. Checks:
   `python3 analysis/conformance_v4.py` (reference comparison)
   `python3 analysis/prefix_v4.py` (7 recorded cuts from raw fields)
   `python3 analysis/coverage_tests.py` and `python3 analysis/spec_fixtures_v4.py` (synthetic, need no data)
4. Models (each fit is about 1 min; writes `out4/pred_*.npz`):
   `python3 analysis/train_v4.py C D Dbatch D_minus_P1 D_minus_P2 D_minus_P3 D_minus_P4 D_minus_P5 D_minus_P6`
   `python3 analysis/make_drop_files_v4.py`
   `python3 analysis/train_v4.py C@drop1 D@drop1 C@drop2 D@drop2 C@drop3 D@drop3 C@drop4 D@drop4`
5. Statistics, rules and SHAP:
   `python3 analysis/stats_v4.py` (about 40 min, 1,000 group-bootstrap replicates per comparison)
   `python3 analysis/rules_v4.py`
   `python3 analysis/shap_v4.py`
6. Figures and traces:
   `FIGDIR=figures python3 analysis/make_fig_v4.py`
   `python3 analysis/make_traces_v4.py && python3 analysis/replay_trace.py out4/traces/*.json`

Expected: LightGBM training is deterministic with these versions, and best iterations are listed in Supplement S6.
