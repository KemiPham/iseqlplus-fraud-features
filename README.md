# Traceable payment-fraud features with ISEQL+ interval specifications

Supplementary material and code for:

> T. T. Pham and F. Persia, "Toward Traceable Payment Fraud Features with ISEQL+ Interval Specifications," submitted to the IEEE International Conference on Semantic Computing (ICSC 2027).

- **[supplement.pdf](supplement.pdf)**: parameter and provenance ledger, conformance and prefix-invariance checks, set-level coverage proofs and tests, specification fixtures, replayable traces, configuration, and additional results.
- **`src/`**: feature pipeline (steps 01–06, `common.py`). It builds `data/train_enriched.parquet` from the IEEE-CIS files.
- **`analysis/`**: generators for every reported number (training-fitted, frozen device preprocessing), including `diagnostics_v4.py` (Section V diagnostics) and the grouping sensitivity (`stats_by_key.py`, `train_uid.py`, `eval_uid.py`, `grouping_sensitivity.py`; protocol in `analysis/results/PROTOCOL_uid.md`). See **[RUN.md](RUN.md)** for the command order; every check exits non-zero on failure.
- **`analysis/results/`**: result tables, logs, the device map, the drop manifest, and ID-aligned model scores (`predictions_C_D_Dbatch.csv.gz`) and SHAP summaries (`shap_strat_v4_test.csv`); labels are removed, join them by `TransactionID` from the official data. `clean_run.log` records a clean-directory run of RUN.md steps 2–7; `MANIFEST.sha256` lists file hashes.
- **`traces/`**: three replayable P1 trace fixtures. They need no data:
  `python3 traces/replay_trace.py traces/*.json`
- **Synthetic checks** that need no data:
  `python3 analysis/coverage_tests.py` and `python3 analysis/spec_fixtures_v4.py`

## Data
The IEEE-CIS Fraud Detection data are **not** redistributed here. Download `train_transaction.csv` and `train_identity.csv` from <https://www.kaggle.com/c/ieee-fraud-detection/data> (after accepting the competition rules) into `data/`. The trace fixtures contain only the handful of transactions needed to replay three examples.

## Environment
Python 3.13 and the package versions in `requirements.txt`. With these versions, LightGBM training is deterministic.

## Scope
The statistics are manually derived from the ISEQL+ specifications; no query engine is included. The checks are finite and conditional on the conventions stated in the supplement. See the paper and the supplement for the limitations.
