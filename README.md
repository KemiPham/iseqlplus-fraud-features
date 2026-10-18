# Traceable payment-fraud features with ISEQL+ interval specifications

Companion material for T. T. Pham and F. Persia, *Toward Traceable Payment Fraud
Features with ISEQL+ Interval Specifications*, submitted to IEEE ICSC 2027.

## Contents

- `supplement.pdf` — online supplementary material (S1–S9): parameter and
  provenance ledger, conformance and prefix-invariance checks, the set-level
  coverage constraint with its proofs and the pairwise-profile proposition,
  specification fixtures, replayable traces, configuration, additional results,
  grouping sensitivity, and the interval-join evaluation.
- `src/` — feature pipeline (01–06) building the enriched training file from the
  IEEE-CIS data.
- `analysis/` — scripts generating the reported numbers for the manual-feature
  checks, models, rules, diagnostics and grouping sensitivity.
- `engine/` — the scoped interval-join evaluator and its checks (Supplement S9).
- `analysis/results/` — result tables, logs, device map, drop manifest, model
  scores and SHAP summaries keyed by `TransactionID` (labels removed), hashes.
- `traces/` — three replayable trace fixtures; need no data.
- `RUN.md` — execution order. Every check exits non-zero on failure.

## Data

The IEEE-CIS Fraud Detection data are not redistributed. Download
`train_transaction.csv` and `train_identity.csv` from
<https://www.kaggle.com/c/ieee-fraud-detection/data> into `data/`.

## Quick checks that need no data

```
python3 traces/replay_trace.py traces/*.json
python3 analysis/coverage_tests.py
python3 analysis/spec_fixtures_v4.py
python3 engine/test_ops.py
python3 engine/test_inexpressibility.py
```

## Scope

The eight statistics reported in the paper were derived manually from the ISEQL+
specifications. The `engine/` directory additionally provides a **scoped
evaluator** that executes the exactly expressible specifications (P1, P4, P5)
as interval joins and reproduces those manually derived vectors on all 590,540
rows; it also evaluates the composed burst constraint, the wall-clock reading of
P1, and P3's conceptual escalation-span predicate. This is **not** a
general-purpose or production ISEQL+ query engine, and it does not execute P2 or
P6, whose required session and authentication events the dataset does not
record. Agreement is between implementations under the conventions declared in
the supplement; it does not validate those conventions, nor the interpretation
of the generic `isFraud` label as evidence of any specific fraud typology.
All checks are finite.

## Environment

Python 3.13 with the versions in `requirements.txt` for `src/` and `analysis/`.
The `engine/` scripts need only `numpy`, `pandas` and `pyarrow`. LightGBM
training is deterministic under these versions.
