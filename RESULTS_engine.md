# Interval-join execution of the ISEQL+ specifications — results

All numbers below are reproduced by the scripts in this folder, run against
`data/train_enriched.parquet` (590,540 rows). Each script writes its own
`results_*.txt` log beside it. Test month = elapsed day >= 150 (92,427 rows,
fraud base rate 3.4763%).

Run from the repository root, in the order given in `RUN.md` step 8:
`engine/test_ops.py`, `engine/ref_pandas.py` (writes `engine/out/ref.parquet`),
`engine/run_eval.py`, `engine/run_alerts.py`, `engine/run_p1_readings.py`,
`engine/run_p3.py`, `engine/test_inexpressibility.py`. Paths can be overridden
with the `ISEQL_DATA` and `ISEQL_OUT` environment variables.

## 1. The operators are executable and agree with the paper's examples

`iseql_join.py` implements `DJ` with a cardinality constraint and `RDJ` with the
set-level coverage constraint `C(r)` of eq. (3), group equality folded into the
join predicate. Two independent implementations:

- `reference_eval*` materialises every window and partner interval and computes
  `C(r)` by clipping to `r`, merging overlaps and counting chronons;
- `sweep_*` sorts once by (group, start point) and sweeps each group with
  monotone endpoints, one pass per operator.

Ten operator tests pass (`results_operator_tests.txt`), including the paper's own
worked examples: `r=[1,10]`, `s1=[1,8]`, `s2=[3,10]` gives normalised pairwise
sum 1.6 against `C(r) = 1.0`; `s1=[2,9]`, `s2=[4,5]` gives capped sum 1 against
`C = 0.8`. Boundedness, duplicate-invariance, monotonicity under added partners,
and equality with the normalised sum under disjointness all hold.

## 2. Conformance at full scale, and timing

`results_conformance.txt`. The sweep agrees with the manually derived (pandas)
values on **all 590,540 rows**, for every exactly-expressible statistic:

| statistic | specification | disagreement |
|---|---|---|
| `micro_tx_ratio_3min` | `w_180 ^{C(>=0.6)} RDJ sigma_amt<10(Tx)` | max abs diff 0 |
| `subthreshold_count_elapsed` | `sigma_amt<9500(Tx) DJ_{>=3} [d(t), t-1]` | 0 rows |
| `tx_count_1h` | `Tx DJ_{>=5} w_3600` | 0 rows |
| `tx_count_6h` | `Tx DJ_{>=5} w_21600` | 0 rows |
| `tx_count_24h` | `Tx DJ_{>=5} w_86400` | 0 rows |

The literal implementation then agrees with both the sweep and the manual values
on 178,237 rows spanning all 62 tied-timestamp `card1` groups plus 300 random
groups — the same scope as the paper's existing reference check.

Evaluation time for all five specifications over 590,540 anchors: **1.77 s**
including the sort (P1 0.59 s, each P5 window 0.28-0.29 s, P4 0.27 s, sort
0.06 s), against 4.4 s for the pandas derivation. `day = (dt - 86400) // 86400`,
so the P4 bucket starts at `86400*(day+1)`.

Alert level reproduces the published table exactly (`results_alerts.txt`):

| rule | alerts | TP | precision | lift |
|---|---|---|---|---|
| P1 `C >= 0.6` | 216 | 22 | 10.185% | 2.930 |
| P4 `card >= 3` | 41,590 | 1,501 | 3.609% | 1.038 |
| P5 `card >= 5` (1h) | 6,275 | 152 | 2.422% | 0.697 |
| P5 `card >= 5` (6h) | 26,064 | 907 | 3.480% | 1.001 |
| P5 `card >= 5` (24h) | 47,506 | 1,707 | 3.593% | 1.034 |

## 3. The burst composition fires and catches nothing

The paper proposes `w_180 ^{(>=3), C(>=0.6)} RDJ sigma_amt<10(Tx)` as the
specification a genuine burst would need. Evaluated on the test month:

| specification | alerts | TP |
|---|---|---|
| `C >= 0.6` (P1 as published) | 216 | 22 |
| `C >= 0.6` and micro-cardinality `>= 2` | 64 | **0** |
| `C >= 0.6` and micro-cardinality `>= 3` | 9 | **0** |
| `C >= 0.6` and micro-cardinality `>= 5` | 0 | — |

152 of the 216 alerts have a history of exactly one event, and all 22 true
positives lie there; the 64 alerts with two or more prior events contain no
fraud. Adding the cardinality constraint removes every true positive. So the
dominance reading is not merely weaker than the burst reading — on this data it
is the only one that fires at all.

## 4. P1 is satisfiable only under the declared rank reading

`results_p1_readings.txt`. Under the declared bijection from the history to
ranks `1..n`, P1 yields 883 alerts overall and 216 in the test month. Under a
literal wall-clock reading, where `r = [t-180, t-1]` and each transaction is the
point interval `[t_j, t_j]`:

- point-interval partners give `C(r) <= n/|r|`, so `C >= 0.6` over 180 chronons
  would need at least 108 covered seconds;
- the largest 180 s history anywhere in the data is `n = 21`, capping attainable
  coverage at 21/180 = 0.1167;
- the maximum actually observed is **0.0222**, and explicit evaluation of all
  51,996 rows with a non-empty history gives **0 alerts** at `p = 0.6` — and
  still 0 at `p = 0.05`.

The rank bijection is therefore load-bearing rather than a notational
convenience: without it the constraint is unsatisfiable on point-interval data.

## 5. P3's faithful specification, evaluated for the first time

`results_p3_specification.txt`. P3 is the row labelled a lossy summary and it has
no rule alert in the published table. The engine evaluates
`Esc ^{(>=0.8)}(LOJ union DJ) w_3600` directly. Candidate spans satisfy
`inside <= 3600` and `inside/L >= 0.8`, hence `L <= 4500`, so a 4500 s lookback
is exact rather than a heuristic cut.

| | alerts | TP | precision | lift |
|---|---|---|---|---|
| faithful span spec, history-only | 2,445 | 68 | 2.781% | **0.800** |
| faithful span spec, anchor-closing | 3,676 | 120 | 3.264% | 0.939 |
| `escalation_ratio_1h >= 0.8` (the statistic) | 11,209 | 373 | 3.328% | 0.957 |
| `escalation_ratio_1h >= 2.2685` (count-matched) | 2,446 | 55 | 2.249% | 0.647 |

Both faithful readings have lift below 1, i.e. they select transactions *less*
likely to be fraud than the base rate. The specification does not fix whether
the anchor may close the span; the two readings differ on 1,231 test-month
anchors, which is a specification ambiguity the statistic hides.

Query-to-feature disagreement (history-only reading): at a count-matched
threshold the statistic and the specification share only 1,587 of their alerts,
Jaccard **0.480**, with 858 spec-only and 859 statistic-only. 15,652 anchors have
a non-zero statistic with no qualifying span; 63 have a qualifying span with a
zero statistic. The summary is not a conservative approximation of the
specification in either direction.

## 6. Set-level coverage is inexpressible by pairwise overlap ratios

`results_inexpressibility.txt`. The paper's 1.6-against-1.0 counterexample shows
the normalised pairwise sum is not `C`; it does not show `C` is inexpressible.

**Proposition.** `C(r)` is not a function of the anchor, the number of partners,
and the multisets of pairwise left ratios `#(r^s)/|r|` and right ratios
`#(r^s)/|s|`. Hence no constraint built from pairwise overlap ratios on the same
operator — under any boolean combination or aggregate (sum, capped sum, max,
min, mean, order statistics) — expresses `C(r) theta p`.

**Witnesses** (verified in exact rational arithmetic; all partners distinct and
of equal length):

- `r = [1,10]`, A = `{[1,5],[6,10]}` and B = `{[1,5],[2,6]}`: both have left
  ratios `{1/2, 1/2}` and right ratios `{1, 1}`, yet `C(A) = 1.0` and
  `C(B) = 0.6`. Normalised pairwise sum is 1.0 for both, capped sum 1.0 for both.
- `r = [1,12]`, A = `{[1,4],[5,8],[9,12]}` and B = `{[1,4],[2,5],[3,6]}`: left
  ratios `{1/3,1/3,1/3}`, right ratios `{1,1,1}`, `C(A) = 1.0`, `C(B) = 0.5`.

**Corollary (the gap is unbounded).** With `|r| = k^2`, A tiling `r` in `k`
blocks of length `k` and B the `k` unit-shifted intervals `[i, i+k-1]`, the
profiles are identical while `C(A) = 1` and `C(B) = (2k-1)/k^2`, so the gap is
`1 - (2k-1)/k^2 -> 1`. Verified for k = 2..8: gaps 0.250, 0.444, 0.563, 0.640,
0.694, 0.735, 0.766.

## 7. What this says about the absence of a predictive gain

The pre-registered gain experiments found no gain over the 436-feature baseline
and none over a raw-field baseline (`gain_experiments.zip`: RS vs R test
dAUC +0.0003, p = 0.70; the one positive variant, RS_uid at +0.0074, uses D1 in
its grouping key, and the D1 control shows `R + D1` alone gives +0.0233 with the
statistics adding only +0.0027, CI [-0.0001, 0.0057]).

Sections 3 to 5 supply a mechanism rather than a restatement. Where the
specifications are written faithfully, they either stop firing (the burst
composition, 0 true positives), are unsatisfiable (P1 under wall-clock
coverage, 0 alerts), or are anti-predictive (P3 faithful, lift 0.800). The
absence of gain is therefore consistent with the specified behaviours not being
what the chargeback labels mark, which is a statement about the dataset rather
than about the language.
