"""
STEP 3 -- Compute the 6 ISEQL+ patterns / 8 output columns (core thesis
contribution). All computed with vectorized pandas groupby + time-based
rolling windows (pandas' Cython-implemented rolling, NOT row-by-row Python
loops), on the FULL chronologically-sorted dataset before splitting, so
val/test rows retain correct historical context from earlier data.

Windows use closed='left' semantics throughout: the window is
[t - W, t) -- i.e. strictly PRECEDING transactions, excluding the current
row itself, which matches the "preceding N seconds" wording in the brief.

IMPORTANT implementation detail: pandas' `groupby(...).rolling(window,
on=<datetime col>)` returns results ordered by GROUP (not by original row
order) -- assigning `.values` straight back to the original dataframe would
silently misalign every row. We fix this with an explicit scatter-back: sort
once by the group key with a stable sort to get the exact order pandas'
grouped rolling produces, capture that permutation, and scatter every
rolling result back to original row positions through it.

Output: data/train_enriched.parquet (fast cache) and data/train_enriched.csv
(deliverable), plus outputs/feature_verification.csv.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

t0 = time.time()
print(f"Loading {C.MERGED_PARQUET} ...")
df = pd.read_parquet(C.MERGED_PARQUET)
df = df.sort_values(C.TIME_COL, kind="mergesort").reset_index(drop=True)
print(f"  shape: {df.shape}")

# pseudo-datetime needed for pandas time-based rolling (TransactionDT is
# seconds elapsed, not a real timestamp, but any monotonic epoch works)
df["_dt"] = pd.to_datetime(df[C.TIME_COL], unit="s")
df["_is_micro10"] = (df["TransactionAmt"] < 10).astype("float64")
df["_is_fail1"] = (df["TransactionAmt"] < 1).astype("float64")
df["_orig_idx"] = np.arange(len(df))

# Permutation pandas' grouped rolling produces: sort=True on the groupby
# guarantees deterministic (card1 ascending, then original chronological
# order within each card1) iteration order; a stable sort on 'card1' alone
# reproduces exactly that order since df is already time-sorted.
order_df = df[["card1", "_orig_idx"]].sort_values("card1", kind="mergesort")
ORIG_POS = order_df["_orig_idx"].to_numpy()

g = df.groupby("card1", sort=True)


def scatter_back(grouped_values, fill_value=0.0):
    """Map a rolling result (in grouped order) back to original row order."""
    arr = np.empty(len(df), dtype="float64")
    arr[ORIG_POS] = np.asarray(grouped_values, dtype="float64")
    arr[np.isnan(arr)] = fill_value
    return arr


def scatter_back_raw(grouped_values):
    """Same as scatter_back but keeps NaN (for later custom handling)."""
    arr = np.empty(len(df), dtype="float64")
    arr[ORIG_POS] = np.asarray(grouped_values, dtype="float64")
    return arr


# ---------------------------------------------------------------------------
# P1 -- micro_tx_ratio_3min (180s): ratio of same-card tx with amt<$10 to all
# same-card tx in the preceding 180s. 0 if no prior tx in window.
# ---------------------------------------------------------------------------
print("Computing P1 micro_tx_ratio_3min (180s rolling) ...")
r180 = g.rolling("180s", on="_dt", closed="left")["_is_micro10"].agg(["sum", "count"])
micro_sum = scatter_back_raw(r180["sum"].to_numpy())
micro_count = scatter_back_raw(r180["count"].to_numpy())
with np.errstate(invalid="ignore", divide="ignore"):
    ratio = micro_sum / micro_count
ratio = np.where((micro_count == 0) | ~np.isfinite(ratio), 0.0, ratio)
df["micro_tx_ratio_3min"] = ratio

# ---------------------------------------------------------------------------
# P2 -- device_novelty_flag: 1 if this card1's cleaned device_family has NOT
# appeared in the preceding 30 days (2,592,000s); first tx per card = 1.
# Implemented via groupby(card1, device_family).shift() on TransactionDT --
# this IS a transform, so it stays correctly aligned to the original index
# (no scatter-back needed).
# ---------------------------------------------------------------------------
print("Computing P2 device_novelty_flag (30-day pair recency) ...")
pair_time = df.groupby(["card1", "device_family"], dropna=False, sort=False,
                        observed=True)[C.TIME_COL]
prev_time = pair_time.shift(1)
gap = df[C.TIME_COL] - prev_time
df["device_novelty_flag"] = ((prev_time.isna()) | (gap > 2_592_000)).astype("int8")

# ---------------------------------------------------------------------------
# P3 -- escalation_ratio_1h (3600s): (max-min)/mean of TransactionAmt among
# preceding same-card tx in the last hour. 0 if <2 prior tx in window.
# Capped at 100.
# tx_count_1h (P5, 1h leg) is derived from the same rolling call to avoid a
# second pass over the data.
# ---------------------------------------------------------------------------
print("Computing P3 escalation_ratio_1h + tx_count_1h (3600s rolling) ...")
r3600 = g.rolling("3600s", on="_dt", closed="left")["TransactionAmt"].agg(
    ["max", "min", "mean", "count"]
)
amt_max = scatter_back_raw(r3600["max"].to_numpy())
amt_min = scatter_back_raw(r3600["min"].to_numpy())
amt_mean = scatter_back_raw(r3600["mean"].to_numpy())
amt_count_1h = scatter_back(r3600["count"].to_numpy(), fill_value=0.0)

with np.errstate(invalid="ignore", divide="ignore"):
    escalation = (amt_max - amt_min) / amt_mean
escalation = np.where((amt_count_1h < 2) | ~np.isfinite(escalation), 0.0, escalation)
escalation = np.clip(escalation, a_min=None, a_max=100.0)
df["escalation_ratio_1h"] = escalation
df["tx_count_1h"] = amt_count_1h.astype("int32")

# ---------------------------------------------------------------------------
# P4 -- structuring_count_day: count of same-card tx with amt<9500 on the
# same calendar day (the 'day' bucket derived from TransactionDT in step 1).
# This is a whole-day total (not a running "so far today" count). Computed
# via groupby(...).transform('sum'), which is index-aligned by construction.
# ---------------------------------------------------------------------------
print("Computing P4 structuring_count_day (whole-day total per card1) ...")
is_structuring = (df["TransactionAmt"] < 9500).astype("float64")
df["structuring_count_day"] = (
    is_structuring.groupby([df["card1"], df["day"]]).transform("sum").astype("int32")
)

# ---------------------------------------------------------------------------
# P5 -- tx_count_6h / tx_count_24h: rolling counts of same-card tx in the
# preceding 21600s / 86400s. (tx_count_1h already computed above.)
# ---------------------------------------------------------------------------
print("Computing P5 tx_count_6h (21600s rolling) ...")
r6h = g.rolling("21600s", on="_dt", closed="left")["_is_micro10"].count()
df["tx_count_6h"] = scatter_back(r6h.to_numpy(), fill_value=0.0).astype("int32")

print("Computing P5 tx_count_24h (86400s rolling) ...")
r24h = g.rolling("86400s", on="_dt", closed="left")["_is_micro10"].count()
df["tx_count_24h"] = scatter_back(r24h.to_numpy(), fill_value=0.0).astype("int32")

# ---------------------------------------------------------------------------
# P6 -- micro_fail_count_5min (300s): count of same-card tx with amt<$1 in
# the preceding 300s.
# ---------------------------------------------------------------------------
print("Computing P6 micro_fail_count_5min (300s rolling) ...")
r300 = g.rolling("300s", on="_dt", closed="left")["_is_fail1"].sum()
df["micro_fail_count_5min"] = scatter_back(r300.to_numpy(), fill_value=0.0)

df.drop(columns=["_dt", "_is_micro10", "_is_fail1", "_orig_idx"], inplace=True)

print(f"\nAll 8 ISEQL+ columns computed in {time.time()-t0:.1f}s.")

# ---------------------------------------------------------------------------
# Sanity check on the fix: for a handful of random card1 groups, recompute
# tx_count_1h with a plain, unambiguous (if slow) per-group loop and compare.
# ---------------------------------------------------------------------------
print("\nSanity-checking scatter-back alignment on 25 random card1 groups ...")
rng = np.random.default_rng(C.RANDOM_STATE)
sample_cards = rng.choice(df["card1"].unique(), size=min(25, df["card1"].nunique()), replace=False)
mismatches = 0
for card in sample_cards:
    sub = df[df["card1"] == card].sort_values(C.TIME_COL, kind="mergesort")
    times = sub[C.TIME_COL].to_numpy()
    expected = np.array([np.sum((times < t) & (times >= t - 3600)) for t in times])
    actual = sub["tx_count_1h"].to_numpy()
    if not np.array_equal(expected, actual):
        mismatches += 1
if mismatches == 0:
    print("  PASS: tx_count_1h matches an independent brute-force recomputation "
          f"on all {len(sample_cards)} sampled card1 groups.")
else:
    raise AssertionError(f"{mismatches} of {len(sample_cards)} sampled groups mismatched -- "
                          "scatter-back alignment is broken, aborting.")

# ---------------------------------------------------------------------------
# Verification: mean for isFraud=1 vs isFraud=0, and null count, per column.
# ---------------------------------------------------------------------------
rows = []
print("\nFeature verification (mean isFraud=1 vs isFraud=0, null count):")
for col in C.ISEQL_COLS:
    mean_fraud = df.loc[df[C.TARGET_COL] == 1, col].mean()
    mean_nonfraud = df.loc[df[C.TARGET_COL] == 0, col].mean()
    null_count = df[col].isna().sum()
    rows.append({
        "feature": col,
        "mean_isFraud_1": mean_fraud,
        "mean_isFraud_0": mean_nonfraud,
        "null_count": int(null_count),
    })
    print(f"  {col:26s}  fraud={mean_fraud:.5f}  non-fraud={mean_nonfraud:.5f}  nulls={null_count}")

verification_df = pd.DataFrame(rows)
verification_path = C.OUT_DIR / "feature_verification.csv"
verification_df.to_csv(verification_path, index=False)
print(f"\nSaved feature verification table -> {verification_path}")

print(
    "\nNOTE: D1-D15 timedelta features (e.g., days since card first seen) "
    "capture temporal recency implicitly. This partially overlaps "
    "conceptually with ISEQL+ features P2 (device_novelty_flag) and P4 "
    "(structuring_count_day), which are FORMAL, regulation-traceable "
    "definitions rather than opaque engineered deltas. This distinction -- "
    "implicit vs. formally-defined temporal signal -- is a key thesis "
    "argument for Chapter 6. Both feature types are retained since they "
    "answer conceptually different questions: D-features answer 'how long "
    "since X', ISEQL+ features answer 'how many times X occurred with "
    "what overlap/count within a regulation-defined window'."
)

# ---------------------------------------------------------------------------
# Persist the full enriched dataframe
# ---------------------------------------------------------------------------
df.to_parquet(C.ENRICHED_PARQUET, index=False)
print(f"\nSaved enriched dataframe (fast cache) -> {C.ENRICHED_PARQUET} (shape={df.shape})")

df.to_csv(C.ENRICHED_CSV, index=False)
print(f"Saved enriched dataframe (deliverable) -> {C.ENRICHED_CSV}")

if C.MERGED_PARQUET.exists():
    C.MERGED_PARQUET.unlink()
    print(f"Removed internal intermediate {C.MERGED_PARQUET} (superseded by train_enriched.*)")

print(f"\nTotal step 3 runtime: {time.time()-t0:.1f}s")
print("\nSTEP 3 complete.")
