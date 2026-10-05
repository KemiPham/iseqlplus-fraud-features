"""
STEP 1 -- Load and prepare data.

Loads train_transaction.csv and train_identity.csv, left-joins on
TransactionID (transaction is the base -- not every transaction has a
matching identity record), builds has_identity, sorts chronologically by
TransactionDT, and assigns a train/val/test split by elapsed day.

Output: data/_merged.parquet (internal pipeline state, consumed by
02_preprocessing.py) and outputs/split_summary.csv (deliverable).
"""
import time
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

t0 = time.time()

print(f"Loading {C.RAW_TRANSACTION_CSV} ...")
transaction = pd.read_csv(C.RAW_TRANSACTION_CSV, low_memory=False)
print(f"  transaction shape: {transaction.shape}  ({time.time()-t0:.1f}s elapsed)")

print(f"Loading {C.RAW_IDENTITY_CSV} ...")
identity = pd.read_csv(C.RAW_IDENTITY_CSV, low_memory=False)
print(f"  identity shape: {identity.shape}  ({time.time()-t0:.1f}s elapsed)")

print("Merging (left join on TransactionID, transaction as base) ...")
merged = transaction.merge(identity, on=C.ID_COL, how="left", validate="one_to_one")
print(f"  merged shape: {merged.shape}")
assert len(merged) == len(transaction), "left join must preserve transaction row count"

# has_identity = 1 if the row had a matching identity record: any identity
# column (DeviceInfo or any id_* column) is not null.
identity_cols = [c for c in identity.columns if c != C.ID_COL]
merged["has_identity"] = merged[identity_cols].notna().any(axis=1).astype("int8")
print(f"  has_identity rate: {merged['has_identity'].mean():.4f} "
      f"({merged['has_identity'].sum()} / {len(merged)} rows matched an identity record)")

print("Sorting chronologically by TransactionDT ...")
merged = merged.sort_values(C.TIME_COL, kind="mergesort").reset_index(drop=True)

# day = elapsed days since first transaction (integer bucket)
day = (merged[C.TIME_COL] - merged[C.TIME_COL].min()) // 86400
merged["day"] = day.astype("int32")

conditions_train = merged["day"] < C.TRAIN_MAX_DAY
conditions_val = (merged["day"] >= C.TRAIN_MAX_DAY) & (merged["day"] < C.VAL_MAX_DAY)
conditions_test = merged["day"] >= C.VAL_MAX_DAY

merged["split"] = "test"
merged.loc[conditions_train, "split"] = "train"
merged.loc[conditions_val, "split"] = "val"

print("\nRow counts and fraud rate per split:")
summary = (
    merged.groupby("split")[C.TARGET_COL]
    .agg(n_rows="count", n_fraud="sum", fraud_rate="mean")
    .reindex(["train", "val", "test"])
    .reset_index()
)
print(summary.to_string(index=False))

summary_path = C.OUT_DIR / "split_summary.csv"
summary.to_csv(summary_path, index=False)
print(f"\nSaved split summary -> {summary_path}")

merged.to_parquet(C.MERGED_PARQUET, index=False)
print(f"Saved merged+split dataframe -> {C.MERGED_PARQUET}  "
      f"(shape={merged.shape}, {time.time()-t0:.1f}s total)")

# sanity checks before handing off to step 2
assert summary_path.exists()
assert C.MERGED_PARQUET.exists()
print("\nSTEP 1 complete.")
