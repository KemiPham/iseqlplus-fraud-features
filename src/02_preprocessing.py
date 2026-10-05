"""
STEP 2 -- Feature cleaning and preprocessing.

- device_family: cleaned/bucketed DeviceInfo (needed as an ISEQL+ P2 group key)
- os_family / browser_family: parsed from id_30 / id_31
- screen_width / screen_height: parsed from id_33
- was_missing indicators for D1-D15, dist1-dist2 (16 cols), created BEFORE any
  NaN filling (none of these raw columns are filled here -- that happens at
  model-matrix build time in 04_train_models.py)
- categorical dtype casting for LightGBM (card1 stays numeric -- it's the
  ISEQL+ groupby key in step 3; it is cast to category only when building the
  modeling feature matrix in 04_train_models.py)

Output: data/_merged.parquet (overwritten in place, consumed by
03_iseql_features.py).
"""
import re
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
print(f"  shape: {df.shape}")

new_cols_before = set(df.columns)

# ---------------------------------------------------------------------------
# device_family from DeviceInfo
# ---------------------------------------------------------------------------
def clean_device(raw):
    if pd.isna(raw):
        return np.nan
    s = str(raw)
    if " Build/" in s:
        s = s.split(" Build/")[0]
    return s.strip().lower()

device_family = df["DeviceInfo"].map(clean_device)
counts = device_family.value_counts(dropna=True)
rare_values = counts[counts < 20].index
device_family = device_family.where(~device_family.isin(rare_values), "rare_device")
df["device_family"] = device_family

n_families_before = df["DeviceInfo"].nunique(dropna=True)
n_families_after = df["device_family"].nunique(dropna=True)
print(f"device_family: {n_families_before} raw DeviceInfo values -> "
      f"{n_families_after} families after Build/-stripping + rare-bucketing "
      f"(<20 occurrences -> 'rare_device')")

# ---------------------------------------------------------------------------
# os_family / browser_family: take the substring before the first digit
# ("Windows 10" -> "Windows", "iOS 11.2.1" -> "iOS", "Mac OS X 10.11.6" ->
#  "Mac OS X"); strings with no digit (e.g. "other") are kept as-is.
# ---------------------------------------------------------------------------
def family_before_version(raw):
    if pd.isna(raw):
        return np.nan
    s = str(raw)
    m = re.search(r"\d", s)
    if m is None:
        return s.strip().lower()
    return s[: m.start()].strip().lower() or s.strip().lower()

df["os_family"] = df["id_30"].map(family_before_version) if "id_30" in df.columns else np.nan
df["browser_family"] = df["id_31"].map(family_before_version) if "id_31" in df.columns else np.nan
print(f"os_family: {df['os_family'].nunique(dropna=True)} unique values "
      f"(from id_30, {df['id_30'].nunique(dropna=True)} raw values)")
print(f"browser_family: {df['browser_family'].nunique(dropna=True)} unique values "
      f"(from id_31, {df['id_31'].nunique(dropna=True)} raw values)")

# ---------------------------------------------------------------------------
# screen_width / screen_height from id_33 ("1920x1080")
# ---------------------------------------------------------------------------
def parse_resolution(raw):
    if pd.isna(raw):
        return (np.nan, np.nan)
    m = re.fullmatch(r"(\d+)x(\d+)", str(raw).strip())
    if not m:
        return (np.nan, np.nan)
    return (float(m.group(1)), float(m.group(2)))

if "id_33" in df.columns:
    res = df["id_33"].map(parse_resolution)
    df["screen_width"] = res.map(lambda t: t[0])
    df["screen_height"] = res.map(lambda t: t[1])
else:
    df["screen_width"] = np.nan
    df["screen_height"] = np.nan
n_parsed = df["screen_width"].notna().sum()
n_malformed = df["id_33"].notna().sum() - n_parsed if "id_33" in df.columns else 0
print(f"screen_width/height: parsed {n_parsed} rows from id_33 "
      f"({n_malformed} non-null id_33 values did not match WxH and became NaN)")

# ---------------------------------------------------------------------------
# Missingness indicators for D1-D15, dist1, dist2 (16 cols) -- BEFORE any fill
# ---------------------------------------------------------------------------
print(f"\nCreating {len(C.MISSINGNESS_SOURCE_COLS)} missingness indicator columns "
      f"for D1-D15 and dist1-dist2 ...")
for col in C.MISSINGNESS_SOURCE_COLS:
    ind_col = f"{col}_was_missing"
    if col in df.columns:
        df[ind_col] = df[col].isna().astype("int8")
    else:
        df[ind_col] = 0
        print(f"  WARNING: source column {col} not found -- {ind_col} set to all-0")
print("NOTE: missingness indicators are created for D1-D15/dist1-dist2 only; "
      "the V-block (339 columns) is skipped to avoid excessive dimensionality "
      "from adding 339 more binary columns.")

# ---------------------------------------------------------------------------
# Categorical dtype casting (persisted df) -- card1 excluded, stays numeric
# ---------------------------------------------------------------------------
print(f"\nCasting {len(C.CATEGORICAL_COLS_FOR_DTYPE)} columns to pandas 'category' dtype "
      f"(card1 stays numeric int for step-3 groupby key) ...")
missing_cat_cols = []
for col in C.CATEGORICAL_COLS_FOR_DTYPE:
    if col in df.columns:
        df[col] = df[col].astype("category")
    else:
        missing_cat_cols.append(col)
if missing_cat_cols:
    print(f"  WARNING: expected categorical columns not found in data: {missing_cat_cols}")

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
new_cols = sorted(set(df.columns) - new_cols_before)
print(f"\nNew columns created in step 2 ({len(new_cols)}): {new_cols}")
expected_new = {"device_family", "os_family", "browser_family", "screen_width", "screen_height"}
expected_new |= set(C.get_missingness_cols())
missing_expected = expected_new - set(df.columns)
assert not missing_expected, f"expected new columns missing: {missing_expected}"
print("Confirmed: all expected new columns were created successfully.")

standard_cols = C.get_standard_cols(df)
print(f"\nTotal STANDARD_COLS after all preprocessing: {len(standard_cols)}")
print(f"  (numeric Vesta/C/D/V/dist/id_01-11 features: {len(C.get_numeric_standard_cols(df))}, "
      f"categorical: {len(C.get_categorical_cols(df))}, "
      f"missingness indicators: {len(C.get_missingness_cols())}, "
      f"has_identity: 1)")

df.to_parquet(C.MERGED_PARQUET, index=False)
print(f"\nSaved preprocessed dataframe -> {C.MERGED_PARQUET} "
      f"(shape={df.shape}, {time.time()-t0:.1f}s total)")
print("\nSTEP 2 complete.")
