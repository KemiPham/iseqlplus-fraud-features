"""
Shared constants and helper functions used across all pipeline scripts
(01_load_data.py ... 06_ablation.py). Kept in one place so column
definitions and metric/encoding logic stay consistent across scripts
that run as separate processes.
"""
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, average_precision_score, roc_curve, precision_score
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
MODELS_DIR = ROOT / "models"
for d in (DATA_DIR, OUT_DIR, MODELS_DIR):
    d.mkdir(exist_ok=True)

RAW_TRANSACTION_CSV = DATA_DIR / "train_transaction.csv"
RAW_IDENTITY_CSV = DATA_DIR / "train_identity.csv"
MERGED_PARQUET = DATA_DIR / "_merged.parquet"          # internal, pipeline-state only
ENRICHED_PARQUET = DATA_DIR / "train_enriched.parquet"  # fast-load cache of the deliverable
ENRICHED_CSV = DATA_DIR / "train_enriched.csv"          # deliverable requested by user

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Chronological split boundaries (days since first transaction)
# ---------------------------------------------------------------------------
TRAIN_MAX_DAY = 120   # train:  day < 120
VAL_MAX_DAY = 150     # val:    120 <= day < 150 ; test: day >= 150

# ---------------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------------
TARGET_COL = "isFraud"
TIME_COL = "TransactionDT"
ID_COL = "TransactionID"

MISSINGNESS_SOURCE_COLS = [f"D{i}" for i in range(1, 16)] + ["dist1", "dist2"]  # 16 cols

# Categorical columns (per Kaggle docs + derived identity/device signals).
# NOTE: card1 is included here because at MODELING time it must be treated
# as a categorical identifier. It is deliberately NOT cast to category dtype
# in the persisted enriched dataframe, because it is needed as a plain
# numeric groupby key for the ISEQL+ feature computation in step 3.
CATEGORICAL_COLS_BASE = (
    ["ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
     "addr1", "addr2", "P_emaildomain", "R_emaildomain"]
    + [f"M{i}" for i in range(1, 10)]
    + ["DeviceType", "device_family", "os_family", "browser_family",
       "id_12", "id_15", "id_16", "id_28", "id_29",
       "id_34", "id_35", "id_36", "id_37", "id_38"]
)

# Columns explicitly cast to 'category' dtype when the enriched df is
# persisted (everything in CATEGORICAL_COLS_BASE except card1).
CATEGORICAL_COLS_FOR_DTYPE = [c for c in CATEGORICAL_COLS_BASE if c != "card1"]

# High-cardinality columns named explicitly in the brief; the general rule
# implemented in fit_lr_encoder/transform_lr is: ANY categorical column with
# > HIGH_CARD_THRESHOLD unique values (measured on the train split only, to
# avoid leakage) gets top-K + 'other' bucketing before one-hot encoding.
# These three are simply the columns that in practice exceed the threshold.
HIGH_CARD_LR_COLS_EXAMPLES = ["P_emaildomain", "R_emaildomain", "addr1"]
HIGH_CARD_THRESHOLD = 20
HIGH_CARD_TOP_K = 15

ISEQL_COLS = [
    "micro_tx_ratio_3min",
    "device_novelty_flag",
    "escalation_ratio_1h",
    "structuring_count_day",
    "tx_count_1h",
    "tx_count_6h",
    "tx_count_24h",
    "micro_fail_count_5min",
]

# Mapping of ISEQL+ pattern -> the 1+ output columns it produces (for ablation)
ISEQL_PATTERNS = {
    "P1_micro_tx_ratio_3min": ["micro_tx_ratio_3min"],
    "P2_device_novelty_flag": ["device_novelty_flag"],
    "P3_escalation_ratio_1h": ["escalation_ratio_1h"],
    "P4_structuring_count_day": ["structuring_count_day"],
    "P5_tx_count_velocity": ["tx_count_1h", "tx_count_6h", "tx_count_24h"],
    "P6_micro_fail_count_5min": ["micro_fail_count_5min"],
}


# ---------------------------------------------------------------------------
# Shared figure style (thesis figures)
#
# Every figure in outputs/ is generated at (approximately) its final printed
# size -- roughly 5.5-6in wide for a single-column figure -- so nothing is
# shrunk down in LaTeX and no text ends up below ~8.5pt on the page. All
# figures share one font family, one gridline colour/alpha, left-aligned
# titles and the same margins, and every one is written twice: a 300 dpi PNG
# preview plus a vector PDF with the same basename.
# ---------------------------------------------------------------------------
FIG_BG = "#fcfcfb"
GRID_COLOR = "#d8d7cf"
GRID_ALPHA = 0.9
GRID_LW = 0.7
SPINE_COLOR = "#c3c2b7"
TICK_COLOR = "#52514e"
TEXT_COLOR = "#0b0b0b"
LABEL_COLOR = "#52514e"
ZERO_LINE_COLOR = "#3d3c39"

# Font sizes, chosen for the final printed size (never below 8.5pt).
FS_TICK = 9.5
FS_LABEL = 10.0
FS_LEGEND = 9.5
FS_TITLE = 11.0
FS_ANNOT = 9.0
FS_ANNOT_SMALL = 8.5

# Colourblind-safe palette (Okabe-Ito derived). Colour is never the only
# encoding -- markers/linestyles/hatching carry the same distinction.
COL_STANDARD = "#0173b2"   # standard features only  (Models A, C)
COL_ISEQL = "#de8f05"      # standard + ISEQL+       (Models B, D)
COL_POS = "#d55e00"        # SHAP pushes toward fraud
COL_NEG = "#0173b2"        # SHAP pushes away from fraud
COL_HIGHLIGHT = "#009e73"  # highlighted ISEQL+ feature in the case comparison

MARK_STANDARD = "o"
MARK_ISEQL = "s"

# Single-column figure width used across the thesis figures.
FIG_WIDTH = 6.0

TITLE_LOC = "left"  # consistent title placement everywhere


def apply_fig_style():
    """Set the shared rcParams. Call once at the top of every plotting script."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": FS_TICK,
        "axes.titlesize": FS_TITLE,
        "axes.labelsize": FS_LABEL,
        "axes.titlelocation": TITLE_LOC,
        "axes.titlepad": 8.0,
        "axes.labelcolor": LABEL_COLOR,
        "axes.edgecolor": SPINE_COLOR,
        "axes.facecolor": FIG_BG,
        "axes.grid": False,
        "figure.facecolor": FIG_BG,
        "figure.titlesize": FS_TITLE,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "xtick.color": TICK_COLOR,
        "ytick.color": TICK_COLOR,
        "legend.fontsize": FS_LEGEND,
        "legend.frameon": False,
        "grid.color": GRID_COLOR,
        "grid.alpha": GRID_ALPHA,
        "grid.linewidth": GRID_LW,
        "savefig.facecolor": FIG_BG,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
        "pdf.fonttype": 42,   # embed TrueType, keeps text selectable/searchable
        "ps.fonttype": 42,
    })


def style_axes(ax, grid_axis="y"):
    """Apply the shared spine/tick/grid treatment to one Axes."""
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(SPINE_COLOR)
    ax.set_facecolor(FIG_BG)
    ax.tick_params(colors=TICK_COLOR, labelsize=FS_TICK)
    if grid_axis in ("x", "both"):
        ax.xaxis.grid(True, color=GRID_COLOR, alpha=GRID_ALPHA, linewidth=GRID_LW)
    if grid_axis in ("y", "both"):
        ax.yaxis.grid(True, color=GRID_COLOR, alpha=GRID_ALPHA, linewidth=GRID_LW)
    ax.set_axisbelow(True)


def signed_bar_xlim(vals, pad_frac=0.28):
    """x-limits for a horizontal bar chart whose bars can point either way,
    leaving room on BOTH sides for the value label printed outside each bar.
    Without the left-hand pad, labels on short negative bars collide with the
    y tick labels."""
    vals = [float(v) for v in vals]
    mag = max(abs(min(vals)), abs(max(vals))) or 1.0
    pad = mag * pad_frac
    return min(0.0, min(vals)) - pad, max(0.0, max(vals)) + pad


def save_figure(fig, basename, out_dir=None):
    """Save a figure as BOTH a 300 dpi PNG (preview/fallback) and a vector PDF
    with the same basename. Returns (png_path, pdf_path)."""
    out_dir = OUT_DIR if out_dir is None else Path(out_dir)
    png_path = out_dir / f"{basename}.png"
    pdf_path = out_dir / f"{basename}.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)  # vector: dpi is ignored
    return png_path, pdf_path


def get_missingness_cols():
    return [f"{c}_was_missing" for c in MISSINGNESS_SOURCE_COLS]


def get_categorical_cols(df):
    return [c for c in CATEGORICAL_COLS_BASE if c in df.columns]


def get_numeric_standard_cols(df):
    v_cols = sorted([c for c in df.columns if re.fullmatch(r"V\d+", c)],
                     key=lambda c: int(c[1:]))
    c_cols = sorted([c for c in df.columns if re.fullmatch(r"C\d+", c)],
                     key=lambda c: int(c[1:]))
    d_cols = sorted([c for c in df.columns if re.fullmatch(r"D\d+", c)],
                     key=lambda c: int(c[1:]))
    id_num_cols = [f"id_{i:02d}" for i in range(1, 12) if f"id_{i:02d}" in df.columns]
    others = [c for c in ["TransactionAmt", "dist1", "dist2", "screen_width", "screen_height"]
              if c in df.columns]
    return others + c_cols + d_cols + v_cols + id_num_cols


def get_standard_cols(df):
    """Numeric Vesta features + encoded categoricals + missingness indicators
    + has_identity + parsed identity signals (os/browser family, screen w/h are
    already included via get_numeric_standard_cols / get_categorical_cols)."""
    cat = get_categorical_cols(df)
    num = get_numeric_standard_cols(df)
    miss = [c for c in get_missingness_cols() if c in df.columns]
    extra = ["has_identity"] if "has_identity" in df.columns else []
    return num + cat + miss + extra


def get_all_cols(df):
    return get_standard_cols(df) + [c for c in ISEQL_COLS if c in df.columns]


# ---------------------------------------------------------------------------
# Feature-matrix builders
# ---------------------------------------------------------------------------
def build_lgbm_matrix(df, cols, cat_cols):
    """Cast categorical columns to 'category' dtype (NaN left as-is, LightGBM
    handles missing categories natively). Numeric columns are filled with 0.
    card1 is included in cat_cols here even though it stays numeric in the
    persisted dataframe -- the cast happens only on this transient copy."""
    X = df[cols].copy()
    cat_present = [c for c in cat_cols if c in X.columns]
    for c in cat_present:
        X[c] = X[c].astype("category")
    numeric_cols = [c for c in cols if c not in cat_present]
    X[numeric_cols] = X[numeric_cols].fillna(0)
    return X, cat_present


def fit_lr_encoder(train_df, cols, cat_cols, top_k=HIGH_CARD_TOP_K,
                    high_card_threshold=HIGH_CARD_THRESHOLD):
    """Fit one-hot encoding + scaling on the TRAIN split only (no leakage).
    Categorical NaNs become the explicit string 'missing'. Any categorical
    column with > high_card_threshold unique train values is bucketed to its
    top_k most frequent categories + 'other' before dummying."""
    cat_cols = [c for c in cat_cols if c in cols]
    num_cols = [c for c in cols if c not in cat_cols]

    cat_df = train_df[cat_cols].astype(object)
    top_values = {}
    for c in cat_cols:
        col = cat_df[c].where(cat_df[c].notna(), "missing")
        nun = col[col != "missing"].nunique()
        if nun > high_card_threshold:
            top_vals = col[col != "missing"].value_counts().nlargest(top_k).index.tolist()
            top_values[c] = top_vals
            col = col.where(col.isin(top_vals) | (col == "missing"), "other")
        else:
            top_values[c] = None
        cat_df[c] = col

    dummies_train = pd.get_dummies(cat_df, columns=cat_cols, drop_first=True)
    scaler = StandardScaler()
    num_train = train_df[num_cols].fillna(0)
    num_train_scaled = scaler.fit_transform(num_train)

    X_train = np.hstack([num_train_scaled, dummies_train.values.astype(float)])
    encoder = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "top_values": top_values,
        "dummy_cols": dummies_train.columns.tolist(),
        "scaler": scaler,
        "high_card_threshold": high_card_threshold,
        "top_k": top_k,
    }
    return X_train, encoder


def transform_lr(df, encoder):
    """Apply an encoder fit by fit_lr_encoder to a new split (val/test)."""
    num_cols, cat_cols = encoder["num_cols"], encoder["cat_cols"]
    cat_df = df[cat_cols].astype(object)
    for c in cat_cols:
        col = cat_df[c].where(cat_df[c].notna(), "missing")
        tv = encoder["top_values"][c]
        if tv is not None:
            col = col.where(col.isin(tv) | (col == "missing"), "other")
        cat_df[c] = col
    dummies = pd.get_dummies(cat_df, columns=cat_cols, drop_first=True)
    dummies = dummies.reindex(columns=encoder["dummy_cols"], fill_value=False)
    num = df[num_cols].fillna(0)
    num_scaled = encoder["scaler"].transform(num)
    return np.hstack([num_scaled, dummies.values.astype(float)])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def precision_at_fpr(y_true, y_score, target_fpr=0.05):
    fpr, tpr, thresh = roc_curve(y_true, y_score)
    idx = int(np.argmin(np.abs(fpr - target_fpr)))
    thr = thresh[idx]
    y_pred = (y_score >= thr).astype(int)
    return precision_score(y_true, y_pred, zero_division=0)


def evaluate_model(y_true, y_score):
    return {
        "roc_auc": roc_auc_score(y_true, y_score),
        "f1_at_0.5": f1_score(y_true, (y_score >= 0.5).astype(int), zero_division=0),
        "pr_auc": average_precision_score(y_true, y_score),
        "precision_at_5pct_fpr": precision_at_fpr(y_true, y_score, 0.05),
    }


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
def save_artifact(obj, name):
    joblib.dump(obj, MODELS_DIR / f"{name}.pkl")


def load_artifact(name):
    return joblib.load(MODELS_DIR / f"{name}.pkl")


def load_enriched():
    """Load the enriched dataframe, preferring the fast parquet cache."""
    if ENRICHED_PARQUET.exists():
        return pd.read_parquet(ENRICHED_PARQUET)
    return pd.read_csv(ENRICHED_CSV, low_memory=False)


def split_masks(df):
    return {
        "train": df["split"] == "train",
        "val": df["split"] == "val",
        "test": df["split"] == "test",
    }
