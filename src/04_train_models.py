"""
STEP 4 -- Train 4 models (A: LR/standard, B: LR/+ISEQL+, C: LGBM/standard,
D: LGBM/+ISEQL+ -- the main model) and evaluate on the validation split.

Uses the 'split' column already assigned chronologically in step 1 (carried
through steps 2-3 untouched) rather than recomputing split boundaries.

Output: outputs/model_results.csv, outputs/model_auc_comparison.{png,pdf}, and
persisted models/encoders in models/ for reuse by 05_shap_analysis.py and
06_ablation.py.
"""
import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

C.apply_fig_style()

t0 = time.time()
print(f"Loading enriched data ...")
df = C.load_enriched()
print(f"  shape: {df.shape}")

masks = C.split_masks(df)
train_df, val_df, test_df = df[masks["train"]], df[masks["val"]], df[masks["test"]]
print(f"  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

STANDARD_COLS = C.get_standard_cols(df)
ISEQL_COLS = [c for c in C.ISEQL_COLS if c in df.columns]
ALL_COLS = STANDARD_COLS + ISEQL_COLS
CAT_COLS = C.get_categorical_cols(df)

print(f"STANDARD_COLS: {len(STANDARD_COLS)}  ISEQL_COLS: {len(ISEQL_COLS)}  "
      f"ALL_COLS: {len(ALL_COLS)}  (categorical among them: {len(CAT_COLS)})")

y_train = train_df[C.TARGET_COL].to_numpy()
y_val = val_df[C.TARGET_COL].to_numpy()

results = []
MODEL_LABELS = {
    "A": "A: LR / standard",
    "B": "B: LR / +ISEQL+",
    "C": "C: LightGBM / standard",
    "D": "D: LightGBM / +ISEQL+",
}


def report(name, y_true, y_score):
    m = C.evaluate_model(y_true, y_score)
    m["model"] = name
    print(f"  [{name}] ROC-AUC={m['roc_auc']:.4f}  F1@0.5={m['f1_at_0.5']:.4f}  "
          f"PR-AUC={m['pr_auc']:.4f}  Precision@5%FPR={m['precision_at_5pct_fpr']:.4f}")
    return m


# ---------------------------------------------------------------------------
# Model A -- Logistic Regression, STANDARD_COLS
# ---------------------------------------------------------------------------
print("\n=== Model A: LogisticRegression, STANDARD_COLS ===")
X_train_a, encoder_a = C.fit_lr_encoder(train_df, STANDARD_COLS, CAT_COLS)
X_val_a = C.transform_lr(val_df, encoder_a)
lr_a = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=C.RANDOM_STATE)
lr_a.fit(X_train_a, y_train)
score_a = lr_a.predict_proba(X_val_a)[:, 1]
results.append(report("A", y_val, score_a))

# ---------------------------------------------------------------------------
# Model B -- Logistic Regression, ALL_COLS
# ---------------------------------------------------------------------------
print("\n=== Model B: LogisticRegression, ALL_COLS (+ISEQL+) ===")
X_train_b, encoder_b = C.fit_lr_encoder(train_df, ALL_COLS, CAT_COLS)
X_val_b = C.transform_lr(val_df, encoder_b)
lr_b = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=C.RANDOM_STATE)
lr_b.fit(X_train_b, y_train)
score_b = lr_b.predict_proba(X_val_b)[:, 1]
results.append(report("B", y_val, score_b))

# NOTE: metric="auc" + first_metric_only=True is required here. Without it,
# LightGBM's sklearn API also tracks the default binary_logloss metric; under
# is_unbalance=True the loss stays uncalibrated and gets monotonically WORSE
# every round even while AUC keeps improving, so early stopping's "all
# tracked metrics must improve" rule freezes best_iteration_ at round 1.
# ---------------------------------------------------------------------------
# Model C -- LightGBM, STANDARD_COLS
# ---------------------------------------------------------------------------
print("\n=== Model C: LightGBM, STANDARD_COLS ===")
X_train_c, cat_present_c = C.build_lgbm_matrix(train_df, STANDARD_COLS, CAT_COLS)
X_val_c, _ = C.build_lgbm_matrix(val_df, STANDARD_COLS, CAT_COLS)
lgbm_c = LGBMClassifier(
    is_unbalance=True, n_estimators=500, learning_rate=0.05, num_leaves=31,
    random_state=C.RANDOM_STATE, verbosity=-1, metric="auc",
)
lgbm_c.fit(
    X_train_c, y_train,
    eval_set=[(X_val_c, y_val)],
    categorical_feature=cat_present_c,
    callbacks=[early_stopping(stopping_rounds=50, first_metric_only=True, verbose=False),
               log_evaluation(period=0)],
)
score_c = lgbm_c.predict_proba(X_val_c)[:, 1]
print(f"  best_iteration_: {lgbm_c.best_iteration_}")
results.append(report("C", y_val, score_c))

# ---------------------------------------------------------------------------
# Model D -- LightGBM, ALL_COLS (main model)
# ---------------------------------------------------------------------------
print("\n=== Model D: LightGBM, ALL_COLS (+ISEQL+, main model) ===")
X_train_d, cat_present_d = C.build_lgbm_matrix(train_df, ALL_COLS, CAT_COLS)
X_val_d, _ = C.build_lgbm_matrix(val_df, ALL_COLS, CAT_COLS)
lgbm_d = LGBMClassifier(
    is_unbalance=True, n_estimators=500, learning_rate=0.05, num_leaves=31,
    random_state=C.RANDOM_STATE, verbosity=-1, metric="auc",
)
lgbm_d.fit(
    X_train_d, y_train,
    eval_set=[(X_val_d, y_val)],
    categorical_feature=cat_present_d,
    callbacks=[early_stopping(stopping_rounds=50, first_metric_only=True, verbose=False),
               log_evaluation(period=0)],
)
score_d = lgbm_d.predict_proba(X_val_d)[:, 1]
print(f"  best_iteration_: {lgbm_d.best_iteration_}")
results.append(report("D", y_val, score_d))

# ---------------------------------------------------------------------------
# Save results table
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(results)[
    ["model", "roc_auc", "f1_at_0.5", "pr_auc", "precision_at_5pct_fpr"]
]
results_df["model_label"] = results_df["model"].map(MODEL_LABELS)
results_path = C.OUT_DIR / "model_results.csv"
results_df.to_csv(results_path, index=False)
print(f"\nSaved model results -> {results_path}")
print(results_df.to_string(index=False))

# ---------------------------------------------------------------------------
# AUC dot plot (Figure 5.1)
#
# A dot plot rather than a bar chart: the four AUCs differ only in the third
# decimal, and bars anchored at zero would need a truncated y-axis, which
# visually exaggerates that difference. In a dot plot the encoding is position
# on a common scale, so a restricted axis range close to the data is legitimate.
# The 2x2 design is made visible directly: one panel per algorithm (row of the
# design), and colour + marker shape per feature set (column of the design).
# ---------------------------------------------------------------------------
auc_by_model = dict(zip(results_df["model"], results_df["roc_auc"]))
# (algorithm panel, [(model key, feature-set label), ...])
DESIGN = [
    ("Logistic Regression", [("A", "standard features only"), ("B", "standard + ISEQL+")]),
    ("LightGBM", [("C", "standard features only"), ("D", "standard + ISEQL+")]),
]
FEATURESET_STYLE = {
    "standard features only": (C.COL_STANDARD, C.MARK_STANDARD),
    "standard + ISEQL+": (C.COL_ISEQL, C.MARK_ISEQL),
}

all_aucs = list(auc_by_model.values())
x_lo = min(0.84, min(all_aucs) - 0.01)
# headroom on the right so the value label next to the rightmost dot stays
# inside the axes rather than overhanging the spine
x_hi = max(0.93, max(all_aucs) + 0.02)

fig, axes = plt.subplots(2, 1, figsize=(C.FIG_WIDTH, 4.0), sharex=True)
for ax, (algo, rows) in zip(axes, DESIGN):
    ypos = [1, 0]
    for y, (mkey, fs_label) in zip(ypos, rows):
        color, marker = FEATURESET_STYLE[fs_label]
        auc = auc_by_model[mkey]
        ax.plot([auc], [y], marker=marker, markersize=9, color=color,
                markeredgecolor=color, linestyle="none", zorder=3)
        ax.text(auc + (x_hi - x_lo) * 0.015, y, f"{auc:.4f}", va="center", ha="left",
                fontsize=C.FS_ANNOT, color=C.TEXT_COLOR, zorder=4)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"Model {mkey}" for mkey, _ in rows])
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlim(x_lo, x_hi)
    ax.set_title(algo, fontsize=C.FS_LABEL, color=C.TEXT_COLOR, loc=C.TITLE_LOC)
    C.style_axes(ax, grid_axis="x")

axes[-1].set_xlabel("ROC-AUC (validation split)")
handles = [
    plt.Line2D([], [], marker=mark, color=col, linestyle="none", markersize=8, label=lbl)
    for lbl, (col, mark) in FEATURESET_STYLE.items()
]
axes[0].legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.30),
               ncol=2, fontsize=C.FS_LEGEND, handletextpad=0.4, columnspacing=1.4)
fig.suptitle("Validation ROC-AUC by model (2x2 design)", fontsize=C.FS_TITLE,
             color=C.TEXT_COLOR, x=0.0, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.97))
png_path, pdf_path = C.save_figure(fig, "model_auc_comparison")
plt.close(fig)
print(f"Saved AUC comparison dot plot -> {png_path} and {pdf_path}")

# ---------------------------------------------------------------------------
# Persist models + encoders + column lists for reuse in steps 5-9
# ---------------------------------------------------------------------------
C.save_artifact(lr_a, "model_a")
C.save_artifact(lr_b, "model_b")
C.save_artifact(lgbm_c, "model_c")
C.save_artifact(lgbm_d, "model_d")
C.save_artifact(encoder_a, "encoder_a")
C.save_artifact(encoder_b, "encoder_b")

column_meta = {
    "STANDARD_COLS": STANDARD_COLS,
    "ISEQL_COLS": ISEQL_COLS,
    "ALL_COLS": ALL_COLS,
    "CAT_COLS": CAT_COLS,
}
with open(C.MODELS_DIR / "column_meta.json", "w") as f:
    json.dump(column_meta, f, indent=2)
print(f"\nPersisted 4 models + 2 LR encoders + column_meta.json -> {C.MODELS_DIR}")

print(f"\nTotal step 4 runtime: {time.time()-t0:.1f}s")
print("\nSTEP 4 complete.")
