"""
STEP 7 -- Temporal stability: Model C vs Model D, validation (month 5) vs
test (month 6).
STEP 8 -- Ablation study: drop each of the 6 ISEQL+ patterns from ALL_COLS,
retrain LightGBM, measure the AUC contribution of each pattern.
STEP 9 -- Assemble outputs/results_summary.md from all prior steps' outputs.

Output: outputs/temporal_stability.csv, outputs/temporal_drift.{png,pdf},
outputs/ablation_study.csv, outputs/ablation_chart.{png,pdf},
outputs/results_summary.md
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

C.apply_fig_style()

t0 = time.time()
BLUE, RED, AQUA = C.COL_STANDARD, C.COL_ISEQL, "#009e73"

print("Loading enriched data + models + column metadata ...")
df = C.load_enriched()
masks = C.split_masks(df)
train_df, val_df, test_df = df[masks["train"]], df[masks["val"]], df[masks["test"]]

with open(C.MODELS_DIR / "column_meta.json") as f:
    meta = json.load(f)
STANDARD_COLS, ISEQL_COLS, ALL_COLS, CAT_COLS = (
    meta["STANDARD_COLS"], meta["ISEQL_COLS"], meta["ALL_COLS"], meta["CAT_COLS"]
)
model_c = C.load_artifact("model_c")
model_d = C.load_artifact("model_d")

y_val = val_df[C.TARGET_COL].to_numpy()
y_test = test_df[C.TARGET_COL].to_numpy()

model_results = pd.read_csv(C.OUT_DIR / "model_results.csv")

# ---------------------------------------------------------------------------
# STEP 7 -- Temporal stability: Model C & D on val (month 5) vs test (month 6)
# ---------------------------------------------------------------------------
print("\n=== STEP 7: Temporal stability (Model C & D, val vs test) ===")

X_val_c, _ = C.build_lgbm_matrix(val_df, STANDARD_COLS, CAT_COLS)
X_test_c, _ = C.build_lgbm_matrix(test_df, STANDARD_COLS, CAT_COLS)
X_val_d, _ = C.build_lgbm_matrix(val_df, ALL_COLS, CAT_COLS)
X_test_d, _ = C.build_lgbm_matrix(test_df, ALL_COLS, CAT_COLS)

auc_c_val = roc_auc_score(y_val, model_c.predict_proba(X_val_c)[:, 1])
auc_c_test = roc_auc_score(y_test, model_c.predict_proba(X_test_c)[:, 1])
auc_d_val = roc_auc_score(y_val, model_d.predict_proba(X_val_d)[:, 1])
auc_d_test = roc_auc_score(y_test, model_d.predict_proba(X_test_d)[:, 1])

drop_c = auc_c_test - auc_c_val
drop_d = auc_d_test - auc_d_val

temporal_df = pd.DataFrame([
    {"model": "C: LightGBM / standard", "auc_month5_val": auc_c_val,
     "auc_month6_test": auc_c_test, "auc_drop": drop_c, "abs_drop": abs(drop_c)},
    {"model": "D: LightGBM / +ISEQL+", "auc_month5_val": auc_d_val,
     "auc_month6_test": auc_d_test, "auc_drop": drop_d, "abs_drop": abs(drop_d)},
])
print(temporal_df.to_string(index=False))

more_stable = temporal_df.loc[temporal_df["abs_drop"].idxmin(), "model"]
print(f"\nMore stable model (smaller |AUC drop| month5->month6): {more_stable}")

temporal_path = C.OUT_DIR / "temporal_stability.csv"
temporal_df.to_csv(temporal_path, index=False)
print(f"Saved -> {temporal_path}")

# Model C = solid line + circles, Model D = dashed line + squares, so the two
# series stay distinguishable in greyscale and for colourblind readers. The two
# drops are near-identical, so the exact numbers are annotated on the figure
# rather than left to be inferred from how close the lines look.
fig, ax = plt.subplots(figsize=(C.FIG_WIDTH, 4.0))
periods = ["Month 5 (val)", "Month 6 (test)"]
ax.plot(periods, [auc_c_val, auc_c_test], marker="o", markersize=7, linewidth=2.0,
        linestyle="-", color=BLUE, label="C: LightGBM / standard")
ax.plot(periods, [auc_d_val, auc_d_test], marker="s", markersize=7, linewidth=2.0,
        linestyle="--", color=RED, label="D: LightGBM / +ISEQL+")
for x, y in zip(periods, [auc_c_val, auc_c_test]):
    ax.text(x, y - 0.004, f"{y:.4f}", ha="center", va="top",
            fontsize=C.FS_ANNOT, color=C.TEXT_COLOR)
for x, y in zip(periods, [auc_d_val, auc_d_test]):
    ax.text(x, y + 0.004, f"{y:.4f}", ha="center", va="bottom",
            fontsize=C.FS_ANNOT, color=C.TEXT_COLOR)
ax.margins(x=0.14, y=0.55)  # headroom for the annotation box above the lines
ax.set_ylabel("ROC-AUC")
ax.set_title("Temporal stability: ROC-AUC by period", color=C.TEXT_COLOR,
             fontsize=C.FS_TITLE, loc=C.TITLE_LOC)
ax.legend(loc="lower left", fontsize=C.FS_LEGEND)
drift_note = (f"Drop month 5 → month 6:  C = {drop_c:+.4f},  D = {drop_d:+.4f}"
              f"  (difference = {abs(drop_d - drop_c):.5f})")
ax.text(0.99, 0.99, drift_note, transform=ax.transAxes, ha="right", va="top",
        fontsize=C.FS_ANNOT, color=C.TEXT_COLOR,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff",
                  edgecolor=C.SPINE_COLOR, linewidth=0.7))
C.style_axes(ax, grid_axis="y")
fig.tight_layout()
drift_png, drift_pdf = C.save_figure(fig, "temporal_drift")
plt.close(fig)
print(f"Saved -> {drift_png} and {drift_pdf}")

# ---------------------------------------------------------------------------
# STEP 8 -- Ablation study: drop each ISEQL+ pattern, retrain, measure AUC drop
# ---------------------------------------------------------------------------
print("\n=== STEP 8: Ablation study (6 ISEQL+ patterns) ===")
y_train = train_df[C.TARGET_COL].to_numpy()
full_auc_d = float(model_results.loc[model_results["model"] == "D", "roc_auc"].iloc[0])
print(f"Full Model D (all 8 ISEQL+ columns) validation AUC: {full_auc_d:.4f}")

ablation_rows = []
for pattern_name, drop_cols in C.ISEQL_PATTERNS.items():
    cols_ablated = [c for c in ALL_COLS if c not in drop_cols]
    print(f"\nTraining LightGBM without {pattern_name} (dropping {drop_cols}) ...")
    X_train_ab, cat_ab = C.build_lgbm_matrix(train_df, cols_ablated, CAT_COLS)
    X_val_ab, _ = C.build_lgbm_matrix(val_df, cols_ablated, CAT_COLS)
    m = LGBMClassifier(
        is_unbalance=True, n_estimators=500, learning_rate=0.05, num_leaves=31,
        random_state=C.RANDOM_STATE, verbosity=-1, metric="auc",
    )
    m.fit(
        X_train_ab, y_train, eval_set=[(X_val_ab, y_val)], categorical_feature=cat_ab,
        callbacks=[early_stopping(stopping_rounds=50, first_metric_only=True, verbose=False),
                   log_evaluation(period=0)],
    )
    score = m.predict_proba(X_val_ab)[:, 1]
    auc = roc_auc_score(y_val, score)
    auc_drop = full_auc_d - auc
    print(f"  AUC without {pattern_name}: {auc:.4f}  (drop vs full Model D: {auc_drop:+.4f})")
    ablation_rows.append({
        "pattern": pattern_name, "dropped_columns": ", ".join(drop_cols),
        "auc_without_pattern": auc, "auc_drop_vs_full_model_d": auc_drop,
    })

ablation_df = pd.DataFrame(ablation_rows).sort_values(
    "auc_drop_vs_full_model_d", ascending=False
).reset_index(drop=True)
ablation_df["rank"] = np.arange(1, len(ablation_df) + 1)
print("\nRanked pattern contribution (largest AUC drop when removed = most important):")
print(ablation_df[["rank", "pattern", "auc_without_pattern", "auc_drop_vs_full_model_d"]]
      .to_string(index=False))

ablation_path = C.OUT_DIR / "ablation_study.csv"
ablation_df.to_csv(ablation_path, index=False)
print(f"\nSaved -> {ablation_path}")

# Positive bars (removing the pattern costs AUC -> the pattern contributes) and
# negative bars (removing it helps) get different colours AND different hatching,
# so the direction of the contribution reads at a glance without relying on hue.
fig, ax = plt.subplots(figsize=(C.FIG_WIDTH, 3.8))
plot_df = ablation_df.iloc[::-1]
ab_vals = plot_df["auc_drop_vs_full_model_d"].to_numpy()
bar_colors = [C.COL_POS if v >= 0 else C.COL_NEG for v in ab_vals]
bars = ax.barh(plot_df["pattern"], ab_vals, color=bar_colors, height=0.62)
for bar, val in zip(bars, ab_vals):
    if val < 0:
        bar.set_hatch("///")
        bar.set_edgecolor("#ffffff")
        bar.set_linewidth(0.5)
for bar, val in zip(bars, ab_vals):
    offset = (abs(ab_vals).max() * 0.05) * (1 if val >= 0 else -1)
    ax.text(val + offset, bar.get_y() + bar.get_height() / 2, f"{val:+.4f}",
            va="center", ha="left" if val >= 0 else "right",
            fontsize=C.FS_ANNOT, color=C.TEXT_COLOR)
ax.axvline(0, color=C.ZERO_LINE_COLOR, linewidth=2.0, zorder=3)
ax.set_xlim(*C.signed_bar_xlim(ab_vals, pad_frac=0.55))
ax.tick_params(axis="y", pad=6)
ax.set_xlabel("AUC drop vs full Model D when pattern is removed")
ax.set_title("Ablation study: ISEQL+ pattern contribution", color=C.TEXT_COLOR,
             fontsize=C.FS_TITLE, loc=C.TITLE_LOC)
legend_handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor=C.COL_POS,
                  label="removing the pattern lowers AUC (positive contribution)"),
    plt.Rectangle((0, 0), 1, 1, facecolor=C.COL_NEG, hatch="///", edgecolor="#ffffff",
                  linewidth=0.5, label="removing the pattern raises AUC"),
]
ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.0, -0.20),
          fontsize=C.FS_LEGEND, handlelength=1.6, handletextpad=0.5,
          borderaxespad=0.0)
C.style_axes(ax, grid_axis="x")
fig.tight_layout()
ablation_png, ablation_pdf = C.save_figure(fig, "ablation_chart")
plt.close(fig)
print(f"Saved -> {ablation_png} and {ablation_pdf}")

# ---------------------------------------------------------------------------
# STEP 9 -- Final results_summary.md
# ---------------------------------------------------------------------------
print("\n=== STEP 9: Assembling results_summary.md ===")

shap_importance = pd.read_csv(C.OUT_DIR / "shap_feature_importance.csv")
shap_iseql = shap_importance[shap_importance["feature"].isin(ISEQL_COLS)].sort_values("rank")
n_total_features = len(shap_importance)

lines = []
lines.append("# ISEQL+ Fraud Detection -- Results Summary")
lines.append("")
lines.append("Thesis-ready numbers, organized by section. Generated automatically by "
              "`src/06_ablation.py` from the CSV/PNG artifacts in `outputs/`.")
lines.append("")

lines.append("## 5.1 Performance (4 models, validation split)")
lines.append("")
lines.append("| Model | ROC-AUC | F1 @ 0.5 | PR-AUC | Precision @ 5% FPR |")
lines.append("|---|---|---|---|---|")
for _, row in model_results.iterrows():
    lines.append(f"| {row['model_label']} | {row['roc_auc']:.4f} | {row['f1_at_0.5']:.4f} "
                 f"| {row['pr_auc']:.4f} | {row['precision_at_5pct_fpr']:.4f} |")
lines.append("")
lines.append(f"![Model AUC comparison](model_auc_comparison.png)")
lines.append("")

lines.append("## 5.2 SHAP global importance -- 8 ISEQL+ features (Model D)")
lines.append("")
lines.append(f"Ranked among all {n_total_features} features used by Model D.")
lines.append("")
lines.append("| Rank | ISEQL+ feature | mean(|SHAP|) |")
lines.append("|---|---|---|")
for _, row in shap_iseql.iterrows():
    lines.append(f"| #{int(row['rank'])} of {n_total_features} | {row['feature']} "
                 f"| {row['mean_abs_shap']:.4f} |")
lines.append("")
lines.append("![SHAP global importance](shap_global_importance.png)")
lines.append("")
lines.append("![ISEQL+ feature ranks](shap_iseql_ranks.png)")
lines.append("")
lines.append("See `interpretability_comparison.md` for the Model C vs Model D "
             "per-case comparison (Step 6).")
lines.append("")

lines.append("## 5.3 Temporal stability (Model C vs Model D, month 5 -> month 6)")
lines.append("")
lines.append("| Model | AUC month 5 (val) | AUC month 6 (test) | AUC drop |")
lines.append("|---|---|---|---|")
for _, row in temporal_df.iterrows():
    lines.append(f"| {row['model']} | {row['auc_month5_val']:.4f} "
                 f"| {row['auc_month6_test']:.4f} | {row['auc_drop']:+.4f} |")
lines.append("")
lines.append(f"**More stable model (smaller |AUC drop|): {more_stable}**")
lines.append("")
lines.append("![Temporal drift](temporal_drift.png)")
lines.append("")

lines.append("## 5.4 Ablation study (6 ISEQL+ patterns, ranked by contribution)")
lines.append("")
lines.append(f"Full Model D validation AUC: {full_auc_d:.4f}")
lines.append("")
lines.append("| Rank | Pattern | Dropped column(s) | AUC without pattern | AUC drop vs full Model D |")
lines.append("|---|---|---|---|---|")
for _, row in ablation_df.iterrows():
    lines.append(f"| {int(row['rank'])} | {row['pattern']} | `{row['dropped_columns']}` "
                 f"| {row['auc_without_pattern']:.4f} | {row['auc_drop_vs_full_model_d']:+.4f} |")
lines.append("")
lines.append("![Ablation chart](ablation_chart.png)")
lines.append("")

lines.append("## Notes / caveats carried over from earlier steps")
lines.append("")
lines.append("- The brief's parenthetical count for the missingness-indicator block "
             "(\"D1-D15 and dist1-dist2 (16 columns)\") undercounts by one -- D1-D15 is "
             "15 columns + dist1/dist2 is 2, i.e. 17 columns/indicators. All 17 were "
             "created and used (`get_missingness_cols()` in `src/common.py`).")
lines.append("- D1-D15 timedelta features capture temporal recency implicitly, which "
             "partially overlaps conceptually with ISEQL+ P2 (device_novelty_flag) and "
             "P4 (structuring_count_day) -- the latter are FORMAL, regulation-traceable "
             "definitions rather than opaque engineered deltas. Both are retained: "
             "D-features answer 'how long since X', ISEQL+ features answer 'how many "
             "times X occurred with what overlap/count within a regulation-defined "
             "window'.")
lines.append("- Only 1 of 43 sampled fraud cases had an ISEQL+ feature among Model D's "
             "top-5 local SHAP drivers; the 4 highest-confidence fraud cases in Step 5/6 "
             "are dominated by anonymized `V*`/`C*` Vesta features for both models. "
             "ISEQL+ features contribute more broadly across borderline transactions "
             "than at the extreme high-confidence end -- see `interpretability_comparison.md`.")
lines.append("")

summary_path = C.OUT_DIR / "results_summary.md"
with open(summary_path, "w") as f:
    f.write("\n".join(lines))
print(f"Saved -> {summary_path}")

print(f"\nTotal step 6 (7-9) runtime: {time.time()-t0:.1f}s")
print("\nSTEPS 7-9 complete.")
