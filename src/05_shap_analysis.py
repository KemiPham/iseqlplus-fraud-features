"""
STEP 5 -- SHAP explainability analysis (Model D, the main LightGBM+ISEQL+
model).
STEP 6 -- Interpretability comparison: for the same 5 fraud cases, compare
Model C's (standard-only) top SHAP features against Model D's.

Output (each figure as both a 300 dpi .png and a vector .pdf):
outputs/shap_global_importance, outputs/shap_iseql_ranks,
outputs/shap_waterfall_fraud_1..5, outputs/shap_case5_comparison,
plus outputs/interpretability_comparison.md and
outputs/shap_feature_importance.csv (full ranking, saved as a bonus alongside
the console printout).
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
import shap

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

C.apply_fig_style()

t0 = time.time()
BLUE, RED = C.COL_STANDARD, C.COL_ISEQL       # standard feature vs ISEQL+ pattern
POS_RED, NEG_BLUE = C.COL_POS, C.COL_NEG      # SHAP convention: red pushes toward fraud, blue away
ISEQL_HATCH = "///"  # ISEQL+ bars are also hatched, so colour is never the only cue

# Display-only relabelling for chart text. Column names, the CSV/markdown
# outputs and all computations keep the original feature names.
DISPLAY_NAMES = {"card1": "card_number", "card2": "card_attr2"}
def display_label(name):
    return DISPLAY_NAMES.get(name, name)

print("Loading enriched data + Model C/D + column metadata ...")
df = C.load_enriched()
masks = C.split_masks(df)
val_df = df[masks["val"]].reset_index(drop=True)

with open(C.MODELS_DIR / "column_meta.json") as f:
    meta = json.load(f)
STANDARD_COLS, ISEQL_COLS, ALL_COLS, CAT_COLS = (
    meta["STANDARD_COLS"], meta["ISEQL_COLS"], meta["ALL_COLS"], meta["CAT_COLS"]
)
model_c = C.load_artifact("model_c")
model_d = C.load_artifact("model_d")

# ---------------------------------------------------------------------------
# Sample 1000 validation rows for global SHAP (Model D)
# ---------------------------------------------------------------------------
sample_df = val_df.sample(n=min(1000, len(val_df)), random_state=C.RANDOM_STATE).reset_index(drop=True)
print(f"Sampled {len(sample_df)} validation rows for SHAP analysis "
      f"({sample_df[C.TARGET_COL].sum()} fraud, {(sample_df[C.TARGET_COL]==0).sum()} non-fraud)")

X_sample_d, cat_present_d = C.build_lgbm_matrix(sample_df, ALL_COLS, CAT_COLS)
explainer_d = shap.TreeExplainer(model_d)
shap_values_d = explainer_d.shap_values(X_sample_d)
if isinstance(shap_values_d, list):
    shap_values_d = shap_values_d[1]
print(f"SHAP values (Model D) computed: shape={shap_values_d.shape}")

# ---------------------------------------------------------------------------
# Global mean |SHAP| ranking
# ---------------------------------------------------------------------------
mean_abs_shap = np.abs(shap_values_d).mean(axis=0)
importance_df = pd.DataFrame({"feature": ALL_COLS, "mean_abs_shap": mean_abs_shap})
importance_df = importance_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
importance_df["rank"] = np.arange(1, len(importance_df) + 1)
importance_df["is_iseql"] = importance_df["feature"].isin(ISEQL_COLS)

print(f"\nFull global SHAP importance ranking (Model D, {len(importance_df)} features):")
print(importance_df[["rank", "feature", "mean_abs_shap", "is_iseql"]].to_string(index=False))

importance_path = C.OUT_DIR / "shap_feature_importance.csv"
importance_df.to_csv(importance_path, index=False)
print(f"\nSaved full SHAP importance ranking -> {importance_path}")

print(f"\nISEQL+ feature ranks (of {len(importance_df)} total features):")
n_total = len(importance_df)
for col in ISEQL_COLS:
    row = importance_df[importance_df["feature"] == col].iloc[0]
    print(f"  {col}: rank #{int(row['rank'])} of {n_total}, mean|SHAP|={row['mean_abs_shap']:.4f}")

# ---------------------------------------------------------------------------
# Global importance chart -- top 20, colored ISEQL+=red / standard=blue
# ---------------------------------------------------------------------------
top20 = importance_df.head(20).iloc[::-1]  # reverse for horizontal bar (largest on top)
colors = [RED if is_iseql else BLUE for is_iseql in top20["is_iseql"]]
hatches = [ISEQL_HATCH if is_iseql else "" for is_iseql in top20["is_iseql"]]

fig, ax = plt.subplots(figsize=(C.FIG_WIDTH, 6.2))
bars = ax.barh([display_label(f) for f in top20["feature"]], top20["mean_abs_shap"], color=colors, height=0.68)
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)
    bar.set_edgecolor(C.FIG_BG if not hatch else "#ffffff")
    bar.set_linewidth(0.0 if not hatch else 0.5)
for bar, val in zip(bars, top20["mean_abs_shap"]):
    ax.text(bar.get_width() + top20["mean_abs_shap"].max() * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=C.FS_ANNOT_SMALL, color=C.TEXT_COLOR)
ax.set_xlim(0, top20["mean_abs_shap"].max() * 1.16)
ax.set_xlabel("mean(|SHAP value|)")
subtitle = ("(orange + hatched = ISEQL+ pattern, blue = standard feature)"
            if top20["is_iseql"].any() else
            "(all top-20 are standard features; for the ISEQL+ patterns see "
            "shap_iseql_ranks)")
ax.set_title("Model D — top 20 features by global SHAP importance\n" + subtitle,
             fontsize=C.FS_TITLE, color=C.TEXT_COLOR, loc=C.TITLE_LOC)
C.style_axes(ax, grid_axis="x")
fig.tight_layout()
png_path, pdf_path = C.save_figure(fig, "shap_global_importance")
plt.close(fig)
print(f"\nSaved global SHAP importance chart -> {png_path} and {pdf_path}")

# ---------------------------------------------------------------------------
# Focused ISEQL+ ranking chart -- all 8 ISEQL+ features with their rank out of
# the full feature set. None of them reach the top-20 chart above, so this
# figure makes the ISEQL+ evidence directly visible without forcing the reader
# to cross-reference the full ranking table.
# ---------------------------------------------------------------------------
iseql_df = importance_df[importance_df["is_iseql"]].sort_values("rank")
iseql_plot = iseql_df.iloc[::-1]  # best rank ends up on top of a horizontal bar
labels = [f"{display_label(row.feature)} — rank {int(row.rank)}/{n_total}"
          for row in iseql_plot.itertuples()]

fig, ax = plt.subplots(figsize=(C.FIG_WIDTH, 3.6))
bars = ax.barh(labels, iseql_plot["mean_abs_shap"], color=RED, height=0.66,
               hatch=ISEQL_HATCH, edgecolor="#ffffff", linewidth=0.5)
for bar, val in zip(bars, iseql_plot["mean_abs_shap"]):
    ax.text(bar.get_width() + iseql_plot["mean_abs_shap"].max() * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=C.FS_ANNOT, color=C.TEXT_COLOR)
ax.set_xlim(0, iseql_plot["mean_abs_shap"].max() * 1.22)
ax.set_xlabel("mean(|SHAP value|)")
ax.set_title(f"Model D — the 8 ISEQL+ features, ranked among all {n_total} features",
             fontsize=C.FS_TITLE, color=C.TEXT_COLOR, loc=C.TITLE_LOC)
C.style_axes(ax, grid_axis="x")
fig.tight_layout()
png_path, pdf_path = C.save_figure(fig, "shap_iseql_ranks")
plt.close(fig)
print(f"Saved ISEQL+ rank chart -> {png_path} and {pdf_path}")

# ---------------------------------------------------------------------------
# Pick 5 fraud cases: highest predicted P(fraud) among true frauds in sample
# ---------------------------------------------------------------------------
feature_names = np.array(ALL_COLS)

pred_proba_d = model_d.predict_proba(X_sample_d)[:, 1]
sample_df = sample_df.copy()
sample_df["pred_proba_d"] = pred_proba_d
fraud_candidates = sample_df[sample_df[C.TARGET_COL] == 1].sort_values(
    "pred_proba_d", ascending=False
)

# Selection: 4 cases by highest predicted confidence (shows the general/typical
# pattern), plus -- if one exists in the sample -- 1 additional case where an
# ISEQL+ feature actually ranks in Model D's top-5 local SHAP drivers (an
# honest illustration of ISEQL+ impact; only 1 of 43 sampled frauds qualifies,
# so we surface it explicitly rather than silently omitting it or cherry-
# picking around the fact that anonymized V-block features dominate the very
# highest-confidence cases for both models).
def has_iseql_in_top5(pos, k=5):
    order = np.argsort(-np.abs(shap_values_d[pos]))[:k]
    return any(feature_names[j] in ISEQL_COLS for j in order)

iseql_driven = [pos for pos in fraud_candidates.index if has_iseql_in_top5(pos)]
top_conf_positions = fraud_candidates.index[:4].tolist()
illustrative_positions = [p for p in iseql_driven if p not in top_conf_positions][:1]
case_positions = top_conf_positions + illustrative_positions
is_illustrative = {p: False for p in top_conf_positions}
is_illustrative.update({p: True for p in illustrative_positions})
n_cases = len(case_positions)

print(f"\nSelected {len(top_conf_positions)} highest-confidence fraud cases + "
      f"{len(illustrative_positions)} ISEQL+-driven illustrative case "
      f"({len(iseql_driven)} of {len(fraud_candidates)} sampled frauds have an "
      f"ISEQL+ feature in Model D's top-5 local SHAP drivers):")

def waterfall_style_plot(shap_row, feature_names, title, basename, top_n=12):
    import textwrap
    order = np.argsort(-np.abs(shap_row))[:top_n]
    feats = feature_names[order][::-1]
    vals = shap_row[order][::-1]
    # Sign is encoded by colour AND by which side of zero the bar sits on, plus
    # the explicit +/- in every value label -- readable without colour.
    colors_local = [POS_RED if v > 0 else NEG_BLUE for v in vals]
    fig, ax = plt.subplots(figsize=(C.FIG_WIDTH, 4.6))
    bars = ax.barh([display_label(f) for f in feats], vals, color=colors_local, height=0.68)
    for bar, val in zip(bars, vals):
        offset = (abs(vals).max() * 0.03) * (1 if val >= 0 else -1)
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2, f"{val:+.3f}",
                va="center", ha="left" if val >= 0 else "right",
                fontsize=C.FS_ANNOT_SMALL, color=C.TEXT_COLOR)
    ax.axvline(0, color=C.ZERO_LINE_COLOR, linewidth=1.4, zorder=2)
    ax.tick_params(axis="y", pad=6)
    ax.set_xlim(*C.signed_bar_xlim(vals))
    ax.set_xlabel("SHAP value (impact on model output, log-odds)")
    wrapped_title = "\n".join(textwrap.wrap(title, width=64))
    ax.set_title(wrapped_title, fontsize=C.FS_TITLE, color=C.TEXT_COLOR, loc=C.TITLE_LOC)
    C.style_axes(ax, grid_axis="x")
    fig.tight_layout()
    paths = C.save_figure(fig, basename)
    plt.close(fig)
    return paths


case_summaries = []  # for step 6 reuse
for i, pos in enumerate(case_positions, start=1):
    tx_id = sample_df.loc[pos, C.ID_COL]
    proba = sample_df.loc[pos, "pred_proba_d"]
    shap_row = shap_values_d[pos]
    order = np.argsort(-np.abs(shap_row))
    top_feats = feature_names[order][:3]
    top_vals = shap_row[order][:3]

    tag = " [ISEQL+-driven illustrative case]" if is_illustrative[pos] else ""
    title = (f"Fraud case {i}{tag} — TransactionID {tx_id} "
             f"(Model D predicted P(fraud)={proba:.3f})")
    path, pdf_path = waterfall_style_plot(
        shap_row, feature_names, title, f"shap_waterfall_fraud_{i}")

    direction_words = [f"{f} ({'+' if v>0 else ''}{v:.3f})" for f, v in zip(top_feats, top_vals)]
    sentence = (
        f"Case {i}{tag} (TransactionID {tx_id}, predicted P(fraud)={proba:.3f}): "
        f"flagged mainly by {direction_words[0]}, {direction_words[1]}, and {direction_words[2]}."
    )
    print(f"  {sentence}")
    print(f"  Saved -> {path} and {pdf_path}")

    case_summaries.append({
        "case": i, "position": pos, "TransactionID": tx_id, "pred_proba_d": proba,
    })

# ---------------------------------------------------------------------------
# STEP 6 -- Interpretability comparison: Model C vs Model D, same 5 cases
# ---------------------------------------------------------------------------
print("\n=== STEP 6: Interpretability comparison (Model C vs Model D) ===")
X_sample_c, cat_present_c = C.build_lgbm_matrix(sample_df.drop(columns=["pred_proba_d"]),
                                                 STANDARD_COLS, CAT_COLS)
explainer_c = shap.TreeExplainer(model_c)
shap_values_c = explainer_c.shap_values(X_sample_c)
if isinstance(shap_values_c, list):
    shap_values_c = shap_values_c[1]
standard_feature_names = np.array(STANDARD_COLS)

md_lines = [
    "# Interpretability comparison -- Model C (standard) vs Model D (+ISEQL+)",
    "",
    "For 5 fraud cases from the same 1000-row validation sample used in Step 5 "
    "(the 4 highest-confidence true frauds, plus 1 case chosen because an "
    "ISEQL+ feature ranks in Model D's top-5 local SHAP drivers there), the "
    "top-5 SHAP features are shown for each model. Model C's top features are "
    "frequently anonymized `V*`/`C*` Vesta engineering columns with no "
    "disclosed semantic meaning; Model D's top features are, wherever an "
    "ISEQL+ pattern ranks highly, human-readable and traceable to a named "
    "regulatory source (PCI DSS, FATF, EBA PSD2, ACFE).",
    "",
    "**Honest caveat:** only 1 of the 43 sampled fraud cases had an ISEQL+ "
    "feature in Model D's top-5 local drivers -- for the very highest-"
    "confidence frauds, anonymized Vesta engineering features (especially "
    "`V258`) dominate for both models, since these are cases already caught "
    "by strong pre-existing signals. ISEQL+ features rank respectably in the "
    "*global* importance (see Step 5: `structuring_count_day` #30, "
    "`tx_count_24h` #32, `tx_count_1h` #58, `device_novelty_flag` #71 of 444) "
    "but are rarely the single dominant driver for the most extreme cases -- "
    "their contribution is broader-based across many more borderline "
    "transactions than the handful of highest-confidence ones.",
    "",
]

for i, pos in enumerate(case_positions, start=1):
    tx_id = sample_df.loc[pos, C.ID_COL]

    order_c = np.argsort(-np.abs(shap_values_c[pos]))[:5]
    order_d = np.argsort(-np.abs(shap_values_d[pos]))[:5]
    top_c = [(standard_feature_names[j], shap_values_c[pos][j]) for j in order_c]
    top_d = [(feature_names[j], shap_values_d[pos][j]) for j in order_d]

    print(f"\nCase {i} (TransactionID {tx_id}):")
    print(f"  {'Model C (standard) top 5':38s} | {'Model D (+ISEQL+) top 5'}")
    for (fc, vc), (fd, vd) in zip(top_c, top_d):
        d_marker = " [ISEQL+]" if fd in ISEQL_COLS else ""
        print(f"  {fc:28s} {vc:+.3f}      | {fd:28s} {vd:+.3f}{d_marker}")

    case_tag = " (ISEQL+-driven illustrative case)" if is_illustrative[pos] else ""
    md_lines.append(f"## Case {i}{case_tag} -- TransactionID {tx_id}")
    md_lines.append("")
    md_lines.append("| Rank | Model C feature (standard) | SHAP | Model D feature (+ISEQL+) | SHAP |")
    md_lines.append("|---|---|---|---|---|")
    for rank_idx, ((fc, vc), (fd, vd)) in enumerate(zip(top_c, top_d), start=1):
        fd_label = f"**{fd}** (ISEQL+)" if fd in ISEQL_COLS else fd
        md_lines.append(f"| {rank_idx} | `{fc}` | {vc:+.3f} | {fd_label} | {vd:+.3f} |")
    md_lines.append("")

# ---------------------------------------------------------------------------
# Side-by-side comparison figure for the illustrative case: Model C's top-5
# local SHAP drivers vs Model D's, on a shared x-axis so bar lengths are
# directly comparable. The ISEQL+ feature that enters Model D's explanation is
# highlighted (distinct colour AND hatch AND a "(ISEQL+)" suffix on its label).
# ---------------------------------------------------------------------------
cmp_pos = illustrative_positions[0] if illustrative_positions else case_positions[-1]
cmp_case_no = case_positions.index(cmp_pos) + 1
cmp_tx_id = sample_df.loc[cmp_pos, C.ID_COL]
HIGHLIGHT_FEATURE = "structuring_count_day"

def top_k_drivers(shap_matrix, names, pos, k=5):
    order = np.argsort(-np.abs(shap_matrix[pos]))[:k]
    return [(names[j], float(shap_matrix[pos][j])) for j in order][::-1]  # smallest at bottom

drivers_c = top_k_drivers(shap_values_c, standard_feature_names, cmp_pos)
drivers_d = top_k_drivers(shap_values_d, feature_names, cmp_pos)

all_vals = [v for _, v in drivers_c + drivers_d]
mag = max(abs(min(all_vals)), abs(max(all_vals)))
# wider pad than the default: this figure's labels are drawn at FS_SCALE, so
# they need proportionally more room outside the bar ends
shared_xlim = C.signed_bar_xlim(all_vals, pad_frac=0.62)

# This is a full-text-width two-panel figure (11in) rather than a single-column
# one, so its fonts are scaled by the same factor the page shrink applies --
# that way the PRINTED text size matches every other figure in the thesis.
FS_SCALE = 11.0 / 6.3

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, (panel_title, drivers) in zip(
    axes,
    [("Model C (standard features only)", drivers_c),
     ("Model D (standard + ISEQL+)", drivers_d)],
):
    labels, vals = [f for f, _ in drivers], [v for _, v in drivers]
    labels = [f"{f} (ISEQL+)" if f == HIGHLIGHT_FEATURE else display_label(f) for f in labels]
    colors_local, hatches_local = [], []
    for (feat, val) in drivers:
        if feat == HIGHLIGHT_FEATURE:
            colors_local.append(C.COL_HIGHLIGHT)
            hatches_local.append(ISEQL_HATCH)
        else:
            colors_local.append(POS_RED if val > 0 else NEG_BLUE)
            hatches_local.append("")
    bars = ax.barh(labels, vals, color=colors_local, height=0.66)
    for bar, hatch in zip(bars, hatches_local):
        if hatch:
            bar.set_hatch(hatch)
            bar.set_edgecolor("#ffffff")
            bar.set_linewidth(0.6)
    for bar, val in zip(bars, vals):
        offset = mag * 0.03 * (1 if val >= 0 else -1)
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2, f"{val:+.3f}",
                va="center", ha="left" if val >= 0 else "right",
                fontsize=C.FS_ANNOT * FS_SCALE, color=C.TEXT_COLOR)
    ax.axvline(0, color=C.ZERO_LINE_COLOR, linewidth=1.4, zorder=2)
    ax.set_xlim(*shared_xlim)  # identical scale on both panels
    ax.set_title(panel_title, fontsize=C.FS_LABEL * FS_SCALE, color=C.TEXT_COLOR,
                 loc=C.TITLE_LOC)
    C.style_axes(ax, grid_axis="x")
    ax.tick_params(labelsize=C.FS_TICK * FS_SCALE, pad=6)

fig.supxlabel("SHAP value (impact on model output, log-odds)",
              fontsize=C.FS_LABEL * FS_SCALE, color=C.LABEL_COLOR, y=0.02)
fig.suptitle(f"Case {cmp_case_no} — TransactionID {cmp_tx_id}: top-5 SHAP drivers, "
             f"Model C vs Model D (shared scale)",
             fontsize=C.FS_TITLE * FS_SCALE, color=C.TEXT_COLOR, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.02, 1, 0.93))
png_path, pdf_path = C.save_figure(fig, "shap_case5_comparison")
plt.close(fig)
print(f"\nSaved Model C vs D case comparison (case {cmp_case_no}, "
      f"TransactionID {cmp_tx_id}) -> {png_path} and {pdf_path}")

interp_path = C.OUT_DIR / "interpretability_comparison.md"
with open(interp_path, "w") as f:
    f.write("\n".join(md_lines))
print(f"\nSaved interpretability comparison -> {interp_path}")

print(f"\nTotal step 5+6 runtime: {time.time()-t0:.1f}s")
print("\nSTEPS 5-6 complete.")
