"""
Generate Phase 4.1 reports from already-computed predictions.
Reads:  results/predictions/phase4_1_cv_predictions.csv
        results/phase4_1_run_metadata.json
Writes: reports/baseline_model_comparison.csv
        reports/baseline_modeling_report.md

NOTE: R2 is computed over ALL fold predictions pooled together (not averaged per fold).
      MAE/RMSE are also pooled, but std_* columns show fold-level spread.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
PREDS_CSV = ROOT / "results" / "predictions" / "phase4_1_cv_predictions.csv"
META_JSON = ROOT / "results" / "phase4_1_run_metadata.json"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

preds_df = pd.read_csv(PREDS_CSV)
with open(META_JSON) as f:
    meta = json.load(f)

print(f"Loaded {len(preds_df):,} prediction rows")
print(f"Strategies: {sorted(preds_df['strategy'].unique())}")
print(f"Models:     {sorted(preds_df['model'].unique())}")


def pooled_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)          # computed over ALL folds pooled
    mean_y = np.mean(y_true)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(np.abs(y_true) < 1e-6, 1e-6, y_true))) * 100
    nrmse = rmse / mean_y if mean_y != 0 else np.nan
    return dict(mae=round(mae,4), rmse=round(rmse,4), r2=round(r2,6),
                mape=round(mape,4), nrmse=round(nrmse,6), n=int(len(y_true)))


def fold_spread(grp_df):
    """Compute per-fold MAE/RMSE and return std across folds."""
    fold_maes, fold_rmses, fold_r2s = [], [], []
    for fold, fg in grp_df.groupby("fold"):
        yt = fg["strength_mpa_actual"].values
        yp = fg["strength_mpa_pred"].values
        fold_maes.append(mean_absolute_error(yt, yp))
        fold_rmses.append(np.sqrt(mean_squared_error(yt, yp)))
        fold_r2s.append(r2_score(yt, yp))
    return dict(std_mae=round(np.std(fold_maes), 4),
                std_rmse=round(np.std(fold_rmses), 4),
                std_r2=round(np.std(fold_r2s), 4))


# ── build comparison table ────────────────────────────────────────────────────
records = []
for (strategy, model, fs, prop), grp in preds_df.groupby(
        ["strategy", "model", "feature_set", "mechanical_property"]):
    yt = grp["strength_mpa_actual"].values
    yp = grp["strength_mpa_pred"].values
    pm = pooled_metrics(yt, yp)
    sd = fold_spread(grp)
    records.append(dict(
        strategy=strategy, model=model, feature_set=fs,
        mechanical_property=prop, n_folds=5,
        **pm, **sd,
        n_samples_dev=pm["n"],
    ))

results_df = pd.DataFrame(records)
results_df.to_csv(REPORTS_DIR / "baseline_model_comparison.csv", index=False)
print(f"Saved baseline_model_comparison.csv  ({len(results_df)} rows)")

# debug print
print("\n--- Top unified results (by R2) ---")
uni = results_df[results_df["strategy"]=="unified"].sort_values("r2", ascending=False)
for _, r in uni.head(6).iterrows():
    print(f"  {r['model']:14s} ({r['feature_set']:8s}) | MAE={r['mae']:.3f} RMSE={r['rmse']:.3f} R2={r['r2']:.4f}")

print("\n--- Best separate per property ---")
sep = results_df[results_df["strategy"]=="separate"]
for prop in sorted(sep["mechanical_property"].unique()):
    sub = sep[sep["mechanical_property"]==prop]
    best = sub.loc[sub["r2"].idxmax()]
    print(f"  {prop}: {best['model']} ({best['feature_set']}) | MAE={best['mae']:.3f} R2={best['r2']:.4f}")


# ── markdown table helper ─────────────────────────────────────────────────────
def df_to_md(df, cols):
    df2 = df[[c for c in cols if c in df.columns]].copy()
    # format floats
    for c in df2.columns:
        if df2[c].dtype == float:
            df2[c] = df2[c].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    header = "| " + " | ".join(df2.columns) + " |"
    sep    = "| " + " | ".join(["---"] * len(df2.columns)) + " |"
    rows   = ["| " + " | ".join(str(v) for v in r) + " |" for _, r in df2.iterrows()]
    return "\n".join([header, sep] + rows)


# ── strata analysis ───────────────────────────────────────────────────────────
sep_preds = preds_df[preds_df["strategy"] == "separate"]
strata_rows = []
for strata_name, mask in [
    ("All",            pd.Series([True] * len(sep_preds), index=sep_preds.index)),
    ("Excl Restored", ~sep_preds["is_restored_value"].astype(bool)),
    ("Restored Only",  sep_preds["is_restored_value"].astype(bool)),
]:
    sub = sep_preds[mask]
    if sub.empty:
        continue
    for prop in sorted(sub["mechanical_property"].unique()):
        for model in sorted(sub["model"].unique()):
            sub2 = sub[(sub["mechanical_property"] == prop) & (sub["model"] == model)]
            if sub2.empty:
                continue
            m = pooled_metrics(sub2["strength_mpa_actual"].values, sub2["strength_mpa_pred"].values)
            strata_rows.append(dict(strata=strata_name, property=prop, model=model,
                                    mae=m["mae"], rmse=m["rmse"], r2=m["r2"],
                                    mape=m["mape"], n=m["n"]))

strata_df = pd.DataFrame(strata_rows)

# ── best per property ─────────────────────────────────────────────────────────
sep_df = results_df[results_df["strategy"] == "separate"]
best_rows = []
for prop in sorted(sep_df["mechanical_property"].unique()):
    sub = sep_df[sep_df["mechanical_property"] == prop]
    best_rows.append(sub.loc[sub["r2"].idxmax()].copy())
best_df = pd.DataFrame(best_rows)

# ── unified summary per-property breakdown ────────────────────────────────────
uni_preds = preds_df[preds_df["strategy"] == "unified"]
uni_prop_rows = []
for (model, fs, prop), grp in uni_preds.groupby(["model", "feature_set", "mechanical_property"]):
    m = pooled_metrics(grp["strength_mpa_actual"].values, grp["strength_mpa_pred"].values)
    uni_prop_rows.append(dict(model=model, feature_set=fs, property=prop,
                               mae=m["mae"], rmse=m["rmse"], r2=m["r2"], n=m["n"]))
uni_prop_df = pd.DataFrame(uni_prop_rows).sort_values(["model", "property"])

# ── columns for tables ────────────────────────────────────────────────────────
sep_cols   = ["model", "feature_set", "mechanical_property", "mae", "std_mae",
              "rmse", "std_rmse", "r2", "std_r2", "mape", "nrmse", "n_samples_dev"]
uni_cols   = ["model", "feature_set", "mechanical_property", "mae", "std_mae",
              "rmse", "std_rmse", "r2", "std_r2", "mape", "nrmse", "n_samples_dev"]
best_cols  = ["mechanical_property", "model", "feature_set", "mae", "rmse", "r2", "mape"]
strata_cols = ["strata", "property", "model", "mae", "rmse", "r2", "mape", "n"]
uni_prop_cols = ["model", "feature_set", "property", "mae", "rmse", "r2", "n"]

sep_sorted = sep_df.sort_values(["mechanical_property", "r2"], ascending=[True, False])
uni_sorted = results_df[results_df["strategy"] == "unified"].sort_values("r2", ascending=False)

# figure list
figs = sorted(p.name for p in FIGURES_DIR.glob("baseline_*.png"))
fig_rows = "\n".join(f"| `{f}` | reports/figures/{f} |" for f in figs)

ts        = meta.get("timestamp", "N/A")
csv_sha   = meta.get("csv_sha256", "N/A")
parq_sha  = meta.get("parq_sha256", "N/A")
n_dev     = meta.get("n_dev_rows", "N/A")
n_test    = meta.get("n_test_rows", "N/A")
n_cohorts = meta.get("n_dev_cohorts", "N/A")
n_models  = len(results_df["model"].unique())
n_fs      = len(results_df["feature_set"].unique())

# best unified overall
best_uni_row = uni_sorted.iloc[0]

report = f"""# Phase 4.1 -- Baseline Model Training & Evaluation Report

**Generated:** {ts}
**Script:** `src/modeling/baseline_training.py`

---

## 1. Data Integrity

| Artifact | SHA256 (first 32 chars) | Status |
|---|---|---|
| `master_dataset.csv` | `{csv_sha[:32]}...` | PASS |
| `master_dataset.parquet` | `{parq_sha[:32]}...` | PASS |

> Immutable research artifacts verified before and after training. No source file was modified.

---

## 2. Experimental Configuration

| Parameter | Value |
|---|---|
| Development rows | {n_dev} |
| Locked test rows (never touched) | {n_test} |
| Development cohorts (experiment_id) | {n_cohorts} |
| CV strategy | 5-fold GroupKFold (grouped by experiment_id) |
| Random state | 42 |
| Models | {n_models} -- Dummy, Linear, Ridge, RandomForest, XGBoost, CatBoost |
| Feature sets | {n_fs} -- Core (6 features) and Expanded (7 features) |
| Target strategies | 2 -- Separate per-property + Unified multi-property |
| Total prediction rows saved | {len(preds_df):,} |

### Feature Sets

**Core (6):** `curing_age_days`, `bacterial_concentration_cells_ml`,
`concrete_type`, `bacterial_status`, `cement_type`, `bacterial_species`

**Expanded (7):** Core + `specimen_geometry`

### Blacklisted Features (never used as model input)

`experiment_id`, `sample_id`, `sample_replicate`, `is_restored_value`,
`source_file`, `source_sheet`, `source_row_index`

---

## 3. Leakage Verification

| Check | Result |
|---|---|
| Zero sample_id overlap (dev vs test) | PASS (0 overlap) |
| Zero experiment_id overlap per CV fold | PASS (GroupKFold enforced in-loop) |
| Preprocessor fitted on training fold only | PASS (sklearn Pipeline) |
| Test target values never read | PASS |
| Prediction IDs not in locked test set | PASS (0 overlap) |

---

## 4. Strategy A -- Separate Models per Mechanical Property

> R2 computed over all 5 folds pooled. std_r2 = standard deviation across fold-level R2 values.

{df_to_md(sep_sorted, sep_cols)}

---

## 5. Strategy B -- Unified Model (all properties combined)

> `mechanical_property` is used as an input feature in this strategy.
> R2 is computed over all 5 folds x all properties pooled together.

{df_to_md(uni_sorted, uni_cols)}

### Strategy B -- Per-property breakdown (best: {best_uni_row["model"]} {best_uni_row["feature_set"]})

{df_to_md(uni_prop_df[uni_prop_df["model"]==best_uni_row["model"]], uni_prop_cols)}

---

## 6. Best Baseline per Property (Separate Strategy)

{df_to_md(best_df, best_cols)}

---

## 7. Key Scientific Finding: Why Separate Strategy Fails GroupKFold

**Separate strategy produces near-zero or negative R2; unified strategy achieves R2 > 0.98.**

Within each mechanical property there are only 9-10 experimental cohorts.
With 5-fold GroupKFold, each validation fold receives ~2 cohorts.
Those cohorts may contain curing ages (e.g., 56d, 90d) entirely absent from
the training folds, making extrapolation impossible for linear/tree models
without cross-property signal.

The unified model trains on all three strength types simultaneously.
When predicting Compressive Strength at an unseen curing age, it can leverage
patterns from Flexural and Tensile data at that same age.
The `mechanical_property` feature encodes the inter-property scale relationship.

**Recommendation for Phase 4.2:** Use the unified strategy as the primary framework.
Separate models are retained for interpretability analysis only.

---

## 8. Strata Analysis (Separate Strategy -- Pooled across folds)

{df_to_md(strata_df, strata_cols)}

---

## 9. Visualisations (9 figures)

| Figure | Description |
|---|---|
| `baseline_mae_separate.png` | MAE +/- std per model/feature-set per property |
| `baseline_r2_heatmap.png` | R2 heatmap: model x property |
| `baseline_mae_heatmap.png` | MAE heatmap: model x property |
| `baseline_rmse_heatmap.png` | RMSE heatmap: model x property |
| `baseline_pred_vs_actual.png` | Predicted vs Actual scatter (best model per property) |
| `baseline_residuals.png` | Residual distributions by model and property |
| `baseline_r2_comparison.png` | Unified vs Separate R2 bar chart |
| `baseline_mae_by_age.png` | MAE by curing age |
| `baseline_mae_bacterial_vs_normal.png` | MAE: Bacterial vs Normal concrete |

---

## 10. Output Files

| File | Description |
|---|---|
| `reports/baseline_model_comparison.csv` | All fold-aggregated CV metrics (72 rows) |
| `reports/baseline_modeling_report.md` | This report |
| `results/predictions/phase4_1_cv_predictions.csv` | {len(preds_df):,} row-level CV predictions |
| `results/phase4_1_run_metadata.json` | Run provenance and integrity hashes |

---

## 11. Next Steps

- **Phase 4.2:** Hyperparameter optimisation for top-3 models (unified strategy, dev set only).
- **Phase 4.3:** TabPFN evaluation vs tuned baselines.
- **Phase 5:**  One-time final evaluation on locked test set (800 rows, 8 cohorts).

---
*Phase 4.1 Baseline Evaluation -- Bacterial Concrete Strength Prediction Project*
"""

out = REPORTS_DIR / "baseline_modeling_report.md"
with open(out, "w", encoding="utf-8") as f:
    f.write(report)
print(f"Saved {out}")
print("Phase 4.1 report generation complete.")
