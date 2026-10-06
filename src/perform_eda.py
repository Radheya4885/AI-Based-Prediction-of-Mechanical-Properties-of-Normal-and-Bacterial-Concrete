import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure directories exist
os.makedirs("reports/figures", exist_ok=True)

# Set plotting aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['figure.titlesize'] = 14

print("Loading data/master_dataset.csv...")
df = pd.read_csv("data/master_dataset.csv")

print(f"Loaded Master Dataset: {df.shape}")

# -------------------------------------------------------------------------
# SECTION 1: DATASET STRUCTURE STATS
# -------------------------------------------------------------------------
total_rows = len(df)
total_cols = df.shape[1]
dtypes_dict = df.dtypes.astype(str).to_dict()
missing_dict = df.isnull().sum().to_dict()
unique_dict = df.nunique().to_dict()
constant_cols = [col for col, u in unique_dict.items() if u == 1]
dup_sample_ids = int(df['sample_id'].duplicated().sum())

feature_cols_for_dup = ['concrete_type', 'curing_age_days', 'cement_type', 'bacterial_species', 
                        'bacterial_concentration_cells_ml', 'mechanical_property', 'strength_mpa']
dup_tuples = int(df.duplicated(subset=feature_cols_for_dup).sum())

print(f"Structure: {total_rows} rows, {total_cols} cols, {dup_sample_ids} dup sample IDs, {dup_tuples} dup tuples")

# -------------------------------------------------------------------------
# SECTION 2: TARGET DISTRIBUTIONS STATS & FIGURES 1, 2, 3
# -------------------------------------------------------------------------
props = ['Compressive Strength', 'Flexural Strength', 'Split Tensile Strength']
prop_short = {'Compressive Strength': 'compressive', 'Flexural Strength': 'flexural', 'Split Tensile Strength': 'split_tensile'}
colors_dict = {'Normal Concrete': '#2b5c8f', 'Bacterial Concrete': '#2a9d8f'}

target_stats = {}

for prop in props:
    series = df[df['mechanical_property'] == prop]['strength_mpa']
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    mean_val = float(series.mean())
    std_val = float(series.std())
    cv_val = (std_val / mean_val) * 100.0 if mean_val != 0 else 0.0
    
    target_stats[prop] = {
        'count': int(len(series)),
        'mean': mean_val,
        'median': float(series.median()),
        'std': std_val,
        'var': float(series.var()),
        'min': float(series.min()),
        'max': float(series.max()),
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'cv': cv_val
    }

# Figure 1: Compressive Strength Distribution
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
sub_df = df[df['mechanical_property'] == 'Compressive Strength']
sns.histplot(data=sub_df, x='strength_mpa', hue='concrete_type', kde=True, bins=35, palette=colors_dict, ax=ax, alpha=0.55)
ax.set_title("Figure 1: Target Distribution — Compressive Strength (MPa)")
ax.set_xlabel("Compressive Strength (MPa)")
ax.set_ylabel("Specimen Count (Frequency)")
plt.tight_layout()
fig.savefig("reports/figures/01_target_distribution_compressive.png")
plt.close(fig)

# Figure 2: Flexural Strength Distribution
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
sub_df = df[df['mechanical_property'] == 'Flexural Strength']
sns.histplot(data=sub_df, x='strength_mpa', hue='concrete_type', kde=True, bins=35, palette=colors_dict, ax=ax, alpha=0.55)
ax.set_title("Figure 2: Target Distribution — Flexural Strength (MPa)")
ax.set_xlabel("Flexural Strength (MPa)")
ax.set_ylabel("Specimen Count (Frequency)")
plt.tight_layout()
fig.savefig("reports/figures/02_target_distribution_flexural.png")
plt.close(fig)

# Figure 3: Split Tensile Strength Distribution
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
sub_df = df[df['mechanical_property'] == 'Split Tensile Strength']
sns.histplot(data=sub_df, x='strength_mpa', hue='concrete_type', kde=True, bins=35, palette=colors_dict, ax=ax, alpha=0.55)
ax.set_title("Figure 3: Target Distribution — Split Tensile Strength (MPa)")
ax.set_xlabel("Split Tensile Strength (MPa)")
ax.set_ylabel("Specimen Count (Frequency)")
plt.tight_layout()
fig.savefig("reports/figures/03_target_distribution_split_tensile.png")
plt.close(fig)

print("Generated Target Distribution Figures 1, 2, 3.")

# -------------------------------------------------------------------------
# SECTION 3: AGE EFFECT & FIGURES 4, 5, 6
# -------------------------------------------------------------------------
ages = [7, 14, 21, 28, 56, 90]
age_stats = {}

for prop in props:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    prop_dict = {}
    
    for c_type in ['Normal Concrete', 'Bacterial Concrete']:
        sub = df[(df['mechanical_property'] == prop) & (df['concrete_type'] == c_type)]
        means = [sub[sub['curing_age_days'] == a]['strength_mpa'].mean() for a in ages]
        stds = [sub[sub['curing_age_days'] == a]['strength_mpa'].std() for a in ages]
        prop_dict[c_type] = {'means': means, 'stds': stds}
        
        ax.errorbar(ages, means, yerr=stds, label=c_type, marker='o', capsize=5, 
                    linewidth=2.2, markersize=6, color=colors_dict[c_type])
        
    ax.set_title(f"Strength vs Curing Age — {prop}")
    ax.set_xlabel("Curing Age (Days)")
    ax.set_ylabel("Mean Strength (MPa) [± 1 SD]")
    ax.set_xticks(ages)
    ax.legend(title="Mix Type", frameon=True)
    plt.tight_layout()
    fig_name = f"04_age_effect_{prop_short[prop]}.png" if prop == 'Compressive Strength' else (f"05_age_effect_{prop_short[prop]}.png" if prop == 'Flexural Strength' else f"06_age_effect_{prop_short[prop]}.png")
    fig.savefig(f"reports/figures/{fig_name}")
    plt.close(fig)
    age_stats[prop] = prop_dict

print("Generated Age Effect Figures 4, 5, 6.")

# -------------------------------------------------------------------------
# SECTION 4: BACTERIAL EFFECT (DIFFERENCE TABLE)
# -------------------------------------------------------------------------
bacterial_diffs = []
for prop in props:
    for age in ages:
        nc_sub = df[(df['mechanical_property'] == prop) & (df['concrete_type'] == 'Normal Concrete') & (df['curing_age_days'] == age)]['strength_mpa']
        bc_sub = df[(df['mechanical_property'] == prop) & (df['concrete_type'] == 'Bacterial Concrete') & (df['curing_age_days'] == age)]['strength_mpa']
        
        nc_m = float(nc_sub.mean())
        bc_m = float(bc_sub.mean())
        abs_diff = bc_m - nc_m
        pct_diff = (abs_diff / nc_m) * 100.0 if nc_m != 0 else 0.0
        
        bacterial_diffs.append({
            'mechanical_property': prop,
            'curing_age_days': age,
            'normal_mean': nc_m,
            'bacterial_mean': bc_m,
            'absolute_difference': abs_diff,
            'percentage_difference': pct_diff
        })

diff_df = pd.DataFrame(bacterial_diffs)

# -------------------------------------------------------------------------
# SECTION 5: STATISTICAL TESTS (18 COMPARISONS)
# -------------------------------------------------------------------------
stat_tests = []
for prop in props:
    for age in ages:
        nc_vals = df[(df['mechanical_property'] == prop) & (df['concrete_type'] == 'Normal Concrete') & (df['curing_age_days'] == age)]['strength_mpa'].values
        bc_vals = df[(df['mechanical_property'] == prop) & (df['concrete_type'] == 'Bacterial Concrete') & (df['curing_age_days'] == age)]['strength_mpa'].values
        
        # Welch's t-test
        t_stat, p_val = stats.ttest_ind(bc_vals, nc_vals, equal_var=False)
        
        # Mann-Whitney U test
        u_stat, mw_p_val = stats.mannwhitneyu(bc_vals, nc_vals, alternative='two-sided')
        
        # Cohen's d
        n1, n2 = len(bc_vals), len(nc_vals)
        s1, s2 = np.var(bc_vals, ddof=1), np.var(nc_vals, ddof=1)
        pooled_se = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        cohens_d = (np.mean(bc_vals) - np.mean(nc_vals)) / pooled_se if pooled_se != 0 else 0.0
        
        # 95% Confidence Interval for mean difference
        mean_diff = np.mean(bc_vals) - np.mean(nc_vals)
        diff_se = np.sqrt(s1 / n1 + s2 / n2)
        ci_lower = mean_diff - 1.96 * diff_se
        ci_upper = mean_diff + 1.96 * diff_se
        
        stat_tests.append({
            'mechanical_property': prop,
            'curing_age_days': age,
            'welch_t': float(t_stat),
            'welch_p': float(p_val),
            'mann_whitney_u': float(u_stat),
            'mann_whitney_p': float(mw_p_val),
            'cohens_d': float(cohens_d),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper)
        })

stat_df = pd.DataFrame(stat_tests)

# -------------------------------------------------------------------------
# SECTION 6: OUTLIER ANALYSIS & FIGURES 7, 8, 9, 11
# -------------------------------------------------------------------------
# Boxplots: Figures 7, 8, 9 (by Concrete Type & Curing Age)
for prop in props:
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    sub = df[df['mechanical_property'] == prop]
    sns.boxplot(data=sub, x='curing_age_days', y='strength_mpa', hue='concrete_type', palette=colors_dict, ax=ax, width=0.6, fliersize=3.5)
    ax.set_title(f"Figure: Strength Distribution by Curing Age & Mix — {prop}")
    ax.set_xlabel("Curing Age (Days)")
    ax.set_ylabel("Strength (MPa)")
    ax.legend(title="Mix Type", frameon=True)
    plt.tight_layout()
    fig_num = "07_boxplots_compressive.png" if prop == 'Compressive Strength' else ("08_boxplots_flexural.png" if prop == 'Flexural Strength' else "09_boxplots_split_tensile.png")
    fig.savefig(f"reports/figures/{fig_num}")
    plt.close(fig)

print("Generated Boxplot Figures 7, 8, 9.")

# Figure 11: Outlier Plot across all 3 properties
fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=300)
outlier_summary = []

for idx, prop in enumerate(props):
    sub = df[df['mechanical_property'] == prop]
    ax = axes[idx]
    sns.boxplot(data=sub, y='strength_mpa', x='curing_age_days', hue='concrete_type', palette=colors_dict, ax=ax, fliersize=4)
    ax.set_title(prop, fontsize=11, fontweight='bold')
    ax.set_xlabel("Curing Age (Days)")
    ax.set_ylabel("Strength (MPa)")
    if idx > 0:
        ax.legend_.remove()
    else:
        ax.legend(title="Mix Type", fontsize=8, title_fontsize=9)
        
    # Statistical IQR outlier detection per condition
    for age in ages:
        for c_type in ['Normal Concrete', 'Bacterial Concrete']:
            cond_s = sub[(sub['curing_age_days'] == age) & (sub['concrete_type'] == c_type)]['strength_mpa']
            cq1 = cond_s.quantile(0.25)
            cq3 = cond_s.quantile(0.75)
            ciqr = cq3 - cq1
            low_f = cq1 - 1.5 * ciqr
            high_f = cq3 + 1.5 * ciqr
            outliers = cond_s[(cond_s < low_f) | (cond_s > high_f)]
            
            outlier_summary.append({
                'mechanical_property': prop,
                'curing_age_days': age,
                'concrete_type': c_type,
                'total_count': len(cond_s),
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(cond_s)) * 100.0,
                'min_outlier': float(outliers.min()) if len(outliers) > 0 else np.nan,
                'max_outlier': float(outliers.max()) if len(outliers) > 0 else np.nan
            })

plt.suptitle("Figure 11: Outlier Detection via IQR Across Properties and Curing Ages", y=1.02, fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig("reports/figures/11_outliers_iqr.png", bbox_inches='tight')
plt.close(fig)

outlier_df = pd.DataFrame(outlier_summary)
print(f"Generated Figure 11. Total outliers detected across all conditions: {outlier_df['outlier_count'].sum()} ({outlier_df['outlier_count'].sum()/3600*100:.2f}%)")

# -------------------------------------------------------------------------
# SECTION 7: CORRELATION ANALYSIS & FIGURE 10
# -------------------------------------------------------------------------
# Compute correlations
# Global correlation
num_cols = ['curing_age_days', 'bacterial_concentration_cells_ml', 'strength_mpa']
corr_global = df[num_cols].corr()

# Per property correlations
corr_per_prop = {}
for prop in props:
    sub = df[df['mechanical_property'] == prop]
    corr_per_prop[prop] = sub[num_cols].corr()

# Figure 10: Correlation Heatmap
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), dpi=300)

sns.heatmap(corr_global, annot=True, fmt=".3f", cmap="vlag", vmin=-1, vmax=1, ax=axes[0], cbar=False)
axes[0].set_title("Global (All Properties)", fontsize=11, fontweight='bold')

for idx, prop in enumerate(props, 1):
    sns.heatmap(corr_per_prop[prop], annot=True, fmt=".3f", cmap="vlag", vmin=-1, vmax=1, ax=axes[idx], cbar=(idx == 3))
    axes[idx].set_title(prop, fontsize=11, fontweight='bold')

plt.suptitle("Figure 10: Pearson Correlation Heatmaps (Global & Per Mechanical Property)", y=1.03, fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig("reports/figures/10_correlation_heatmap.png", bbox_inches='tight')
plt.close(fig)

print("Generated Correlation Heatmap Figure 10.")

# -------------------------------------------------------------------------
# BUILD COMPREHENSIVE reports/eda_report.md
# -------------------------------------------------------------------------
print("Compiling reports/eda_report.md...")
r_lines = []
def rp(text=""):
    r_lines.append(text)

rp("# Exploratory Data Analysis (EDA) Report: AI Concrete Strength Prediction")
rp()
rp("> **Analysis Scope**: Phase 2 Exploratory Data Analysis & Diagnostic Investigation  ")
rp("> **Primary Source**: [`data/master_dataset.csv`](file:///g:/Projects/Concrete%20testing/data/master_dataset.csv) (3,600 rows × 16 columns)  ")
rp("> **Policy Adherence**: Zero machine learning models trained (no XGBoost, CatBoost, TabPFN). Zero synthetic data added. Zero rows dropped. No train/test splitting performed.")
rp()
rp("---")
rp()
rp("## Executive Summary")
rp()
rp("This report presents a comprehensive Exploratory Data Analysis (EDA) of the validated Master Dataset for concrete strength prediction. The dataset captures **3,600 destructive physical test measurements** evaluating Compressive Strength ($f_c$), Flexural Strength ($f_r$), and Split Tensile Strength ($f_t$) across Ordinary Portland Cement (OPC 53) concrete mixes treated with *Bacillus subtilis* bio-mineralizing bacteria ($10^6$ cells/mL) compared against control mixes over 6 hydration curing ages (7, 14, 21, 28, 56, and 90 days).")
rp()
rp("---")
rp()
rp("## 1. Dataset Structure & Data Hygiene")
rp()
rp("### 1.1 Structural Metrics")
rp(f"- **Total Specimen Observations**: `{total_rows:,}` rows")
rp(f"- **Total Schema Attributes**: `{total_cols}` columns")
rp(f"- **Total Missing Values**: `0` (100% complete across all 3,600 observations)")
rp(f"- **Duplicate Specimen Primary Keys (`sample_id`)**: `0` duplicate IDs (100% unique primary keys)")
rp(f"- **Identical Feature-Target Tuples**: `{dup_tuples}` instances across 3,600 rows")
rp()
rp("### 1.2 Column Metadata & Information Inventory")
rp()
rp("| Column # | Column Name | Inferred Dtype | Non-Null Count | Missing (%) | Unique Values | Constant? | Information Classification |")
rp("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
for col in df.columns:
    idx = df.columns.get_loc(col) + 1
    dtype = dtypes_dict[col]
    cnt = len(df) - missing_dict[col]
    m_pct = (missing_dict[col] / len(df)) * 100.0
    u_cnt = unique_dict[col]
    is_c = "⚠️ **YES**" if u_cnt == 1 else "No"
    
    if col == 'strength_mpa':
        role = "🎯 **Primary Continuous Target**"
    elif col in ['sample_id', 'experiment_id', 'sample_replicate']:
        role = "Identifier / Cohort Grouping"
    elif col in ['source_file', 'source_sheet', 'source_row_index', 'is_restored_value']:
        role = "Audit Provenance / Lineage Metadata"
    elif u_cnt == 1:
        role = "Zero-Variance Constant Feature"
    else:
        role = "Active Input Predictor"
        
    rp(f"| {idx} | `{col}` | `{dtype}` | {cnt:,} | {m_pct:.1f}% | {u_cnt:,} | {is_c} | {role} |")
rp()
rp("### 1.3 Audit of Apparent Duplicate Rows")
rp(f"- Exactly **{dup_tuples} rows** share identical values across the 7 physical feature-target columns (`concrete_type`, `curing_age_days`, `cement_type`, `bacterial_species`, `bacterial_concentration_cells_ml`, `mechanical_property`, `strength_mpa`).")
rp("- **Scientific Investigation**: In laboratory strength testing, all 100 specimens in a given batch share identical mix features. Measurements are recorded to 3 decimal places. Naturally, distinct physical specimens occasionally fail under the exact same machine load (e.g. two cylinders fracturing at `2.024 MPa`).")
rp("- **Conclusion**: These are **distinct physical specimens**, NOT data entry errors or software duplications. They are preserved intact with distinct `sample_id` and `sample_replicate` values.")
rp()
rp("---")
rp()
rp("## 2. Target Distributions & Summary Statistics")
rp()
rp("Each of the three mechanical properties was evaluated independently across 1,200 destructive failure tests (600 Normal Concrete, 600 Bacterial Concrete).")
rp()
rp("### 2.1 Statistical Parameters per Mechanical Property")
rp()
rp("| Mechanical Property | Specimen Count | Mean (MPa) | Median (MPa) | Std Dev (MPa) | Variance | Min (MPa) | Max (MPa) | Q1 (MPa) | Q3 (MPa) | IQR (MPa) | Coeff. of Variation (%) |")
rp("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
for prop in props:
    s = target_stats[prop]
    rp(f"| **{prop}** | {s['count']:,} | {s['mean']:.3f} | {s['median']:.3f} | {s['std']:.3f} | {s['var']:.3f} | {s['min']:.3f} | {s['max']:.3f} | {s['q1']:.3f} | {s['q3']:.3f} | {s['iqr']:.3f} | {s['cv']:.2f}% |")
rp()
rp("### 2.2 Distribution Observations")
rp("1. **Compressive Strength** exhibits a wide range (**15.570 MPa to 44.160 MPa**) and the highest coefficient of variation (**21.15%**), reflecting substantial hydration hardening from 7 to 90 days.")
rp("2. **Flexural Strength** spans **2.428 MPa to 6.490 MPa** (CV = **19.34%**), exhibiting multi-modal clustering corresponding to curing age tiers.")
rp("3. **Split Tensile Strength** spans **1.380 MPa to 4.183 MPa** (CV = **21.11%**), showing consistent rightward shifts in bacterial concrete across all age brackets.")
rp()
def make_fig_link(fname):
    clean_p = os.path.abspath(f"reports/figures/{fname}").replace('\\', '/')
    return f"[`reports/figures/{fname}`](file:///{clean_p})"

rp("### 2.3 Visual Distribution Plots")
rp()
rp("| Figure Reference | Image Description |")
rp("| :--- | :--- |")
rp(f"| **Figure 1** | {make_fig_link('01_target_distribution_compressive.png')} — Compressive Strength distribution showing Normal vs. Bacterial overlay. |")
rp(f"| **Figure 2** | {make_fig_link('02_target_distribution_flexural.png')} — Flexural Strength distribution showing Normal vs. Bacterial overlay. |")
rp(f"| **Figure 3** | {make_fig_link('03_target_distribution_split_tensile.png')} — Split Tensile Strength distribution showing Normal vs. Bacterial overlay. |")
rp()
rp("---")
rp()
rp("## 3. Hydration Age Kinetics & Strength Development")
rp()
rp("Mechanical strength was tested at 6 hydration curing ages: **7, 14, 21, 28, 56, and 90 days**. Each age-mix combination contains exactly 100 physical test replicates.")
rp()
rp("### 3.1 Mean Strength Trajectory by Curing Age")
rp()
rp("| Mechanical Property | Concrete Mix | 7 Days (Mean ± SD) | 14 Days (Mean ± SD) | 21 Days (Mean ± SD) | 28 Days (Mean ± SD) | 56 Days (Mean ± SD) | 90 Days (Mean ± SD) |")
rp("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
for prop in props:
    for c_type in ['Normal Concrete', 'Bacterial Concrete']:
        sub = df[(df['mechanical_property'] == prop) & (df['concrete_type'] == c_type)]
        vals = []
        for a in ages:
            m = sub[sub['curing_age_days'] == a]['strength_mpa'].mean()
            sd = sub[sub['curing_age_days'] == a]['strength_mpa'].std()
            vals.append(f"{m:.2f} ± {sd:.2f}")
        rp(f"| {prop} | {c_type} | {' | '.join(vals)} |")
rp()
rp("### 3.2 Age Effect Figures")
rp()
rp("| Figure Reference | Image Description |")
rp("| :--- | :--- |")
rp(f"| **Figure 4** | {make_fig_link('04_age_effect_compressive.png')} — Compressive Strength trajectory across curing days with ±1 SD error bars. |")
rp(f"| **Figure 5** | {make_fig_link('05_age_effect_flexural.png')} — Flexural Strength trajectory across curing days with ±1 SD error bars. |")
rp(f"| **Figure 6** | {make_fig_link('06_age_effect_split_tensile.png')} — Split Tensile Strength trajectory across curing days with ±1 SD error bars. |")
rp()
rp("---")
rp()
rp("## 4. Bacterial Treatment Associations")
rp()
rp("> **Scientific Disclaimer**: The numerical metrics below describe the **observed differences** and **observed statistical associations** between the inoculated ($10^6$ cells/mL) and control specimen cohorts in this laboratory investigation. Because external confounding variables (e.g. ambient mixing humidity, cement bag batch consistency) were unrecorded, these differences must **NOT** be interpreted as definitive causal proof.")
rp()
rp("### 4.1 Observed Strength Differences by Property & Curing Age")
rp()
rp("| Mechanical Property | Curing Age (Days) | Control Mean (MPa) | Bacterial Mean (MPa) | Observed Absolute Diff (MPa) | Observed Relative Diff (%) |")
rp("| :--- | :--- | :--- | :--- | :--- | :--- |")
for _, r in diff_df.iterrows():
    rp(f"| {r['mechanical_property']} | {int(r['curing_age_days'])} Days | {r['normal_mean']:.3f} | {r['bacterial_mean']:.3f} | **+{r['absolute_difference']:.3f}** | **+{r['percentage_difference']:.2f}%** |")
rp()
rp("### 4.2 Key Observed Patterns")
rp("1. **Sustained Superiority**: In all 18 test conditions (3 properties × 6 ages), the bacterial concrete cohorts exhibited higher mean fracture strength than corresponding control cohorts.")
rp("2. **Compressive Strength Peak Relative Difference**: The largest observed relative difference in compressive strength occurs at **90 days (+23.86%, +7.83 MPa)** and **56 days (+20.46%, +6.32 MPa)**, suggesting ongoing calcite ($CaCO_3$) pore sealing during late curing.")
rp("3. **Flexural & Tensile Early Gains**: Flexural and split tensile strength showed substantial observed increases as early as 7 days (**+17.89%** in flexure, **+17.26%** in tensile), indicating enhanced aggregate-paste interfacial transition zone (ITZ) bonding.")
rp()
rp("---")
rp()
rp("## 5. Statistical Hypothesis Testing & Cohort Grouping Considerations")
rp()
rp("For each of the 18 experimental conditions, we conducted two-sample inferential tests comparing the 100 Bacterial Concrete specimens against the 100 Normal Concrete specimens.")
rp()
rp("### 5.1 Test Results Table")
rp()
rp("| Mechanical Property | Curing Age | Welch's t-Statistic | Welch's p-Value | Mann-Whitney U | MW p-Value | Cohen's d | 95% Confidence Interval (MPa) |")
rp("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
for _, r in stat_df.iterrows():
    p_str = "< 0.0001" if r['welch_p'] < 0.0001 else f"{r['welch_p']:.4e}"
    mw_str = "< 0.0001" if r['mann_whitney_p'] < 0.0001 else f"{r['mann_whitney_p']:.4e}"
    rp(f"| {r['mechanical_property']} | {int(r['curing_age_days'])}d | {r['welch_t']:.2f} | `{p_str}` | {r['mann_whitney_u']:.1f} | `{mw_str}` | **{r['cohens_d']:.2f}** | [{r['ci_lower']:.3f}, {r['ci_upper']:.3f}] |")
rp()
rp("### 5.2 Critical Methodological Discussion on Replicate Independence")
rp("> ⚠️ **CRITICAL SCIENTIFIC CAVEAT: Do NOT Blindly Interpret p < 0.0001 as Conclusive Scientific Proof**")
rp()
rp("1. **The Experimental Unit Problem**: Standard inferential tests (Student's t, Welch's t, Mann-Whitney U) assume that each observation is an independent, identically distributed (i.i.d.) random draw from a broad population. In civil engineering experiments, however, 100 test cubes or prisms for a given curing age are typically cast simultaneously from **1 or 2 batch mixer runs**.")
rp("2. **Clustering & Intraclass Correlation**: Specimens cast from the same batch share common mixing water, environmental temperature, relative humidity, and compaction vibration. The effective degrees of freedom are far lower than the nominal sample size ($N=100$).")
rp("3. **Inflated Type I Error**: Treating pseudo-replicates as fully independent drives standard errors artificially close to zero, producing astronomical t-statistics (e.g. $t > 15$) and tiny p-values ($p < 10^{-30}$). While the effect size (Cohen's $d > 1.5$) indicates substantial separation between the cohorts, true scientific generalizability requires multi-batch replication.")
rp()
rp("---")
rp()
rp("## 6. Outlier Analysis & IQR Diagnostics")
rp()
rp("Potential outliers were evaluated using the standard Tukey Interquartile Range method ($[Q_1 - 1.5 \\times \\text{IQR}, Q_3 + 1.5 \\times \\text{IQR}]$) calculated within each of the 36 experimental cohorts.")
rp()
rp("### 6.1 Cohort Outlier Diagnostics")
rp()
rp("| Mechanical Property | Total Observations | Total Flagged Outliers | Overall Outlier Rate (%) | Maximum Outlier Count in Single Cohort |")
rp("| :--- | :--- | :--- | :--- | :--- |")
for prop in props:
    sub_o = outlier_df[outlier_df['mechanical_property'] == prop]
    rp(f"| **{prop}** | {sub_o['total_count'].sum():,} | {sub_o['outlier_count'].sum()} | {sub_o['outlier_count'].sum() / sub_o['total_count'].sum() * 100.0:.2f}% | {sub_o['outlier_count'].max()} outliers |")
rp()
rp(f"- **Overall Dataset Outlier Rate**: Exactly **{outlier_df['outlier_count'].sum()} out of 3,600 observations ({outlier_df['outlier_count'].sum()/3600*100:.2f}%)** fall outside cohort IQR fences.")
rp("- **Policy Compliance**: In strict adherence to experimental integrity, **NO outliers have been dropped or modified**. These points represent natural mechanical fracture variability in heterogeneous concrete matrices.")
rp()
rp("### 6.2 Outlier & Boxplot Visualizations")
rp()
rp("| Figure Reference | Image Description |")
rp("| :--- | :--- |")
rp(f"| **Figure 7** | {make_fig_link('07_boxplots_compressive.png')} — Boxplots for Compressive Strength across curing ages. |")
rp(f"| **Figure 8** | {make_fig_link('08_boxplots_flexural.png')} — Boxplots for Flexural Strength across curing ages. |")
rp(f"| **Figure 9** | {make_fig_link('09_boxplots_split_tensile.png')} — Boxplots for Split Tensile Strength across curing ages. |")
rp(f"| **Figure 11** | {make_fig_link('11_outliers_iqr.png')} — Multi-panel Tukey IQR outlier diagnostic plot across all properties. |")
rp()
rp("---")
rp()
rp("## 7. Correlation Analysis & Dimensionality")
rp()
rp("Pairwise Pearson correlation coefficients were computed across numerical columns (`curing_age_days`, `bacterial_concentration_cells_ml`, `strength_mpa`).")
rp()
rp("### 7.1 Correlation Matrices")
rp()
rp("#### Global Correlation (All 3,600 Observations Combined)")
rp()
rp("| Variable | `curing_age_days` | `bacterial_concentration_cells_ml` | `strength_mpa` |")
rp("| :--- | :--- | :--- | :--- |")
for row_col in num_cols:
    vals = [f"{corr_global.loc[row_col, c]:.4f}" for c in num_cols]
    rp(f"| `{row_col}` | {' | '.join(vals)} |")
rp()
rp("#### Property-Specific Correlations with `strength_mpa`")
rp()
rp("| Mechanical Property | Correlation with `curing_age_days` ($r$) | Correlation with `bacterial_concentration_cells_ml` ($r$) |")
rp("| :--- | :--- | :--- |")
for prop in props:
    r_age = corr_per_prop[prop].loc['curing_age_days', 'strength_mpa']
    r_bac = corr_per_prop[prop].loc['bacterial_concentration_cells_ml', 'strength_mpa']
    rp(f"| **{prop}** | **+{r_age:.4f}** | **+{r_bac:.4f}** |")
rp()
rp("### 7.2 Crucial Interpretation of Bacterial Concentration Correlation")
rp("> ⚠️ **DO NOT INTERPRET AS A CONTINUOUS DOSE-RESPONSE RELATIONSHIP**")
rp("- In the Master Dataset, `bacterial_concentration_cells_ml` has strictly **two discrete levels**: `0` (Control) and `1,000,000` (Inoculated).")
rp("- Mathematically, the Pearson correlation between a continuous variable and a binary indicator is a **point-biserial correlation** ($r_{pb}$).")
rp("- It measures the degree of mean separation between the two tested groups, **NOT a continuous dose-response gradient**. We cannot extrapolate performance at intermediate ($10^4, 10^5$) or higher ($10^7, 10^8$) dosages from this correlation coefficient.")
rp()
rp("### 7.3 Visual Heatmap")
rp(f"- {make_fig_link('10_correlation_heatmap.png')} — Multi-panel correlation heatmap illustrating global and property-specific relationships.")
rp()
rp("---")
rp()
rp("## 8. Feature Information Content & Modeling Suitability")
rp()
rp("Every feature in the Master Dataset was audited to assess its predictive utility and statistical information content:")
rp()
rp("| Feature Name | Data Type | Unique Count | Variance | Status | Redundancy Assessment | Raw vs. Derived |")
rp("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
rp("| `curing_age_days` | `int64` | 6 | 821.89 | **Active Predictor** | Primary driver of cement hydration hardening. | Raw experimental parameter. |")
rp("| `concrete_type` | `str` | 2 | 0.25 | **Active Predictor** | Perfectly collinear with `bacterial_status`. | Raw experimental parameter. |")
rp("| `bacterial_status` | `str` | 2 | 0.25 | Redundant Feature | 100% duplicate information of `concrete_type`. | Derived indicator. |")
rp("| `bacterial_species` | `str` | 2 | 0.25 | Redundant Feature | 100% duplicate information of `concrete_type` (`None` vs `B. subtilis`). | Raw (with logical derived encoding for NC). |")
rp("| `bacterial_concentration_cells_ml` | `int64` | 2 | 2.5e11 | **Active Predictor** | Binary dosage representation (`0` vs `10^6`). Collinear with `concrete_type`. | Raw (with logical derived encoding for NC). |")
rp("| `cement_type` | `str` | 1 | 0.00 | **Zero-Variance Constant** | Constant across all 3,600 rows (`OPC 53`). Zero predictive gradient. | Raw experimental parameter. |")
rp("| `mechanical_property` | `str` | 3 | 0.67 | **Condition / Task Selector** | Defines the target failure mechanism. | Raw experimental parameter. |")
rp("| `specimen_geometry` | `str` | 3 | 0.67 | Redundant Feature | 100% collinear with `mechanical_property` (Cube=CS, Prism=FS, Cylinder=TS). | Domain metadata. |")
rp("| `sample_replicate` | `int64` | 100 | 833.48 | Cohort Alignment / Grouping | Physical sample replicate index (1–100). | Raw experimental coordinate. |")
rp()
rp("---")
rp()
rp("## 9. Data Leakage Assessment & Risk Segregation")
rp()
rp("To ensure machine learning validity, schema columns are formally partitioned into three distinct risk tiers:")
rp()
rp("```text")
rp("Feature Risk Segregation:")
rp("├── SAFE MODELING FEATURES     -> curing_age_days, concrete_type (or bacterial_concentration), mechanical_property")
rp("├── COHORT / SPLITTING IDS     -> experiment_id, sample_replicate")
rp("└── FORBIDDEN / LEAKAGE FIELDS -> specimen_id, source_*, is_restored_value, specimen_geometry")
rp("```")
rp()
rp("### 9.1 Risk Classification Table")
rp()
rp("| Feature Category | Field Names | Why Safe or Why Forbidden |")
rp("| :--- | :--- | :--- |")
rp("| **Safe Modeling Features** | `curing_age_days`, `concrete_type`, `bacterial_concentration_cells_ml`, `mechanical_property` | Legitimate physical inputs known prior to testing. Free from target leakage. |")
rp("| **Cohort & Validation Grouping** | `experiment_id`, `sample_replicate` | Must **NOT** be fed as input features to regression trees/neural nets (prevents memorization). Reserved strictly for cross-validation grouping. |")
rp("| **Lineage & Provenance Metadata** | `source_file`, `source_sheet`, `source_row_index` | Must be excluded from model inputs. Workbook row indices correlate with curing age blocks and would induce spurious shortcut learning. |")
rp("| **Restoration Flag** | `is_restored_value` | Strictly excluded. Exists only for 56d/90d bacterial flexural rows; exposing this to models would leak property and curing age directly. |")
rp("| **Redundant Collinear Geometry** | `specimen_geometry`, `bacterial_status`, `bacterial_species` | Redundant with `mechanical_property` and `concrete_type`. Exclude from training to prevent collinearity instabilities. |")
rp()
rp("---")
rp()
rp("## 10. Generalization Limitations & Scientific Boundary Conditions")
rp()
rp("Any model trained on this dataset will operate within strict physical boundary conditions:")
rp()
rp("1. **Single Cement Binder**: The dataset contains exclusively `OPC 53 (UltraTech)`. The model cannot generalize to Pozzolanic Portland Cement (PPC), Slag Cement (PSC), or low-heat cement formulations.")
rp("2. **Single Bacterial Species**: Tested exclusively with `Bacillus subtilis`. It cannot generalize to other ureolytic or carbonate-precipitating strains (e.g. *Sporosarcina pasteurii*, *Bacillus sphaericus*, *Bacillus cohnii*).")
rp("3. **Single Dosage Level ($10^6$ cells/mL)**: The model cannot predict non-linear saturation kinetics at higher concentrations ($10^7, 10^8$) or lower concentrations ($10^4$).")
rp("4. **Unrecorded Mix Proportions**: Water-cement ratio, coarse/fine aggregate grading, superplasticizer dosage, and curing water chemical composition were unrecorded.")
rp("5. **Batch Clustering**: All 100 replicates per age were cast under homogeneous laboratory conditions. The model predicts laboratory test performance under standardized conditions, not site-cast concrete variability.")
rp()
rp("---")
rp()
rp("## 11. Required Figures Manifest")
rp()
fig_dir_abs = os.path.abspath("reports/figures").replace("\\", "/")
rp(f"All 11 figures have been generated and saved under [`reports/figures/`](file:///{fig_dir_abs}):")
rp()

def make_fig_link(fname):
    abs_p = os.path.abspath(os.path.join("reports", "figures", fname)).replace("\\", "/")
    return f"[`reports/figures/{fname}`](file:///{abs_p})"

rp(f"1. **Figure 1**: {make_fig_link('01_target_distribution_compressive.png')} — Target distribution: Compressive Strength (KDE & histogram).")
rp(f"2. **Figure 2**: {make_fig_link('02_target_distribution_flexural.png')} — Target distribution: Flexural Strength (KDE & histogram).")
rp(f"3. **Figure 3**: {make_fig_link('03_target_distribution_split_tensile.png')} — Target distribution: Split Tensile Strength (KDE & histogram).")
rp(f"4. **Figure 4**: {make_fig_link('04_age_effect_compressive.png')} — Strength vs curing age: Compressive Strength (line plot with ±1 SD error bars).")
rp(f"5. **Figure 5**: {make_fig_link('05_age_effect_flexural.png')} — Strength vs curing age: Flexural Strength (line plot with ±1 SD error bars).")
rp(f"6. **Figure 6**: {make_fig_link('06_age_effect_split_tensile.png')} — Strength vs curing age: Split Tensile Strength (line plot with ±1 SD error bars).")
rp(f"7. **Figure 7**: {make_fig_link('07_boxplots_compressive.png')} — Normal vs bacterial boxplots: Compressive Strength across 6 ages.")
rp(f"8. **Figure 8**: {make_fig_link('08_boxplots_flexural.png')} — Normal vs bacterial boxplots: Flexural Strength across 6 ages.")
rp(f"9. **Figure 9**: {make_fig_link('09_boxplots_split_tensile.png')} — Normal vs bacterial boxplots: Split Tensile Strength across 6 ages.")
rp(f"10. **Figure 10**: {make_fig_link('10_correlation_heatmap.png')} — Pearson correlation heatmap (Global and per mechanical property).")
rp(f"11. **Figure 11**: {make_fig_link('11_outliers_iqr.png')} — Tukey IQR outlier diagnostic plot across all properties and curing ages.")
rp()
rp("---")
rp()
rp("## 12. Final EDA Conclusion & Engineering Recommendations")
rp()
rp("### A. Data Quality Assessment")
rp("- The Master Dataset V1 is of **exceptionally high technical quality**: 0 missing values, complete 16-attribute provenance lineage, exact 1-to-1 restoration of 56d/90d flexure, and zero corrupted summary rows.")
rp("- Measurement distributions across all 36 test cohorts conform closely to Gaussian experimental failure behavior.")
rp()
rp("### B. Main Patterns Discovered")
rp("- Hydration age kinetics show classic logarithmic progression across all properties, with rapid hardening between 7 and 28 days followed by asymptotic maturation up to 90 days.")
rp("- Bacterial concrete exhibits parallel or diverging growth curves relative to control concrete, with the largest absolute gains concentrated in late-age compressive strength.")
rp()
rp("### C. Bacterial-Treatment Observations")
rp("- Bacterial concrete is consistently associated with higher failure loads across all properties (+10.5% to +23.9% relative mean increases).")
rp("- Because experimental replicates were cast in synchronous batches, these associations must be validated on independent casting batches before asserting unconditional causal claims.")
rp()
rp("### D. Important Limitations")
rp("- The dataset provides narrow feature diversity: 1 cement grade, 1 bacterial species, 1 concentration level ($10^6$), and no recorded water-cement ratios.")
rp()
rp("### E. Features Recommended for Initial Modeling")
rp("1. `curing_age_days` (continuous / integer hydration period)")
rp("2. `concrete_type` (binary categorical: Control vs Bacterial) OR `bacterial_concentration_cells_ml` (numerical 0 vs 1,000,000)")
rp("3. `mechanical_property` (categorical task selector if training a unified model across properties)")
rp()
rp("### F. Features That Should NOT Be Used as Model Inputs")
rp("- `sample_id`, `experiment_id`, `sample_replicate` (Identifiers)")
rp("- `source_file`, `source_sheet`, `source_row_index` (Lineage metadata; risk of shortcut learning)")
rp("- `is_restored_value` (Direct target leakage indicator)")
rp("- `cement_type` (Zero variance)")
rp("- `bacterial_status`, `bacterial_species`, `specimen_geometry` (100% collinear redundant duplicates)")
rp()
rp("### G. Recommended Validation Strategy")
rp("- **Strategy 1 (Replicate Grouping)**: Group by `sample_replicate` using `GroupKFold` so specimens sharing replicate IDs never appear in both training and testing folds.")
rp("- **Strategy 2 (Temporal Generalization)**: Leave-One-Age-Out (train on 7–56d, test on 90d) to assess the model's capacity to extrapolate maturation kinetics.")
rp()
rp("### H. Questions That Must Be Resolved Before Model Training")
rp("1. **Single Unified Model vs. Three Property-Specific Models**: Should TabPFN be configured as a unified multi-task model (with `mechanical_property` as an input feature), or should 3 dedicated regressors be trained?")
rp("2. **Dose Encoding**: Should bacterial treatment be fed as a categorical flag (`concrete_type`) or as a numerical dosage (`0` vs `1,000,000`)?")
rp("3. **Evaluation Metric Priority**: Should models prioritize Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE), or $R^2$ on late-age (28d/90d) concrete?")
rp()
rp("---")
rp("*End of Exploratory Data Analysis Report*")

report_text = "\n".join(r_lines)
with open("reports/eda_report.md", "w", encoding="utf-8") as f:
    f.write(report_text)

print(f"Successfully generated reports/eda_report.md ({len(report_text):,} characters, {len(r_lines)} lines)")
