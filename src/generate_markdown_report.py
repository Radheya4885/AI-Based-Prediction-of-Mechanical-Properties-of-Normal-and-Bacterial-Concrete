import openpyxl
import pandas as pd
import numpy as np
import os
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Load the schema CSV generated earlier
schema_df = pd.read_csv('reports/dataset_schema.csv')

report_lines = []

def p(line=""):
    report_lines.append(line)

p("# Concrete Strength Prediction Project - Dataset Audit Report")
p()
p("> **Status**: Dataset Audit Complete (Read-Only Mode)")  
p("> **Policy Adherence**: Original datasets have NOT been modified, overwritten, merged, or trimmed. No models have been trained.")
p()
p("---")
p()
p("## Executive Summary")
p()
p("| Metric | Value | Details |")
p("| :--- | :--- | :--- |")
p("| **Total Files Audited** | **3 Excel Workbooks** | `C Strenth Reading 1000000.xlsx`, `F Strength Reading 1000000.xlsx`, `T Strenth Reading 1000000.xlsx` |")
p("| **Total Sheets Audited** | **15 Sheets** | 11 in C Strength, 2 in F Strength, 2 in T Strength |")
p("| **Total Columns Cataloged** | **179 Columns** | Documented in `reports/dataset_schema.csv` |")
p("| **Primary Targets** | **3 Mechanical Properties** | Compressive Strength (MPa), Flexural Strength (MPa), Split Tensile Strength (MPa) |")
p("| **Secondary Durability Tests** | **2 Properties** | Rapid Chloride Permeability Test (RCPT in Coulombs), Water Absorption (%) |")
p("| **Curing Ages** | **6 Ages** | 7, 14, 21, 28, 56, and 90 Days |")
p("| **Bacterial Species** | **1 Species** | *Bacillus subtilis* |")
p("| **Bacterial Concentrations** | **7 Concentrations** | Primary: $10^6$ (1,000,000 cells/mL); Summary sheets also include Control ($0$), $10^3, 10^4, 10^5, 10^7, 10^8, 10^9$ cells/mL |")
p()
p("---")
p()

# SECTION 1: DETAILED AUDIT PER DATASET
p("## 1. Detailed Dataset Audits")
p()

files = [
    {
        'name': 'C Strenth Reading 1000000.xlsx',
        'path': 'data/C Strenth Reading 1000000.xlsx',
        'desc': 'Compressive Strength primary workbook with durability and cross-concentration comparisons',
        'target': 'Compressive Strength (MPa)'
    },
    {
        'name': 'F Strength Reading 1000000.xlsx',
        'path': 'data/F Strength Reading 1000000.xlsx',
        'desc': 'Flexural Strength workbook containing long-format and sample matrix sheets',
        'target': 'Flexural Strength (MPa)'
    },
    {
        'name': 'T Strenth Reading 1000000.xlsx',
        'path': 'data/T Strenth Reading 1000000.xlsx',
        'desc': 'Split Tensile Strength workbook containing long-format and sample matrix sheets',
        'target': 'Split Tensile Strength (MPa)'
    }
]

for f_idx, f_info in enumerate(files, 1):
    fname = f_info['name']
    fpath = f_info['path']
    p(f"### 1.{f_idx} Dataset File: `{fname}`")
    clean_path = os.path.abspath(fpath).replace('\\', '/')
    p(f"- **File Path**: [`{fpath}`](file:///{clean_path})")
    p(f"- **File Size**: {os.path.getsize(fpath):,} bytes")
    p(f"- **Primary Target Variable**: **{f_info['target']}**")
    p(f"- **Description**: {f_info['desc']}")
    p()
    
    wb_val = openpyxl.load_workbook(fpath, data_only=True)
    wb_form = openpyxl.load_workbook(fpath, data_only=False)
    sheet_names = wb_val.sheetnames
    p(f"#### Sheet Overview ({len(sheet_names)} sheets)")
    p()
    p("| # | Sheet Name | Total Rows | Total Cols | Nature of Content | Derived/Formulas | Duplicate Rows |")
    p("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for s_idx, sname in enumerate(sheet_names, 1):
        s_val = wb_val[sname]
        s_form = wb_form[sname]
        max_r = s_val.max_row or 0
        max_c = s_val.max_column or 0
        
        # formulas
        f_count = sum(1 for row in s_form.iter_rows(values_only=True) for cell in row if isinstance(cell, str) and cell.startswith('='))
        
        # df for duplicates
        try:
            df_s = pd.read_excel(fpath, sheet_name=sname, header=None)
            dups = int(df_s.duplicated().sum())
        except:
            dups = 0
            
        nature = "Calculated / Summary"
        if sname in ['Comp 6', 'Sheet1']:
            nature = "**Raw Experimental Measurements**"
        elif sname in ['Sheet2', 'Ten 6']:
            nature = "Raw Sample Matrix + Summary Stats"
        elif sname in ['RCPT', 'Water Absorption']:
            nature = "Durability Benchmark Readings"
            
        f_str = f"Yes ({f_count} formula cells)" if f_count > 0 else "None (hardcoded values)"
        p(f"| {s_idx} | **`{sname}`** | {max_r} | {max_c} | {nature} | {f_str} | {dups} |")
    p()
    
    # Detailed section for major sheets
    p(f"#### Comprehensive Column & Statistical Audit for `{fname}`")
    p()
    for sname in sheet_names:
        sheet_df = schema_df[(schema_df['dataset_file'] == fname) & (schema_df['sheet_name'] == sname)]
        p(f"##### Sheet: `{sname}`")
        p(f"- **Dimensions**: {len(pd.read_excel(fpath, sheet_name=sname, header=None))} rows x {sheet_df.shape[0]} columns cataloged")
        
        # Check formulas in this sheet
        s_form = wb_form[sname]
        sample_formulas = []
        for r_i, row in enumerate(s_form.iter_rows(values_only=True)):
            for c_i, cell in enumerate(row):
                if isinstance(cell, str) and cell.startswith('='):
                    coord = f"{openpyxl.utils.get_column_letter(c_i+1)}{r_i+1}"
                    sample_formulas.append(f"`{coord}` = `{cell}`")
                    if len(sample_formulas) >= 4:
                        break
            if len(sample_formulas) >= 4:
                break
        if sample_formulas:
            p(f"- **Detected Excel Formulas**: {', '.join(sample_formulas)}")
        else:
            p(f"- **Detected Excel Formulas**: None (all cell values are stored as literals)")
            
        p()
        p("| Col # | Column Name | Data Type | Non-Null / Total | Missing (%) | Unique | Constant? | Min | Mean | Max | Nature / Role |")
        p("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        
        for _, row in sheet_df.iterrows():
            c_idx = row['column_index']
            c_name = row['column_name']
            dtype = row['data_type']
            cnt = f"{row['non_null_count']}/{row['total_count']}"
            m_pct = f"{row['missing_percentage']}%"
            u_cnt = row['unique_count']
            is_c = "⚠️ **YES**" if row['is_constant'] else "No"
            
            min_v = f"{row['min_value']}" if pd.notnull(row['min_value']) and str(row['min_value']) != '' else "-"
            mean_v = f"{row['mean_value']}" if pd.notnull(row['mean_value']) and str(row['mean_value']) != '' else "-"
            max_v = f"{row['max_value']}" if pd.notnull(row['max_value']) and str(row['max_value']) != '' else "-"
            
            nature = row['measurement_nature']
            if row['is_target']:
                nature = f"🎯 **{row['target_type']}**"
                
            p(f"| {c_idx} | `{c_name}` | `{dtype}` | {cnt} | {m_pct} | {u_cnt} | {is_c} | {min_v} | {mean_v} | {max_v} | {nature} |")
        p()
    p("---")
    p()

# SECTION 2: SPECIALIZED AUDIT FINDINGS (11 to 14)
p("## 2. Cross-Dataset Metadata & Feature Identification")
p()
p("### 2.1 Concrete Type, Curing Age, Bacteria & Concentrations Identified")
p()
p("| Attribute | Observed Values | Present in Sheets | Significance |")
p("| :--- | :--- | :--- | :--- |")
p("| **Concrete Type** | `Normal Concrete` (NC), `Bacterial Concrete` (BC) | `Comp 6`, `Sheet1` (F & T), `Sheet2`, `Ten 6`, `c`, `f`, `t` | Core categorical input feature. |")
p("| **Curing Age (Days)** | `7`, `14`, `21`, `28`, `56`, `90` | All sheets | Primary temporal numerical feature governing cement hydration and biomineralization. |")
p("| **Cement Type** | `OPC 53 (UltraTech)` | `Comp 6`, `Sheet1` (F & T) | Ordinary Portland Cement 53 grade. Currently **constant across all rows**. |")
p("| **Bacterial Species** | `Bacillus subtilis` | `Comp 6`, `Sheet1` (F & T) | Ureolytic / calcite-precipitating endospore-forming bacteria. **Constant across all rows**. |")
p("| **Bacterial Concentration** | `1,000,000 cells/mL` ($10^6$) | Main reading sheets | In raw sample sheets, concentration is fixed at $10^6$. In summary sheets (`c`, `f`, `t`, `RCPT`, `Water Absorption`), concentrations span `Control (0)`, $10^3, 10^4, 10^5, 10^6, 10^7, 10^8, 10^9$ cells/mL. |")
p("| **Durability: RCPT** | `1900` to `3100` Coulombs | `RCPT` in C Strength | Rapid Chloride Permeability Test: shows lowest permeability at $10^5$ cells/mL ($1900$ C) vs Control ($3100$ C). |")
p("| **Durability: Water Absorption** | `4.3%` to `5.8%` | `Water Absorption` in C Strength | Minimum water absorption observed at $10^5$ cells/mL ($4.3%$) vs $10^3$ ($5.8%$). |")
p()

p("### 2.2 Target Variables Identified")
p()
p("1. **Compressive Strength (MPa)**:")
p("   - Primary target in `C Strenth Reading 1000000.xlsx`.")
p("   - Raw individual sample readings located in `Comp 6` (600 rows paired) and `Sheet2` (100 samples across 12 conditions).")
p("   - Range: Normal concrete 7d min **15.57 MPa** to Bacterial concrete 90d max **44.16 MPa**.")
p()
p("2. **Flexural Strength (MPa)**:")
p("   - Primary target in `F Strength Reading 1000000.xlsx`.")
p("   - Raw individual sample readings in `Sheet1` (1200 rows) and `Sheet2` (100 samples across 12 conditions).")
p("   - Range: Normal concrete 7d min **2.21 MPa** to Bacterial concrete 90d max **6.39 MPa** (summary) / **6.19 MPa** (samples).")
p()
p("3. **Split Tensile Strength (MPa)**:")
p("   - Primary target in `T Strenth Reading 1000000.xlsx`.")
p("   - Raw individual sample readings in `Sheet1` (1200 rows) and `Ten 6` (100 samples across 12 conditions).")
p("   - Range: Normal concrete 7d min **1.44 MPa** to Bacterial concrete 90d max **4.25 MPa**.")
p()

p("### 2.3 Identification of Raw vs. Derived/Calculated Tables")
p()
p("- **Raw Experimental Measurements**:")
p("  - `C Strenth Reading 1000000.xlsx` -> `Comp 6`: 600 side-by-side rows representing 100 concrete cylinder/cube tests per curing age for both Control and Bacterial mix.")
p("  - `F Strength Reading 1000000.xlsx` -> `Sheet1`: 1200 rows (unpivoted long format).")
p("  - `F Strength Reading 1000000.xlsx` -> `Sheet2` (rows 5 to 104): 100 physical beam flexural fracture tests per condition.")
p("  - `T Strenth Reading 1000000.xlsx` -> `Sheet1`: 1200 rows (unpivoted long format).")
p("  - `T Strenth Reading 1000000.xlsx` -> `Ten 6` (rows 4 to 103): 100 physical Brazilian cylinder split tensile tests per condition.")
p("- **Derived / Summary / Benchmark Tables**:")
p("  - `Sheet2` / `Ten 6` (rows 105 onwards): Contain `SUM` formulas and mean calculations (`=AVERAGE(...)` / hardcoded sums up to 4066.08).")
p("  - `Final Result`, `105 c`, `c`, `f`, `t`: Macro-level laboratory summary tables comparing relative gain (%) and average strengths across bacteria dilution series ($10^3$ to $10^8$).")
p("  - `RCPT` and `Water Absorption`: Durability single-value measurements across doses.")
p()
p("---")
p()

# SECTION 3: DATA INTEGRITY & QUALITY ISSUES
p("## 3. Data Quality Issues, Anomalies & Potential Risks")
p()
p("### ⚠️ Critical Issue 1: Missing Data in `F Strength Reading 1000000.xlsx` (`Sheet1`)")
p("- In `Sheet1` of `F Strength Reading 1000000.xlsx`, rows 1001 to 1200 have **200 completely empty (`NaN`) target values**:")
p("  - `Bacterial Concrete` at **56 Days**: 100 missing rows (rows 1001–1100).")
p("  - `Bacterial Concrete` at **90 Days**: 100 missing rows (rows 1101–1200).")
p("- **Root Cause Discovered in Audit**: The data transcriber unpivoted `Sheet2` into `Sheet1` but stopped at 28 days for Bacterial Concrete.")
p("- **Remedy Available**: The original measurements for `56 BC` and `90 BC` **DO EXIST** in `Sheet2` of the same workbook (100 complete readings for each condition). When training/cleaning is permitted, these values can be mapped from `Sheet2` without data loss.")
p()
p("### ⚠️ Critical Issue 2: Summary Rows Embedded in Matrix Sheets (`Sheet2` and `Ten 6`)")
p("- `Sheet2` in `C Strenth Reading 1000000.xlsx`, `Sheet2` in `F Strength Reading 1000000.xlsx`, and `Ten 6` in `T Strenth Reading 1000000.xlsx` contain **100 data rows (samples 1 to 100)** followed immediately by **10–12 summary rows** (column sums, averages, curing age markers).")
p("- Example: In `C Strength Sheet2`, row 105 contains column sums (`1934.44` for 7 N, `4066.08` for 90 BC).")
p("- **Risk**: If these sheets are loaded directly into machine learning pipelines without row slicing, summary sums will be treated as physical sample strengths, resulting in catastrophic model distortion.")
p()
p("### ⚠️ Critical Issue 3: Zero-Variance (Constant) Feature Columns")
p("- Across all primary raw sheets (`Comp 6`, `F Sheet1`, `T Sheet1`):")
p("  - `Cement` is strictly `'OPC 53 (UltraTech)'` (variance = 0).")
p("  - `Bacteria` is strictly `'Bacillus subtilis'` (variance = 0).")
p("  - `Bacterial Concentration (cells/mL)` is strictly `1,000,000` (variance = 0).")
p("- **Implication**: These features cannot provide predictive gradient within the $10^6$ dataset alone. If multi-concentration data (from sheets `c`, `f`, `t`) is later incorporated, concentration will become an active numerical feature.")
p()
p("### ⚠️ Critical Issue 4: Schema Discrepancy Across Target Workbooks")
p("- `T Strength` (`Sheet1`) and `F Strength` (`Sheet1`) use an **unpivoted long format** (1200 rows with a `'Concrete Type'` column separating Normal and Bacterial).")
p("- `C Strength` (`Comp 6`) uses a **wide side-by-side format** (600 rows where Column 5 is Normal Concrete Compressive Strength and Column 6 is Bacterial Concrete Compressive Strength, with a 2-level header).")
p("- **Harmonization Required**: For unified multi-task or tabular modeling, `Comp 6` must be reshaped to match the long schema of F and T.")
p()
p("### ⚠️ Critical Issue 5: Duplicate Feature Tuples & Measurement Repetition")
p("- Because input features (`Concrete Type`, `Curing Age`, `Cement`, `Bacteria`, `Concentration`) are identical for batches of 100 test samples, rows sharing identical strength measurements appear as duplicate rows (e.g. 111 duplicate rows in `T Sheet1`, 51 duplicate non-null rows in `F Sheet1`).")
p("- These are legitimate experimental measurement repetitions, not artificial duplicate recordings, but must be treated carefully to avoid cross-fold data contamination during train/test splitting.")
p()
p("### ⚠️ Critical Issue 6: Data Leakage Risks")
p("1. **Leakage from Summary Sheets**: Sheets `Final Result`, `105 c`, `c`, `f`, `t` contain average values calculated from test specimens. If summary sheet rows are merged with individual sample rows, aggregate target statistics will leak directly into training.")
p("2. **Leakage from Paired Batch Splitting**: Test specimens were produced in concurrent laboratory casting batches. If individual specimens are shuffled randomly with K-fold CV rather than stratified or grouped, test sets will share batch variance with training sets.")
p()
p("---")
p()

# SECTION 4: CONSTRAINTS VERIFICATION
p("## 4. Verification of Operational Constraints")
p()
p("- [x] **Constraint 15**: **No datasets merged**. Each file and sheet was audited strictly in isolation.")
p("- [x] **Constraint 16**: **No model trained**. No machine learning, TabPFN fitting, or statistical modeling was initiated.")
p("- [x] **Constraint 17**: **No rows or columns removed**. All original files in `data/` remain 100% unaltered and intact.")
p("- [x] **Constraint 18**: **Artifacts generated**:")
p("  - Audit report: [`reports/dataset_audit.md`](file:///g:/Projects/Concrete%20testing/reports/dataset_audit.md)")
p("  - Schema metadata CSV: [`reports/dataset_schema.csv`](file:///g:/Projects/Concrete%20testing/reports/dataset_schema.csv)")
p()
p("---")
p()

# SECTION 5: FINAL CONCISE SUMMARY TABLE
p("## 5. Concise Synthesis & Next Steps")
p()
p("| Item | Findings / Summary |")
p("| :--- | :--- |")
p("| **Total Files** | 3 Excel workbooks (`C Strenth Reading 1000000.xlsx`, `F Strength Reading 1000000.xlsx`, `T Strenth Reading 1000000.xlsx`) |")
p("| **Total Sheets** | 15 total sheets (11 in C, 2 in F, 2 in T) |")
p("| **Total Rows per Dataset** | `C Comp 6`: 600 data rows (paired N & BC); `F Sheet1`: 1200 rows; `T Sheet1`: 1200 rows. (Sample matrix sheets `Sheet2` & `Ten 6`: 100 test samples each) |")
p("| **Available Mechanical Properties** | Compressive Strength (15.57 – 44.16 MPa), Flexural Strength (2.21 – 6.19 MPa), Split Tensile Strength (1.44 – 4.25 MPa) |")
p("| **Available Bacterial Concentrations** | $10^6$ cells/mL across all raw sample datasets. Concentrations $10^3, 10^4, 10^5, 10^7, 10^8$ available in summary sheets `c`, `f`, `t`. |")
p("| **Missing / Constant Variables** | **Missing**: 200 values in `F Sheet1` (56d BC & 90d BC). **Constants**: `Cement` ('OPC 53'), `Bacteria` ('Bacillus subtilis'), `Bacterial Concentration` ($10^6$) |")
p("| **Potential Data-Quality Issues** | Transcriber truncation in `F Sheet1`, trailing summary sum rows in matrix sheets, schema divergence (paired columns in C vs long rows in F and T) |")
p("| **Possible Data Leakage Risks** | Merging summary sheets with raw specimen sheets; random train/test splitting across identical sample batches |")

report_text = "\n".join(report_lines)

with open('reports/dataset_audit.md', 'w', encoding='utf-8') as f:
    f.write(report_text)

print("Successfully wrote reports/dataset_audit.md (length:", len(report_text), "chars)")
