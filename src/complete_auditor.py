import openpyxl
import pandas as pd
import numpy as np
import os
import sys
import csv

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

datasets = [
    {
        'filename': 'C Strenth Reading 1000000.xlsx',
        'filepath': 'data/C Strenth Reading 1000000.xlsx',
        'primary_target': 'Compressive Strength'
    },
    {
        'filename': 'F Strength Reading 1000000.xlsx',
        'filepath': 'data/F Strength Reading 1000000.xlsx',
        'primary_target': 'Flexural Strength'
    },
    {
        'filename': 'T Strenth Reading 1000000.xlsx',
        'filepath': 'data/T Strenth Reading 1000000.xlsx',
        'primary_target': 'Tensile Strength'
    }
]

schema_rows = []
audit_results = []

for d in datasets:
    fname = d['filename']
    fpath = d['filepath']
    wb = openpyxl.load_workbook(fpath, data_only=False) # to check formulas
    wb_val = openpyxl.load_workbook(fpath, data_only=True) # to check values
    
    file_info = {
        'filename': fname,
        'filepath': fpath,
        'primary_target': d['primary_target'],
        'sheets': []
    }
    
    for sname in wb.sheetnames:
        sheet = wb[sname]
        sheet_val = wb_val[sname]
        
        # Raw grid
        raw_rows = list(sheet_val.iter_rows(values_only=True))
        raw_formula_rows = list(sheet.iter_rows(values_only=True))
        
        total_excel_rows = len(raw_rows)
        total_excel_cols = sheet.max_column if sheet.max_column else 0
        
        # Check formulas
        has_formulas = False
        formula_cells = []
        for r_i, r in enumerate(raw_formula_rows):
            for c_i, c in enumerate(r):
                if isinstance(c, str) and c.startswith('='):
                    has_formulas = True
                    formula_cells.append((r_i+1, c_i+1, c))
        
        # Analyze as DataFrame
        # Find header row if possible
        # We can read with pandas
        try:
            df_raw = pd.read_excel(fpath, sheet_name=sname, header=None)
        except Exception as e:
            df_raw = pd.DataFrame()
            
        sheet_info = {
            'sheet_name': sname,
            'total_rows': total_excel_rows,
            'total_cols': total_excel_cols,
            'has_formulas': has_formulas,
            'formula_sample': formula_cells[:5],
            'columns_meta': [],
            'duplicate_rows': int(df_raw.duplicated().sum()) if not df_raw.empty else 0,
            'notes': ''
        }
        
        # Detailed column audit using best-effort table interpretation
        # 1. Comp 6
        if fname.startswith('C') and sname == 'Comp 6':
            df_table = pd.read_excel(fpath, sheet_name=sname, skiprows=1)
            # Row 0 had N, BC for compressive strength
            # Actually let's check header=None
            df_comp = pd.read_excel(fpath, sheet_name=sname, header=[0, 1])
            sheet_info['nature'] = 'Raw experimental sample measurements (600 rows per mix, side-by-side Normal and Bacterial)'
            sheet_info['concrete_type'] = 'Normal Concrete & Bacterial Concrete'
            sheet_info['curing_age'] = '7, 14, 21, 28, 56, 90 Days'
            sheet_info['bacteria_species'] = 'Bacillus subtilis'
            sheet_info['bacterial_concentration'] = '1,000,000 cells/mL (10^6)'
            sheet_info['target_variables'] = 'Compressive Strength (MPa) [N and BC]'
        elif fname.startswith('F') and sname == 'Sheet1':
            sheet_info['nature'] = 'Long format experimental measurements (1200 rows: 600 Normal, 600 Bacterial)'
            sheet_info['concrete_type'] = 'Normal Concrete & Bacterial Concrete'
            sheet_info['curing_age'] = '7, 14, 21, 28, 56, 90 Days'
            sheet_info['bacteria_species'] = 'Bacillus subtilis'
            sheet_info['bacterial_concentration'] = '1,000,000 cells/mL (10^6)'
            sheet_info['target_variables'] = 'Flexural Strength (MPa)'
        elif fname.startswith('T') and sname == 'Sheet1':
            sheet_info['nature'] = 'Long format experimental measurements (1200 rows: 600 Normal, 600 Bacterial)'
            sheet_info['concrete_type'] = 'Normal Concrete & Bacterial Concrete'
            sheet_info['curing_age'] = '7, 14, 21, 28, 56, 90 Days'
            sheet_info['bacteria_species'] = 'Bacillus subtilis'
            sheet_info['bacterial_concentration'] = '1,000,000 cells/mL (10^6)'
            sheet_info['target_variables'] = 'Split Tensile Strength (MPa)'
        elif sname in ['Sheet2', 'Ten 6', 'Sheet1'] and fname.startswith(('C', 'F', 'T')) and total_excel_cols in [13, 14]:
            sheet_info['nature'] = 'Wide-format experimental sample readings matrix (Sample 1..100 across 7, 14, 21, 28, 56, 90 days for N & BC) plus summary statistics rows'
            sheet_info['concrete_type'] = 'Normal (N) & Bacterial (BC)'
            sheet_info['curing_age'] = '7, 14, 21, 28, 56, 90 Days'
            sheet_info['bacteria_species'] = 'Bacillus subtilis'
            sheet_info['bacterial_concentration'] = '1,000,000 cells/mL (10^6)'
            sheet_info['target_variables'] = d['primary_target']
        elif sname in ['c', 'f', 't']:
            sheet_info['nature'] = 'Summary / Calculated table of mean strength across varying bacterial concentrations (BC3=10^3, BC4=10^4, BC5=10^5, BC6=10^6, BC7=10^7, BC8=10^8) vs Normal Concrete (NC)'
            sheet_info['concrete_type'] = 'Normal Concrete (NC) & Bacterial Concrete (BC3..BC8)'
            sheet_info['curing_age'] = '7, 14, 21, 28, 56, 90 Days'
            sheet_info['bacteria_species'] = 'Bacillus subtilis'
            sheet_info['bacterial_concentration'] = '10^3, 10^4, 10^5, 10^6, 10^7, 10^8 cells/mL'
            sheet_info['target_variables'] = 'Compressive (c) / Flexural (f) / Tensile (t) Mean Strength'
        elif sname in ['Final Result', '105 c']:
            sheet_info['nature'] = 'Comparative summary tables with calculated percentage increases and differences'
            sheet_info['concrete_type'] = 'Normal & Bacterial Concrete'
            sheet_info['curing_age'] = '7, 14, 21, 28, 56, 90 Days'
            sheet_info['bacteria_species'] = 'Bacillus subtilis'
            sheet_info['bacterial_concentration'] = '10^3 to 10^8 cells/mL'
            sheet_info['target_variables'] = 'Compressive, Flexural, Tensile Strengths'
        elif sname == 'RCPT':
            sheet_info['nature'] = 'Durability test: Rapid Chloride Permeability Test (Coulombs) summary'
            sheet_info['concrete_type'] = 'Normal (Control) & Bacterial Concrete'
            sheet_info['curing_age'] = 'Not specified (typically 28 or 90 days)'
            sheet_info['bacteria_species'] = 'Bacillus subtilis'
            sheet_info['bacterial_concentration'] = 'Control (0), 10^3, 10^4, 10^5, 10^6, 10^7, 10^8, 10^9 cells/mL'
            sheet_info['target_variables'] = 'RCPT (Coulombs)'
        elif sname == 'Water Absorption':
            sheet_info['nature'] = 'Durability test: Water Absorption (%) summary'
            sheet_info['concrete_type'] = 'Bacterial Concrete'
            sheet_info['curing_age'] = 'Not specified (typically 28 days)'
            sheet_info['bacteria_species'] = 'Bacillus subtilis'
            sheet_info['bacterial_concentration'] = '10^3, 10^4, 10^5, 10^6, 10^7, 10^8, 10^9 cells/mL'
            sheet_info['target_variables'] = 'Water Absorption (%)'
        else:
            sheet_info['nature'] = 'Auxiliary / partial sheet'
            sheet_info['concrete_type'] = 'Normal & Bacterial Concrete'
            sheet_info['curing_age'] = 'Various'
            sheet_info['bacteria_species'] = 'Bacillus subtilis'
            sheet_info['bacterial_concentration'] = '10^6 cells/mL'
            sheet_info['target_variables'] = d['primary_target']

        # Add to audit results
        file_info['sheets'].append(sheet_info)
        
    audit_results.append(file_info)

print("Parsed metadata for", len(audit_results), "files.")
