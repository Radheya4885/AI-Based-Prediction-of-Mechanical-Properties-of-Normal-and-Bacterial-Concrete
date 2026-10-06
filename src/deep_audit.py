import openpyxl
import pandas as pd
import numpy as np
import glob
import os

files = {
    'C Strength': 'data/C Strenth Reading 1000000.xlsx',
    'F Strength': 'data/F Strength Reading 1000000.xlsx',
    'T Strength': 'data/T Strenth Reading 1000000.xlsx'
}

for label, fpath in files.items():
    print(f"\n{'='*70}\nFILE: {label} ({fpath})\n{'='*70}")
    wb = openpyxl.load_workbook(fpath, data_only=True)
    for sname in wb.sheetnames:
        sheet = wb[sname]
        max_r = sheet.max_row
        max_c = sheet.max_column
        print(f"\n--- Sheet: '{sname}' | Openpyxl max_row: {max_r}, max_col: {max_c} ---")
        
        # Read all rows as values
        rows = list(sheet.iter_rows(values_only=True))
        non_empty_rows = [r for r in rows if any(cell is not None for cell in r)]
        print(f"Non-empty rows count: {len(non_empty_rows)}")
        
        # Print first 10 non-empty rows
        print("First 8 non-empty rows:")
        for idx, r in enumerate(non_empty_rows[:8]):
            # truncate row cells for printing
            cells_repr = [str(c) if c is not None else "" for c in r[:15]]
            print(f"  Row {idx+1}: {cells_repr}")
