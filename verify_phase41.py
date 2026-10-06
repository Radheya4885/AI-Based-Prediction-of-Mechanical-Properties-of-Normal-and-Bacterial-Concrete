import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

preds = pd.read_csv('results/predictions/phase4_1_cv_predictions.csv')
results = pd.read_csv('reports/baseline_model_comparison.csv')

print("="*65)
print("PHASE 4.1 FINAL VERIFICATION")
print("="*65)
print(f"\nPrediction rows: {len(preds):,}")
print(f"Result rows:     {len(results)}")

print("\n--- STRATEGY A: Separate (pooled over folds, per property) ---")
sep = results[results['strategy']=='separate']
print(sep[['model','feature_set','mechanical_property','mae','rmse','r2','mape']].sort_values(
    ['mechanical_property','r2'], ascending=[True,False]).to_string(index=False))

print("\n--- STRATEGY B: Unified (pooled R2 over all folds+properties) ---")
uni = results[results['strategy']=='unified'].sort_values('r2', ascending=False)
print(uni[['model','feature_set','mechanical_property','mae','rmse','r2']].to_string(index=False))

print("\n--- FIGURES ---")
for f in sorted(Path('reports/figures').glob('baseline_*.png')):
    print(f"  {f.name}: {f.stat().st_size/1024:.1f} KB")

print("\n--- OUTPUT FILES ---")
for f in ['reports/baseline_model_comparison.csv',
          'reports/baseline_modeling_report.md',
          'results/predictions/phase4_1_cv_predictions.csv',
          'results/phase4_1_run_metadata.json']:
    p = Path(f)
    print(f"  {p.name}: {p.stat().st_size/1024:.1f} KB")

print("\n[PHASE 4.1 COMPLETE]")
