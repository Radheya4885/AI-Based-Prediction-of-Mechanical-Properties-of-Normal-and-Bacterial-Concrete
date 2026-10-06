import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

preds = pd.read_csv('results/predictions/phase4_1_cv_predictions.csv')

print("Columns:", list(preds.columns))
print("Strategies:", preds['strategy'].unique())
print("Shape:", preds.shape)
print()

# Check what unified mechanical_property values look like
uni = preds[preds['strategy']=='unified']
print("Unified property distribution:")
print(uni['mechanical_property'].value_counts())
print()

# Manual overall R2 for unified XGBoost expanded
sub = uni[(uni['model']=='XGBoost') & (uni['feature_set']=='expanded')]
print(f"XGBoost unified expanded: {len(sub)} rows")
r2 = r2_score(sub['strength_mpa_actual'], sub['strength_mpa_pred'])
mae = mean_absolute_error(sub['strength_mpa_actual'], sub['strength_mpa_pred'])
print(f"  Overall R2={r2:.4f}  MAE={mae:.4f}")

# Per fold
for fold, grp in sub.groupby('fold'):
    r2_f = r2_score(grp['strength_mpa_actual'], grp['strength_mpa_pred'])
    print(f"  Fold {fold}: R2={r2_f:.4f}  n={len(grp)}")

print()
# Show separate strategy R2 computed directly
sep = preds[preds['strategy']=='separate']
sub2 = sep[(sep['model']=='RandomForest') & (sep['feature_set']=='core') & (sep['mechanical_property']=='Compressive Strength')]
print(f"RF core CS: {len(sub2)} rows")
r2_2 = r2_score(sub2['strength_mpa_actual'], sub2['strength_mpa_pred'])
print(f"  Overall R2={r2_2:.4f}")
for fold, grp in sub2.groupby('fold'):
    r2_f = r2_score(grp['strength_mpa_actual'], grp['strength_mpa_pred'])
    print(f"  Fold {fold}: R2={r2_f:.4f}  n={len(grp)}")
