"""
Step 5 — Export for Power BI
Save clean CSVs into exports/ for dashboard use.
"""

import pandas as pd
import numpy as np
import pickle, os

MODEL_PATH = r"C:\Users\wally\phishing-detector\exports\model.pkl"
DATA_SPLIT = r"C:\Users\wally\phishing-detector\exports\test_split.pkl"
DATA_PATH  = r"C:\Users\wally\phishing-detector\data\PhiUSIIL_Phishing_URL_Dataset.csv"
EXPORT_DIR = r"C:\Users\wally\phishing-detector\exports"

with open(MODEL_PATH, 'rb') as f:
    saved = pickle.load(f)
    rf = saved['model']

with open(DATA_SPLIT, 'rb') as f:
    split = pickle.load(f)
    X_test        = split['X_test']
    y_test        = split['y_test']
    X_train       = split['X_train']
    y_train       = split['y_train']
    feature_names = split['feature_names']

y_pred  = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

print("=" * 65)
print("STEP 5 — EXPORT FOR POWER BI")
print("=" * 65)

# ── (a) Feature importance ────────────────────────────────────────────────────
feat_imp = pd.DataFrame({
    'feature':    feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False).reset_index(drop=True)

feat_imp['rank']            = feat_imp.index + 1
feat_imp['importance_pct']  = (feat_imp['importance'] * 100).round(2)
feat_imp['category'] = feat_imp['feature'].apply(lambda f:
    'URL Structure'  if any(k in f for k in ['URL','Domain','TLD','Char','Digit','Letter','Special','Ampersand','QMark','Equals']) else
    'Page Content'   if any(k in f for k in ['Line','Image','CSS','JS','Ref','iFrame','Popup','Redirect','Title','Favicon','Description','Copyright','Robots','Responsive']) else
    'Security Signal'
)

path_a = os.path.join(EXPORT_DIR, 'feature_importance.csv')
feat_imp[['rank','feature','importance_pct','category']].to_csv(path_a, index=False)
print(f"\n[a] feature_importance.csv  ({len(feat_imp)} rows)")
print(feat_imp[['rank','feature','importance_pct','category']].head(15).to_string(index=False))

# ── (b) Model performance metrics ────────────────────────────────────────────
from sklearn.metrics import confusion_matrix
cm   = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
accuracy  = (tp + tn) / len(y_test)

metrics = pd.DataFrame([
    {'metric': 'Total URLs Tested',        'value': len(y_test),   'notes': 'Test set size (20% of full dataset)'},
    {'metric': 'Phishing URLs in Test',    'value': int(tp + fn),  'notes': 'Ground-truth phishing count'},
    {'metric': 'Benign URLs in Test',      'value': int(tn + fp),  'notes': 'Ground-truth benign count'},
    {'metric': 'True Positives',           'value': int(tp),       'notes': 'Phishing correctly detected'},
    {'metric': 'True Negatives',           'value': int(tn),       'notes': 'Benign correctly cleared'},
    {'metric': 'False Positives',          'value': int(fp),       'notes': 'Benign wrongly flagged (annoying)'},
    {'metric': 'False Negatives',          'value': int(fn),       'notes': 'Phishing missed (dangerous)'},
    {'metric': 'Accuracy',                 'value': round(accuracy,   6), 'notes': 'Overall correct predictions'},
    {'metric': 'Precision',                'value': round(precision,  6), 'notes': 'Of flagged phishing, how many were real'},
    {'metric': 'Recall',                   'value': round(recall,     6), 'notes': 'Of all phishing, how many we caught'},
    {'metric': 'F1 Score',                 'value': round(f1,         6), 'notes': 'Harmonic mean of precision and recall'},
    {'metric': 'False Negative Rate',      'value': round(fn/(fn+tp) if (fn+tp)>0 else 0, 6), 'notes': 'Miss rate — key security metric'},
    {'metric': 'False Positive Rate',      'value': round(fp/(fp+tn) if (fp+tn)>0 else 0, 6), 'notes': 'False alarm rate'},
])

path_b = os.path.join(EXPORT_DIR, 'model_metrics.csv')
metrics.to_csv(path_b, index=False)
print(f"\n[b] model_metrics.csv  ({len(metrics)} rows)")
print(metrics.to_string(index=False))

# ── (c) Sample predictions with risk scores ───────────────────────────────────
df_full = pd.read_csv(DATA_PATH)
DROP_COLS = ['FILENAME', 'Domain', 'Title', 'TLD']
df_full = df_full.drop(columns=[c for c in DROP_COLS if c in df_full.columns])

test_idx = X_test.index
sample_df = df_full.loc[test_idx].copy()
sample_df['true_label']       = y_test.values
sample_df['predicted_label']  = y_pred
sample_df['risk_score']       = y_proba.round(4)
sample_df['true_class']       = sample_df['true_label'].map({0: 'Benign', 1: 'Phishing'})
sample_df['predicted_class']  = sample_df['predicted_label'].map({0: 'Benign', 1: 'Phishing'})
sample_df['correct']          = (sample_df['true_label'] == sample_df['predicted_label'])
sample_df['result_type']      = sample_df.apply(lambda r:
    'True Positive'  if r['true_label']==1 and r['predicted_label']==1 else
    'True Negative'  if r['true_label']==0 and r['predicted_label']==0 else
    'False Positive' if r['true_label']==0 and r['predicted_label']==1 else
    'False Negative', axis=1)

# Sample: 500 phishing + 500 benign + all errors for completeness
phishing_sample = sample_df[sample_df['true_label'] == 1].sample(500, random_state=42)
benign_sample   = sample_df[sample_df['true_label'] == 0].sample(500, random_state=42)
errors          = sample_df[sample_df['correct'] == False]
combined        = pd.concat([phishing_sample, benign_sample, errors]).drop_duplicates()

export_cols = ['URL', 'true_class', 'predicted_class', 'result_type',
               'risk_score', 'URLSimilarityIndex', 'IsHTTPS',
               'HasSocialNet', 'HasPasswordField', 'URLLength',
               'NoOfExternalRef', 'LineOfCode', 'HasSubmitButton',
               'HasHiddenFields', 'HasExternalFormSubmit']
export_cols = [c for c in export_cols if c in combined.columns]

path_c = os.path.join(EXPORT_DIR, 'sample_predictions.csv')
combined[export_cols].to_csv(path_c, index=False)
print(f"\n[c] sample_predictions.csv  ({len(combined)} rows)")
print(f"    Breakdown:")
print(combined['result_type'].value_counts().to_string())

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"""
[All exports saved to: {EXPORT_DIR}]
  feature_importance.csv   -> Power BI bar chart: top features by importance
  model_metrics.csv        -> Power BI KPI cards: accuracy, recall, FP/FN counts
  sample_predictions.csv   -> Power BI table/scatter: URLs with risk scores

[Power BI tips]
  - Use model_metrics.csv for KPI cards (recall, false negatives, F1)
  - Use feature_importance.csv for a sorted bar chart, colored by category
  - Use sample_predictions.csv for a scatter plot (risk_score vs URLLength)
    colored by result_type to visualize where errors cluster
  - Add a slicer on true_class or result_type to filter the URL table
""")

print("=" * 65)
print("End of Step 5.  Say 'Step 6' for the README.")
print("=" * 65)
