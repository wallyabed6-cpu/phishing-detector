"""
Step 4 — Evaluate Like a Security Tool
Confusion matrix, precision, recall, F1, and deep false negative analysis.
"""

import pandas as pd
import numpy as np
import pickle

MODEL_PATH = r"C:\Users\wally\phishing-detector\exports\model.pkl"
DATA_SPLIT = r"C:\Users\wally\phishing-detector\exports\test_split.pkl"

with open(MODEL_PATH, 'rb') as f:
    saved = pickle.load(f)
    rf = saved['model']

with open(DATA_SPLIT, 'rb') as f:
    split = pickle.load(f)
    X_test       = split['X_test']
    y_test       = split['y_test']
    feature_names = split['feature_names']

y_pred  = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]   # risk score (0-1)

print("=" * 65)
print("STEP 4 — SECURITY-FOCUSED EVALUATION")
print("=" * 65)

# ── Confusion matrix ──────────────────────────────────────────────────────────
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"""
[Confusion Matrix]

                     PREDICTED
                  Benign   Phishing
  ACTUAL  Benign  {tn:>6}    {fp:>6}    <- True Neg / False Pos
         Phish.  {fn:>6}    {tp:>6}    <- False Neg / True Pos

  TN (correctly caught benign)    : {tn:>6,}
  FP (benign wrongly flagged)     : {fp:>6,}   <- annoyed users, wasted clicks
  FN (phishing we MISSED)         : {fn:>6,}   <- DANGEROUS: user exposed to attack
  TP (phishing correctly caught)  : {tp:>6,}
""")

# ── Metrics ───────────────────────────────────────────────────────────────────
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0
fnr       = fn / (fn + tp) if (fn + tp) > 0 else 0

print(f"[Metrics]")
print(f"  Accuracy          : {(tn+tp)/(tn+fp+fn+tp):.6f}")
print(f"  Precision         : {precision:.6f}  (of URLs flagged as phishing, how many really were?)")
print(f"  Recall            : {recall:.6f}  (of all phishing URLs, how many did we catch?)")
print(f"  F1 Score          : {f1:.6f}  (balance of precision and recall)")
print(f"  False Positive Rate: {fpr:.6f}  ({fp:,} benign URLs wrongly blocked)")
print(f"  False Negative Rate: {fnr:.6f}  ({fn:,} phishing URLs that slipped through)")

# ── Why false negatives are more costly ───────────────────────────────────────
print(f"""
[Why false negatives are the more dangerous error]

  A FALSE POSITIVE means a real website gets flagged as phishing.
    Cost: user is blocked from a legitimate site. Annoying, fixable.

  A FALSE NEGATIVE means a phishing site gets through undetected.
    Cost: user visits the page, enters credentials, gets compromised.
    In enterprise security this can mean a data breach, ransomware,
    or account takeover — costs that can reach millions of dollars.

  For a phishing detector, RECALL is the metric that matters most.
  We want to catch as many phishing URLs as possible, even if it
  means occasionally blocking a legitimate site.

  Our recall: {recall:.4f} — we caught {recall*100:.2f}% of all phishing URLs.
  We missed : {fn:,} phishing URLs out of {fn+tp:,} total in the test set.
""")

# ── False negative deep dive ──────────────────────────────────────────────────
print("[False Negative Deep Dive — the phishing URLs we missed]")

X_test_df = X_test.copy()
X_test_df['true_label']  = y_test.values
X_test_df['predicted']   = y_pred
X_test_df['risk_score']  = y_proba

false_negatives = X_test_df[
    (X_test_df['true_label'] == 1) & (X_test_df['predicted'] == 0)
]

if len(false_negatives) == 0:
    print(f"\n  Perfect recall — zero phishing URLs were missed.")
    print(f"  This is exceptional. It means every phishing URL in the test")
    print(f"  set had feature patterns distinct enough for the model to catch.")
    print(f"\n  To stress-test the model, we'll look at the lowest-confidence")
    print(f"  phishing predictions — URLs the model was least sure about.")

    # Lowest-confidence correct phishing predictions
    phishing_test = X_test_df[X_test_df['true_label'] == 1].copy()
    borderline = phishing_test.nsmallest(20, 'risk_score')

    print(f"\n  [20 phishing URLs the model was LEAST confident about]")
    print(f"  (Correctly classified but with the lowest risk scores)\n")
    cols = ['risk_score', 'URLSimilarityIndex', 'IsHTTPS', 'HasSocialNet',
            'HasPasswordField', 'URLLength', 'NoOfExternalRef', 'LineOfCode']
    cols = [c for c in cols if c in borderline.columns]
    print(borderline[cols].to_string(index=False))

    print(f"\n  What makes these hard for the model:")
    low_sim  = (borderline['URLSimilarityIndex'] < 80).sum()
    no_https = (borderline['IsHTTPS'] == 0).sum()
    no_social = (borderline['HasSocialNet'] == 0).sum()
    print(f"    - {low_sim}/20 have low URLSimilarityIndex (don't mimic a known site)")
    print(f"    - {no_https}/20 don't use HTTPS (older attack style)")
    print(f"    - {no_social}/20 have no social-net lure (non-credential attacks)")

else:
    print(f"\n  Total missed: {len(false_negatives):,} phishing URLs\n")
    cols = ['risk_score', 'URLSimilarityIndex', 'IsHTTPS', 'HasSocialNet',
            'HasPasswordField', 'URLLength', 'NoOfExternalRef', 'LineOfCode']
    cols = [c for c in cols if c in false_negatives.columns]
    print(false_negatives[cols].describe().round(3).to_string())

    print(f"\n  Comparing missed phishing to all phishing in test set:")
    all_phishing = X_test_df[X_test_df['true_label'] == 1]
    for col in ['URLSimilarityIndex', 'IsHTTPS', 'HasSocialNet', 'LineOfCode']:
        if col in X_test_df.columns:
            missed_mean = false_negatives[col].mean()
            all_mean    = all_phishing[col].mean()
            print(f"    {col:<30} missed={missed_mean:.3f}  all_phishing={all_mean:.3f}")

print("\n" + "=" * 65)
print("End of Step 4.  Say 'Step 5' to export for Power BI.")
print("=" * 65)
