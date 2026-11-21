"""
Step 4 — Evaluate like a security tool
Score three approaches on the same 20% test set (47,159 URLs):
  - rule baseline:  flag as phishing if URLSimilarityIndex < 100   (no machine learning)
  - full model:     Random Forest, all 50 features
  - URL-only model: Random Forest, 18 features from the URL text
For each one: confusion matrix, precision, recall, and a look at the phishing URLs it missed.

run from the repo root:  python scripts/step4_evaluate.py   (after step 3)
"""

import os
import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix

DATA_PATH = os.path.join("data", "PhiUSIIL_Phishing_URL_Dataset.csv")

df = pd.read_csv(DATA_PATH)
df["is_phishing"] = 1 - df["label"]   # PhiUSIIL: label 1 = legitimate

with open(os.path.join("exports", "models.pkl"), "rb") as f:
    saved = pickle.load(f)
test = df.loc[saved["test_idx"]].copy()
y = test["is_phishing"]

# predictions from each approach
preds = {}
preds["rule"] = (test["URLSimilarityIndex"] < 100).astype(int)
for name in ["full", "url_only"]:
    m = saved[name]["model"]
    preds[name] = pd.Series(m.predict(test[saved[name]["features"]]), index=test.index)

names = {"rule": "Rule: URLSimilarityIndex < 100", "full": "Random Forest, all features",
         "url_only": "Random Forest, URL text only"}

print("=" * 70)
print("STEP 4 — SECURITY-FOCUSED EVALUATION")
print("=" * 70)

rows = []
for key in ["rule", "full", "url_only"]:
    tn, fp, fn, tp = confusion_matrix(y, preds[key]).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    rows.append([names[key], (tp + tn) / len(y), precision, recall, fn, fp])

    print(f"\n[{names[key]}]")
    print(f"                       predicted legit   predicted phishing")
    print(f"  actual legitimate    {tn:>15,}   {fp:>18,}   <- false alarms: {fp}")
    print(f"  actual phishing      {fn:>15,}   {tp:>18,}   <- missed phishing: {fn}")

summary = pd.DataFrame(rows, columns=["approach", "accuracy", "precision", "recall", "missed", "false_alarms"])
print("\n[Side by side]")
print(summary.round(5).to_string(index=False))

print("""
[How to read this]
  Recall is the key number for a security tool: of all phishing URLs, how many did we catch?
  A miss (false negative) means a user can land on a credential-stealing page.
  A false alarm (false positive) blocks a real site: annoying, but not a breach.

  The rule baseline needs no machine learning and still scores above 99.6%. That's the honest
  context for the full model's near-perfect score: in this dataset, every legitimate URL is a
  bare https homepage with URLSimilarityIndex = 100, so the classes are easy to separate.
  The URL-only model is the more realistic test of what the URL alone can tell you.
""")

# what did each model actually miss?
print("[Phishing URLs each model missed]")
for key in ["full", "url_only"]:
    missed = test[(y == 1) & (preds[key] == 0)]
    print(f"\n  {names[key]}: {len(missed)} missed")
    for url in missed["URL"].head(8):
        print(f"    {url[:95]}")
    if len(missed) > 0:
        print(f"    share using HTTPS: {missed['IsHTTPS'].mean():.0%}, "
              f"share with no path: {(missed['URL'].str.count('/') <= 3).mean():.0%}")

print("""
  The URLs a model misses tend to look like legitimate ones: HTTPS, short, no path.
  That is exactly the pattern attackers can copy, which is why page content and
  reputation checks still matter in a real deployment.
""")
print("=" * 70)
