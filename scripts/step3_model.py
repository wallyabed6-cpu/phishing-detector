"""
Step 3 — Build and Train the Model
Train a Random Forest classifier and save it for Step 4 evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle, os

DATA_PATH  = r"C:\Users\wally\phishing-detector\data\PhiUSIIL_Phishing_URL_Dataset.csv"
MODEL_PATH = r"C:\Users\wally\phishing-detector\exports\model.pkl"
DATA_SPLIT = r"C:\Users\wally\phishing-detector\exports\test_split.pkl"

os.makedirs(r"C:\Users\wally\phishing-detector\exports", exist_ok=True)

df = pd.read_csv(DATA_PATH)

DROP_COLS = ['FILENAME', 'URL', 'Domain', 'Title', 'TLD']
df = df.drop(columns=DROP_COLS)

X = df.drop(columns=['label'])
y = df['label']

print("=" * 65)
print("STEP 3 — MODEL BUILDING")
print("=" * 65)

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n[Split]")
print(f"  Training set : {len(X_train):,} rows")
print(f"  Test set     : {len(X_test):,}  rows")
print(f"  Stratified   : yes (class ratio preserved in both sets)")
print(f"\n  Train class balance: {y_train.value_counts(normalize=True).to_dict()}")
print(f"  Test  class balance: {y_test.value_counts(normalize=True).to_dict()}")

# ── Why Random Forest? ────────────────────────────────────────────────────────
print("""
[Why Random Forest?]
  Random Forest builds many decision trees on random subsets of data,
  then votes across all trees. For this project it's a good fit because:
    - Handles a mix of binary and continuous features without scaling
    - Gives feature importances (useful for Step 4 and Power BI)
    - Hard to overfit compared to a single decision tree
    - Interpretable: you can trace any prediction through the trees
  We'll also compare against Logistic Regression as a simpler baseline.
""")

# ── Logistic Regression baseline ─────────────────────────────────────────────
print("[Logistic Regression — baseline]")
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_sc, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test_sc))
print(f"  Accuracy: {lr_acc:.4f} ({lr_acc*100:.2f}%)")
print(f"  (Accuracy alone is not enough — Step 4 will dig deeper)")

# ── Random Forest ─────────────────────────────────────────────────────────────
print("\n[Random Forest — main model]  (training, this may take ~30 sec...)")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
print(f"  Accuracy: {rf_acc:.4f} ({rf_acc*100:.2f}%)")

# ── Feature importances ───────────────────────────────────────────────────────
print("\n[Top 15 most important features — Random Forest]")
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)
for feat, score in importances.head(15).items():
    bar = "#" * int(score * 200)
    print(f"  {feat:<35} {score:.4f}  {bar}")

# ── Save model + test data ────────────────────────────────────────────────────
with open(MODEL_PATH, 'wb') as f:
    pickle.dump({'model': rf, 'scaler': scaler, 'lr': lr}, f)

with open(DATA_SPLIT, 'wb') as f:
    pickle.dump({'X_test': X_test, 'y_test': y_test,
                 'X_train': X_train, 'y_train': y_train,
                 'feature_names': X.columns.tolist()}, f)

print(f"\n[Saved]  model  -> exports/model.pkl")
print(f"[Saved]  test split -> exports/test_split.pkl")

print("\n" + "=" * 65)
print("End of Step 3.  Say 'Step 4' to evaluate like a security tool.")
print("=" * 65)
