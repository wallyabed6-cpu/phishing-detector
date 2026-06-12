"""
Step 1 — Data Exploration
Load the CSV and understand what we're working with before touching any models.
"""

import pandas as pd

DATA_PATH = r"C:\Users\wally\phishing-detector\data\PhiUSIIL_Phishing_URL_Dataset.csv"

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("STEP 1 — DATA EXPLORATION")
print("=" * 60)

# ── Shape ─────────────────────────────────────────────────────────────────────
print(f"\n[Shape]  {df.shape[0]:,} rows  ×  {df.shape[1]} columns")

# ── Columns ───────────────────────────────────────────────────────────────────
print(f"\n[Columns]\n  {df.columns.tolist()}")

# ── Sample rows ───────────────────────────────────────────────────────────────
print("\n[Sample rows — first 5]")
sample_cols = [c for c in df.columns if c not in ['FILENAME', 'URL', 'Domain', 'Title']]
print(df[sample_cols].head(5).to_string(index=False))

# ── Data types ────────────────────────────────────────────────────────────────
print("\n[Data types]")
for col, dtype in df.dtypes.items():
    print(f"  {col:<25} {dtype}")

# ── Missing values ────────────────────────────────────────────────────────────
print("\n[Missing values]")
missing = df.isnull().sum()
has_missing = missing[missing > 0]
if len(has_missing) == 0:
    print("  None — dataset is complete.")
else:
    print(has_missing.to_string())

# ── Class balance ─────────────────────────────────────────────────────────────
print("\n[Class balance]")
target_candidates = [c for c in df.columns if c.lower() in
                     ['class', 'label', 'phishing', 'result', 'target', 'status']]
target_col = target_candidates[0] if target_candidates else df.columns[-1]
print(f"  Target column: '{target_col}'")

counts = df[target_col].value_counts().sort_index()
totals = len(df)
for val, cnt in counts.items():
    label = "phishing" if val == 1 else "benign"
    print(f"  {val} ({label:<8}): {cnt:>6,}  ({cnt/totals:.1%})")

# ── Feature distributions ─────────────────────────────────────────────────────
print("\n[Feature distributions — numeric summary]")
feature_cols = [c for c in df.columns if c != target_col]
print(df[feature_cols].describe().round(2).to_string())

# ── Per-class means (key signal) ──────────────────────────────────────────────
print("\n[Per-class feature means — phishing (1) vs benign (0)]")
numeric_features = df[feature_cols].select_dtypes(include='number').columns.tolist()
print(df.groupby(target_col)[numeric_features].mean().round(3).to_string())

print("\n" + "=" * 60)
print("End of Step 1.  Review the output above, then say 'Step 2' to continue.")
print("=" * 60)
