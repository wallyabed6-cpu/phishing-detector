"""
Step 1 — Data Exploration
Load the CSV and understand what we're working with before touching any models.

run from the repo root:  python scripts/step1_explore.py
"""

import os
import pandas as pd

DATA_PATH = os.path.join("data", "PhiUSIIL_Phishing_URL_Dataset.csv")

df = pd.read_csv(DATA_PATH)

# in PhiUSIIL, label 1 = legitimate and 0 = phishing (UCI: 134,850 legitimate, 100,945 phishing).
# flip it into a column that reads the way we think: 1 = phishing
df["is_phishing"] = 1 - df["label"]

print("=" * 60)
print("STEP 1 — DATA EXPLORATION")
print("=" * 60)

print(f"\n[Shape]  {df.shape[0]:,} rows  x  {df.shape[1]} columns")

# missing values
missing = df.isnull().sum()
print("\n[Missing values]", "none" if missing.sum() == 0 else missing[missing > 0].to_string())

# class balance, with a check that the label meaning matches UCI's published counts
print("\n[Class balance]")
counts = df["is_phishing"].value_counts()
for val in [1, 0]:
    name = "phishing" if val == 1 else "legitimate"
    print(f"  {name:<11}: {counts[val]:>7,}  ({counts[val] / len(df):.1%})")
if counts[0] != 134850 or counts[1] != 100945:
    print("  WARNING: counts don't match UCI (134,850 legitimate / 100,945 phishing) — check the label mapping")

# spot-check the mapping with URLs a human can judge
print("\n[Spot check — random URLs from each class]")
for val, name in [(1, "phishing"), (0, "legitimate")]:
    sample = df[df["is_phishing"] == val]["URL"].sample(5, random_state=1)
    print(f"  {name}:")
    for url in sample:
        print(f"    {url[:90]}")

# how the two classes differ on a few features you can see in the URL
print("\n[Quick profile by class]")
has_path = df["URL"].str.replace(r"^https?://", "", regex=True).str.rstrip("/").str.contains("/")
df["has_path"] = has_path.astype(int)
profile = df.groupby("is_phishing")[["IsHTTPS", "has_path", "URLLength", "URLSimilarityIndex"]].mean().round(3)
profile.index = ["legitimate", "phishing"]
print(profile.to_string())

print("""
[What this tells us]
  Every legitimate URL in this dataset is a bare https:// homepage (no path), and every one
  has URLSimilarityIndex = 100. Phishing URLs vary much more. That makes the two classes very
  easy to separate, so expect near-perfect scores in Step 4 and read them with care.
""")
print("=" * 60)
