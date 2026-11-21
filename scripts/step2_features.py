"""
Step 2 — Feature Analysis
Find which features separate phishing from legitimate URLs. All percentages below are
computed from the data, not typed in by hand.

run from the repo root:  python scripts/step2_features.py
"""

import os
import pandas as pd

DATA_PATH = os.path.join("data", "PhiUSIIL_Phishing_URL_Dataset.csv")

df = pd.read_csv(DATA_PATH)
df["is_phishing"] = 1 - df["label"]   # PhiUSIIL: label 1 = legitimate

DROP_COLS = ["FILENAME", "URL", "Domain", "Title", "TLD", "label"]
features = [c for c in df.columns if c not in DROP_COLS + ["is_phishing"]]

phish = df[df["is_phishing"] == 1]
legit = df[df["is_phishing"] == 0]

print("=" * 65)
print("STEP 2 — FEATURE ANALYSIS")
print("=" * 65)

# correlation with is_phishing: positive = more common in phishing
print("\n[1] Correlation with is_phishing (strongest first)")
print("    positive = higher in phishing | negative = higher in legitimate\n")
corr = df[features].corrwith(df["is_phishing"]).sort_values(key=abs, ascending=False)
for feat, val in corr.head(15).items():
    side = "phishing  " if val > 0 else "legitimate"
    print(f"  {feat:<30} {val:+.3f}  higher in {side}")

# yes/no features: how often each appears in each class
print("\n[2] Yes/no features — % present in each class")
print(f"  {'Feature':<26} {'Phishing':>9} {'Legitimate':>11}")
binary = [c for c in features if df[c].nunique() == 2]
rows = []
for col in binary:
    p = phish[col].mean() * 100
    l = legit[col].mean() * 100
    rows.append([col, p, l])
rows = sorted(rows, key=lambda r: abs(r[1] - r[2]), reverse=True)
for col, p, l in rows:
    print(f"  {col:<26} {p:>8.1f}% {l:>10.1f}%")

# a few numeric features worth talking about
print("\n[3] Medians by class")
for col in ["URLLength", "URLSimilarityIndex", "LineOfCode", "NoOfExternalRef", "NoOfImage"]:
    print(f"  {col:<22} phishing {phish[col].median():>9.1f}   legitimate {legit[col].median():>9.1f}")

# plain-language takeaways, built from the numbers above
https_p = phish["IsHTTPS"].mean() * 100
https_l = legit["IsHTTPS"].mean() * 100
social_p = phish["HasSocialNet"].mean() * 100
social_l = legit["HasSocialNet"].mean() * 100
sim_l = (legit["URLSimilarityIndex"] == 100).mean() * 100
sim_p = (phish["URLSimilarityIndex"] == 100).mean() * 100
print(f"""
[4] Plain-language interpretation
  - HTTPS: {https_p:.0f}% of phishing URLs use HTTPS vs {https_l:.0f}% of legitimate ones.
    The padlock is common on phishing, but here every legitimate URL has it, so its absence is a strong red flag.
  - Social links: {social_l:.0f}% of legitimate pages link to social networks vs {social_p:.1f}% of phishing pages.
    Real sites have a full footer (social links, copyright, description); throwaway phishing pages don't.
  - Page size: legitimate pages are far bigger (more code, images, links). Phishing kits are thin.
  - URLSimilarityIndex: {sim_l:.0f}% of legitimate URLs score exactly 100 vs {sim_p:.1f}% of phishing.
    This one feature nearly gives the answer away (see Step 4 baseline).
""")
print("=" * 65)
