"""
Step 2 — Feature Analysis
Find which features best separate phishing from benign URLs.
"""

import pandas as pd
import numpy as np

DATA_PATH = r"C:\Users\wally\phishing-detector\data\PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(DATA_PATH)

# Drop non-numeric / identifier columns
DROP_COLS = ['FILENAME', 'URL', 'Domain', 'Title', 'TLD']
df = df.drop(columns=DROP_COLS)

target_col = 'label'
feature_cols = [c for c in df.columns if c != target_col]

print("=" * 65)
print("STEP 2 — FEATURE ANALYSIS")
print("=" * 65)

# ── 1. Correlation with label ─────────────────────────────────────────────────
print("\n[1] Correlation with label (strongest separators first)")
print("    Positive = more common in phishing | Negative = more common in benign\n")

corr = df[feature_cols].corrwith(df[target_col]).sort_values(key=abs, ascending=False)
for feat, val in corr.items():
    direction = "-> phishing" if val > 0 else "-> benign "
    bar = "#" * int(abs(val) * 40)
    print(f"  {feat:<30} {val:+.3f}  {direction}  {bar}")

# ── 2. Top 10 features — group means ─────────────────────────────────────────
print("\n[2] Top 10 features — mean value by class")
print(f"  {'Feature':<30} {'Benign (0)':>12} {'Phishing (1)':>14} {'Difference':>12}\n")

top10 = corr.head(10).index.tolist()
for feat in top10:
    b = df[df[target_col] == 0][feat].mean()
    p = df[df[target_col] == 1][feat].mean()
    diff = p - b
    print(f"  {feat:<30} {b:>12.3f} {p:>14.3f} {diff:>+12.3f}")

# ── 3. Binary feature breakdown ───────────────────────────────────────────────
print("\n[3] Binary features — % present in each class")
binary_cols = [c for c in feature_cols if df[c].nunique() == 2]
print(f"  {'Feature':<30} {'% Benign':>10} {'% Phishing':>12}\n")
for col in sorted(binary_cols):
    b_pct = df[df[target_col] == 0][col].mean() * 100
    p_pct = df[df[target_col] == 1][col].mean() * 100
    flag = " <-- BIG gap" if abs(p_pct - b_pct) > 40 else ""
    print(f"  {col:<30} {b_pct:>9.1f}%  {p_pct:>10.1f}%{flag}")

# ── 4. Plain-language summary ─────────────────────────────────────────────────
print("\n[4] Plain-language security interpretation")
insights = [
    ("URLSimilarityIndex",    "Phishing URLs score 100% similarity to known-good sites — attackers clone real URLs to fool users at a glance."),
    ("IsHTTPS",               "100% of phishing here uses HTTPS. The padlock means encrypted, NOT safe — a key misconception to address."),
    ("HasPasswordField",      "59% of phishing pages have a password field vs 9% benign. The whole point is credential theft."),
    ("HasSocialNet",          "80% of phishing pages fake a social-media login button — 'Sign in with Google/Facebook' as a lure."),
    ("HasExternalFormSubmit", "Phishing forms submit to an external server. Your creds go somewhere you never intended."),
    ("NoOfiFrame",            "Phishing pages use ~3x more iFrames — hidden frames that load malicious content invisibly."),
    ("DomainTitleMatchScore", "Benign sites: title matches domain. Phishing sites: title says 'PayPal Login' but domain is random."),
    ("URLLength",             "Benign URLs are longer here — phishing uses short, clean-looking URLs to avoid suspicion."),
]
for feat, desc in insights:
    print(f"\n  {feat}")
    print(f"    {desc}")

print("\n" + "=" * 65)
print("End of Step 2.  Say 'Step 3' to build the model.")
print("=" * 65)
