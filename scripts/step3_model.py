"""
Step 3 — Train the models
Train two Random Forests on the same 80/20 split:
  - full model:     all 50 features (needs the page to be downloaded)
  - URL-only model: 18 features computed from the URL text alone, without URLSimilarityIndex
                    or the dataset-derived probability scores
A one-line rule baseline is added in Step 4, so we can see how much the models really add.

run from the repo root:  python scripts/step3_model.py
"""

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = os.path.join("data", "PhiUSIIL_Phishing_URL_Dataset.csv")
EXPORT_DIR = "exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df["is_phishing"] = 1 - df["label"]   # PhiUSIIL: label 1 = legitimate

DROP_COLS = ["FILENAME", "URL", "Domain", "Title", "TLD", "label", "is_phishing"]
all_features = [c for c in df.columns if c not in DROP_COLS]

# features you can compute from the URL string itself, no page visit and no lookup tables
url_features = [
    "URLLength", "DomainLength", "IsDomainIP", "TLDLength", "NoOfSubDomain",
    "HasObfuscation", "NoOfObfuscatedChar", "ObfuscationRatio",
    "NoOfLettersInURL", "LetterRatioInURL", "NoOfDegitsInURL", "DegitRatioInURL",
    "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL", "IsHTTPS",
]

y = df["is_phishing"]

# one split for both models, so their scores are comparable
train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42, stratify=y)
print(f"train rows: {len(train_idx):,}   test rows: {len(test_idx):,}")


def train(features, name):
    # same settings for both models; only the feature list changes
    model = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=5,
                                   random_state=42, n_jobs=-1)
    model.fit(df.loc[train_idx, features], y[train_idx])
    print(f"trained {name} on {len(features)} features")
    return model


full_model = train(all_features, "full model")
url_model = train(url_features, "URL-only model")

with open(os.path.join(EXPORT_DIR, "models.pkl"), "wb") as f:
    pickle.dump({
        "full": {"model": full_model, "features": all_features},
        "url_only": {"model": url_model, "features": url_features},
        "train_idx": list(train_idx),
        "test_idx": list(test_idx),
    }, f)
print("saved exports/models.pkl")
