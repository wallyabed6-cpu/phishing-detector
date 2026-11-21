"""
Step 5 — Export for Power BI
Write clean CSVs to exports/ for the dashboard in powerbi/.

  dim_model.csv           one row per approach (rule, full model, URL-only model)
  model_metrics.csv       accuracy, precision, recall, F1, counts per approach
  confusion_matrix.csv    actual vs predicted counts per approach (long format)
  feature_importance.csv  feature rankings for both models, with a feature group
  score_distribution.csv  how many test URLs fall in each risk score band, by true class
  errors.csv              every test URL a model got wrong, with its risk score

run from the repo root:  python scripts/step5_export.py   (after step 3)
"""

import os
import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix

DATA_PATH = os.path.join("data", "PhiUSIIL_Phishing_URL_Dataset.csv")
EXPORT_DIR = "exports"

df = pd.read_csv(DATA_PATH)
df["is_phishing"] = 1 - df["label"]   # PhiUSIIL: label 1 = legitimate

with open(os.path.join(EXPORT_DIR, "models.pkl"), "rb") as f:
    saved = pickle.load(f)
test = df.loc[saved["test_idx"]].copy()
y = test["is_phishing"]


def save(frame, name):
    frame.to_csv(os.path.join(EXPORT_DIR, name), index=False)
    print(f"  {name:<24} {len(frame):>6} rows")


# approaches, in the order they appear on the dashboard
models = pd.DataFrame([
    ["rule", "Rule baseline", "Flag as phishing if URLSimilarityIndex < 100. No machine learning.", 1, 1],
    ["full", "Full model", "Random Forest on all 50 features (URL + downloaded page content).", 2, 50],
    ["url_only", "URL-only model", "Random Forest on 18 features computed from the URL text alone.", 3, 18],
], columns=["model", "model_label", "description", "sort_order", "n_features"])

# predictions and risk scores (probability of phishing) for each approach
preds = {}
scores = {}
preds["rule"] = (test["URLSimilarityIndex"] < 100).astype(int)
for key in ["full", "url_only"]:
    m = saved[key]["model"]
    X = test[saved[key]["features"]]
    preds[key] = pd.Series(m.predict(X), index=test.index)
    scores[key] = pd.Series(m.predict_proba(X)[:, 1], index=test.index)

print("STEP 5 — EXPORT FOR POWER BI")

# metrics, one row per approach
metric_rows = []
cm_rows = []
for key in models["model"]:
    tn, fp, fn, tp = confusion_matrix(y, preds[key]).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    metric_rows.append([key, len(y), (tp + tn) / len(y), precision, recall,
                        2 * precision * recall / (precision + recall), tp, fp, fn, tn])
    cm_rows.append([key, "Legitimate", "Legitimate", tn])
    cm_rows.append([key, "Legitimate", "Phishing", fp])
    cm_rows.append([key, "Phishing", "Legitimate", fn])
    cm_rows.append([key, "Phishing", "Phishing", tp])
metrics = pd.DataFrame(metric_rows, columns=["model", "test_urls", "accuracy", "precision", "recall", "f1",
                                             "true_positives", "false_positives", "false_negatives", "true_negatives"])
confusion = pd.DataFrame(cm_rows, columns=["model", "actual", "predicted", "urls"])

# feature importance with a plain-language group
lookup_scores = ["URLSimilarityIndex", "CharContinuationRate", "TLDLegitimateProb", "URLCharProb"]
url_text = saved["url_only"]["features"]
imp_rows = []
for key in ["full", "url_only"]:
    m = saved[key]["model"]
    imp = pd.Series(m.feature_importances_, index=saved[key]["features"]).sort_values(ascending=False)
    rank = 1
    for feat, val in imp.items():
        group = "Page content"
        if feat in url_text:
            group = "URL text"
        if feat in lookup_scores:
            group = "Lookup score"
        imp_rows.append([key, rank, feat, round(val * 100, 2), group])
        rank += 1
importance = pd.DataFrame(imp_rows, columns=["model", "rank", "feature", "importance_pct", "feature_group"])

# risk score bands: 0.0-0.1, 0.1-0.2, ... 0.9-1.0
dist_rows = []
for key in ["full", "url_only"]:
    band = (scores[key] * 10).clip(upper=9.999).astype(int)
    for cls_value, cls_name in [(1, "Phishing"), (0, "Legitimate")]:
        counts = band[y == cls_value].value_counts()
        for b in range(10):
            label = f"{b / 10:.1f}–{(b + 1) / 10:.1f}"
            dist_rows.append([key, cls_name, b, label, int(counts.get(b, 0))])
distribution = pd.DataFrame(dist_rows, columns=["model", "true_class", "band_order", "risk_band", "urls"])

# every mistake, so the dashboard can show what each approach got wrong
err_rows = []
for key in models["model"]:
    wrong = test[preds[key] != y]
    for idx, row in wrong.iterrows():
        kind = "Missed phishing" if row["is_phishing"] == 1 else "False alarm"
        score = scores[key][idx] if key in scores else None
        err_rows.append([key, row["URL"], kind, "Phishing" if row["is_phishing"] == 1 else "Legitimate",
                         score, row["IsHTTPS"], row["URLLength"], row["URLSimilarityIndex"]])
errors = pd.DataFrame(err_rows, columns=["model", "url", "error_type", "true_class", "risk_score",
                                         "is_https", "url_length", "url_similarity_index"])

save(models, "dim_model.csv")
save(metrics, "model_metrics.csv")
save(confusion, "confusion_matrix.csv")
save(importance, "feature_importance.csv")
save(distribution, "score_distribution.csv")
save(errors, "errors.csv")

print()
print(metrics[["model", "accuracy", "precision", "recall", "false_negatives", "false_positives"]].round(5).to_string(index=False))
