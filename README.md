# Phishing URL Detector

A machine learning pipeline that detects phishing URLs with near-perfect accuracy using the PhiUSIIL dataset. Built as a portfolio project combining data analytics, cybersecurity, and machine learning.

**[View Power BI Dashboard](https://app.powerbi.com/view?r=eyJrIjoiMzkzYmFmZmMtYTRhYy00YTUxLTg2MjQtZjQzOWU1OGQ3ODk0IiwidCI6IjYwOTU2ODg0LTEwYWQtNDBmYS04NjNkLTRmMzJjMWUzYTM3YSIsImMiOjF9)**

---

## The Problem

Phishing attacks are the most common entry point for data breaches. Attackers clone legitimate websites, register convincing domains, and trick users into submitting credentials. A reliable URL classifier can serve as an automated first line of defense — flagging malicious links before a user ever clicks them.

---

## Dataset

**PhiUSIIL Phishing URL Dataset**
- 235,795 labeled URLs (57% phishing, 43% benign)
- 56 engineered features across two categories:
  - **URL-level features** — analyzable without visiting the page (length, HTTPS, digit ratio, special characters, domain similarity)
  - **Page-level features** — extracted after loading the page (JavaScript count, external references, password fields, social login buttons, iFrames)
- No missing values
- Source: peer-reviewed academic dataset, widely cited in cybersecurity research

---

## Project Structure

```
phishing-detector/
├── data/                          # Raw dataset
├── scripts/
│   ├── step1_explore.py           # Data exploration and class balance
│   ├── step2_features.py          # Feature analysis and security interpretation
│   ├── step3_model.py             # Model training (Random Forest + Logistic Regression)
│   ├── step4_evaluate.py          # Security-focused evaluation and false negative analysis
│   └── step5_export.py            # Power BI exports
├── exports/
│   ├── feature_importance.csv     # Feature rankings for dashboard
│   ├── model_metrics.csv          # Performance KPIs for dashboard
│   └── sample_predictions.csv     # Scored URLs with risk scores for dashboard
└── README.md
```

---

## Approach

### Step 1 — Explore
Loaded and profiled the dataset. Checked class balance, feature distributions, missing values, and data types. Confirmed the dataset was clean and well-balanced — no resampling needed.

### Step 2 — Feature Analysis
Calculated correlation of every feature with the phishing label. Key findings:

| Feature | Insight |
|---|---|
| `URLSimilarityIndex` | Strongest signal (r=0.86) — phishing URLs perfectly clone legitimate site structure |
| `HasSocialNet` | 80% of phishing pages fake a social login button; only 0.5% of benign pages do |
| `IsHTTPS` | 100% of phishing uses HTTPS — the padlock icon does not mean safe |
| `HasPasswordField` | Phishing pages are built for credential theft; benign pages rarely have login forms |
| `HasExternalFormSubmit` | Phishing forms send data to a third-party server |
| `URLLength` | Shorter URLs are more suspicious — modern attackers use clean, minimal URLs |

### Step 3 — Model
Trained two classifiers on an 80/20 stratified train/test split:

- **Logistic Regression** (baseline): 99.99% accuracy
- **Random Forest** (main model): 100.00% accuracy, 100 trees, max depth 20

Random Forest was chosen for its ability to handle mixed feature types, resistance to overfitting, and built-in feature importance scores.

### Step 4 — Security Evaluation
Evaluated using security-relevant metrics, not just accuracy:

| Metric | Score |
|---|---|
| Accuracy | 99.998% |
| Precision | 99.996% |
| **Recall** | **100.000%** |
| F1 Score | 99.998% |
| **False Negatives** | **0** — zero phishing URLs missed |
| False Positives | 1 — one benign URL wrongly flagged |

**Why recall matters more than accuracy:** A false negative means a phishing URL reaches the user — potential credential theft, account takeover, or ransomware. A false positive means a legitimate site gets blocked — an inconvenience, not a breach. For a security tool, minimizing false negatives is the primary objective.

Stress-tested the model by analyzing the 20 lowest-confidence correct phishing predictions. These were minimal HTTPS pages with clean URLs and no social login lures — consistent with malware delivery rather than credential harvesting, and the hardest attack type to detect with page-structure features alone.

### Step 5 — Power BI Export
Exported three dashboard-ready CSVs:
- **feature_importance.csv** — ranked features with category labels for bar charts
- **model_metrics.csv** — KPIs with plain-English descriptions for card visuals
- **sample_predictions.csv** — 1,001 scored URLs with true label, predicted label, risk score, and key features for scatter plots and filterable tables

---

## Results

The Random Forest classifier achieved **perfect recall** on the 47,159-URL test set — every phishing URL was detected, with only one false positive across 20,189 benign URLs.

The most predictive features were page-level signals (`URLSimilarityIndex`, `NoOfExternalRef`, `LineOfCode`, `NoOfSelfRef`) rather than URL-only features. This has a practical implication: URL-only checks work as a fast first filter, but full page inspection is required for high-confidence classification.

---

## Key Security Takeaway

Modern phishing sites are sophisticated. They use HTTPS, short clean URLs, responsive design, and copied branding — none of the obvious tells from a decade ago. What gives them away is behavioral: they request credentials, submit forms to external servers, embed social login buttons, and their page structure differs measurably from real sites. Feature engineering that captures these behavioral signals dramatically outperforms URL-only heuristics.

---

## Setup

```bash
pip install pandas scikit-learn
```

Run steps in order:
```bash
py scripts/step1_explore.py
py scripts/step2_features.py
py scripts/step3_model.py
py scripts/step4_evaluate.py
py scripts/step5_export.py
```

---

## Tech Stack

- **Python 3.13** — pandas, scikit-learn
- **Algorithm** — Random Forest (100 estimators, max depth 20)
- **Dataset** — PhiUSIIL Phishing URL Dataset (235,795 URLs, 56 features)
- **Visualization** — Power BI (exports in `exports/`)

---

## About

Built by a Business IT student at Virginia Tech as a portfolio project combining data analytics and cybersecurity. The goal was to build a production-realistic phishing detector — one evaluated on security metrics (recall, false negatives) rather than just accuracy, with results exportable to a business intelligence dashboard.
