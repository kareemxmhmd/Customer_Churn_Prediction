# E-Commerce Customer Churn Prediction — Final Results & Evaluation Report

## 1. Executive Summary & Key Performance Indicators

Customer attrition is one of the most critical operational risks for e-commerce platforms. Acquiring a new customer costs 5 to 7 times more than retaining an existing one. This project implements an end-to-end, production-ready Machine Learning system that identifies at-risk customers early, allowing retention teams to deploy targeted promotions and proactive customer service.

| Key Metric | Holdout Test Score | Business Significance |
| :--- | :---: | :--- |
| **Accuracy** | **91.2%** | High overall fidelity across the customer portfolio |
| **Recall (Churn = 1)** | **80.0%** | Successfully intercepts **4 out of every 5** customers who intend to churn |
| **Precision (Churn = 1)** | **70.3%** | 7 out of 10 targeted retention offers reach genuinely at-risk customers |
| **F1-Score (Positive)** | **74.9%** | Optimal harmonic trade-off between targeting reach and budget efficiency |
| **PR-AUC** | **0.832** | Area under Precision-Recall Curve (benchmark: random baseline = 0.163) |
| **ROC-AUC** | **0.934** | Outstanding discriminative ability across decision boundaries |

---

## 2. Dataset Overview & Data Quality Audit

- **Raw Data Volume:** 3,941 customer rows, 11 primary features.
- **Deduplication:** Identified and removed **671 exact duplicate records (17.03%)**, stabilizing the training distribution to **3,269 unique customer profiles**.
- **Class Imbalance:** Churn represents **16.3%** of the cleaned population (`scale_pos_weight = 5.13` was computed and applied to penalize false negatives).
- **Missing Value Strategy:**
  - `Tenure` (194 missing): Imputed with median + **missing indicator** (turned out to be a major predictive signal).
  - `WarehouseToHome` (169 missing) & `DaySinceLastOrder` (213 missing): Median imputation.

![Customer Churn Distribution](figures/01_churn_distribution.png)

---

## 3. Exploratory Data Analysis & Behavioral Drivers

Comprehensive exploratory data analysis revealed distinct behavioral patterns differentiating loyal customers from churners:

### 3.1. Customer Tenure (The Onboarding Hazard Zone)
The first 90 days are the most perilous phase of the customer lifecycle:
- Customers with **0–3 months of tenure** churn at a staggering **41.1%**.
- After 12 months, churn drops below **12%**.
- Customers with **25+ months** demonstrate virtually **0% churn**.

![Churn Rate by Tenure Band](figures/07_churn_rate_by_tenure_band.png)

### 3.2. Customer Dissatisfaction & Complaint Escalation
A customer complaint is the single strongest behavioral red flag:
- Customers who filed a recent complaint churn at **31.8%**, compared to only **10.2%** for those without complaints.
- When an unsatisfied customer (`SatisfactionScore <= 2`) files a complaint, the churn probability reaches **45%+**.

![Satisfaction and Complaint Heatmap](figures/08_churn_satisfaction_complaint_heatmap.png)

### 3.3. Product Category & Demographics
- **Device & Category:** Customers primarily ordering `Mobile` devices exhibit the highest churn rate (**26.9%**), whereas `Grocery` (4.1%) and `Laptop & Accessory` (9.8%) show high brand loyalty.
- **Marital Status:** Single customers churn at **25.8%**, more than double the churn rate of married customers (**10.9%**).

![Churn Rate by Categories](figures/06_churn_rate_by_categories.png)

### 3.4. Correlation Analysis
Feature correlations confirm that tenure, cashback incentives, complaints, and satisfaction are the primary linear and non-linear drivers of retention.

![Correlation Matrix](figures/09_correlation_matrix.png)

---

## 4. Feature Engineering Pipeline

To capture non-linear customer dynamics and prevent data leakage, domain-specific features were engineered prior to the ML pipeline:

1. **`cashback_per_month`:** `CashbackAmount / (Tenure + 1)` — Measures the velocity and density of financial incentives relative to customer lifetime.
2. **`days_since_order_per_month`:** `DaySinceLastOrder / (Tenure + 1)` — Captures dormancy velocity.
3. **`complain_and_unhappy`:** `(Complain == 1) & (SatisfactionScore <= 2)` — High-risk interaction flag.
4. **`tenure_band`:** Segmented bins `['0-3m', '4-6m', '7-12m', '13-24m', '25m+']` to allow tree splits on non-linear lifecycle thresholds.
5. **Sklearn Pipeline Integration:** All feature transformations and scaling are encapsulated inside an `sklearn.pipeline.Pipeline` with strict separation between training, validation, and testing partitions.

---

## 5. Model Evaluation & Benchmark Comparison

We evaluated three candidate architectures using a **Stratified 70% Train (2,288), 15% Validation (490), and 15% Test (491)** split. Because churn detection is cost-asymmetric (missing a churner costs substantially more than sending a promotional coupon to a loyal customer), **Recall** and **PR-AUC** were the primary selection criteria.

### Validation Set Performance Benchmark:

| Model Architecture | Precision | Recall | F1-Score | PR-AUC | Selection Verdict |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **XGBoost (Class Balanced)** | **0.670** | **0.863** | **0.754** | **0.832** | **Selected for Production** |
| Random Forest (Balanced) | 0.757 | 0.662 | 0.707 | 0.788 | Sub-optimal recall (misses 34% of churners) |
| Logistic Regression (Balanced) | 0.462 | 0.900 | 0.610 | 0.682 | Severe false positive rate (Precision < 50%) |

![Validation Metrics Comparison](figures/10_model_comparison_metrics.png)

![Precision-Recall Curves](figures/11_precision_recall_curves.png)

**Why XGBoost Was Chosen:**
As illustrated in the Precision-Recall curves above, **XGBoost dominates across all operating thresholds**, achieving the highest area under the curve (`PR-AUC = 0.832`). It successfully balances high recall without degrading precision into unacceptable territory.

---

## 6. Final Holdout Test Set Evaluation

The final XGBoost model was evaluated **once** on the untouched holdout test set (491 records, 80 true churners):

```
              precision    recall  f1-score   support

    Stay (0)      0.960     0.934     0.947       411
   Churn (1)      0.703     0.800     0.749        80

    accuracy                          0.912       491
   macro avg      0.832     0.867     0.848       491
weighted avg      0.918     0.912     0.915       491
```

### Confusion Matrix Breakdown:
- **True Positives (Caught Churners): 64** — 80% of all departing customers are flagged in time for retention action.
- **True Negatives (Correct Stays): 384** — 93.4% of loyal customers are not disturbed with unnecessary retention friction.
- **False Positives (Unnecessary Alerts): 27** — Acceptable marketing cost of proactive engagement.
- **False Negatives (Missed Churners): 16** — Only 16 out of 80 churners slipped through undetected.

![Test Confusion Matrix](figures/13_test_confusion_matrix.png)

---

## 7. Feature Importance Analysis

Analyzing the internal feature weights of the trained XGBoost model reveals the primary factors driving churn risk:

![Top Feature Importances](figures/14_feature_importance.png)

### Detailed Analysis of Key Drivers:
1. **`tenure_band_25m+` & `tenure_band_0-3m` (Top 2 Drivers):**
   - The tenure band is by far the strongest signal. Being past 25 months acts as an almost absolute anchor of loyalty, whereas being under 3 months is the sharpest churn trigger.
2. **`missingindicator_Tenure`:**
   - Customers with missing tenure data exhibit significantly elevated churn risk, representing quick drop-offs during initial signup or profile setup.
3. **`cashback_per_month` & `CashbackAmount`:**
   - Financial incentive density directly impacts retention. Customers receiving less perceived cashback relative to their time on the platform perceive less value and churn quickly.
4. **`Complain` & `complain_and_unhappy`:**
   - An unresolved customer complaint is an immediate catalyst for abandonment. The engineered interaction feature between dissatisfaction and complaint ranks among the top predictive signals.
5. **`PreferedOrderCat_Mobile`:**
   - High churn volatility in mobile devices indicates strong price-sensitivity and competitive shopping behavior.

---

## 8. Strategic Business Retention Playbook

Based on the model's feature importances and empirical findings, retention campaigns should be segmented into three high-ROI business initiatives:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PROACTIVE RETENTION PLAYBOOK                          │
├──────────────────────┬──────────────────────┬───────────────────────────────┤
│ At-Risk Segment      │ Trigger Condition    │ Business Action Plan          │
├──────────────────────┼──────────────────────┼───────────────────────────────┤
│ 1. Early-Stage       │ Tenure <= 3 months   │ • Dedicated onboarding assist │
│    Onboarding        │                      │ • First-order follow-up check │
│                      │                      │ • 15% 2nd-order voucher       │
├──────────────────────┼──────────────────────┼───────────────────────────────┤
│ 2. Unresolved        │ Complain = 1         │ • VIP CS escalation call <24h │
│    Complaints        │ & Satisfaction <= 2  │ • Automated apology gift card │
│                      │                      │ • Senior account manager ping │
├──────────────────────┼──────────────────────┼───────────────────────────────┤
│ 3. Incentive-Lapsed  │ Cashback / Month     │ • Personalized loyalty tier   │
│    Shoppers          │ is bottom 25%        │ • Accelerated reward points   │
│                      │                      │ • Exclusive weekend cashback  │
└──────────────────────┴──────────────────────┴───────────────────────────────┘
```

---

## 9. Artifacts & Reproducibility Guide

All figures and model artifacts are versioned and reproducible:
- **Figures Directory:** [reports/figures/](figures/) (14 figures exported at 300 DPI)
- **Model Pipeline:** `models/best_model.pkl`
- **Data Preprocessing & Training:** `python -m src.train`
- **Inference Engine:** `python -m src.predict`
- **Interactive Dashboard:** `streamlit run app/streamlit_app.py`
