# E-Commerce Customer Churn Prediction

> **Proactive e-commerce customer retention engine powered by XGBoost, intercepting 80% of churning accounts before they defect while keeping marketing outreach budgets disciplined with 70.3% precision.**

[Live Streamlit App](https://customerchurnprediction-ose.streamlit.app/) · [Full Evaluation Report](reports/results.md) · REST API Docs (`/docs`)

---

## The Business Problem

Acquiring a new customer costs **5 to 7 times more** than retaining an existing one. In high-velocity e-commerce, customer attrition quietly erodes recurring revenue, diminishes Customer Lifetime Value (CLV), and inflates marketing Customer Acquisition Costs (CAC).

Most e-commerce platforms operate reactively—only recognizing that a customer has churned after months of purchase inactivity, by which point win-back campaigns suffer dismal single-digit conversion rates.

This project delivers a **proactive customer retention system** that:
1. Detects customer defection signals up to 90 days in advance.
2. Stratifies accounts into clear, actionable risk tiers (**Low, Medium, High**).
3. Delivers high precision (**70.3%**) so marketing budgets are not squandered on discounts for customers who would have remained loyal anyway.
4. Prescribes targeted operational playbooks (VIP customer service escalation, tailored onboarding perks, incentive restimulation).

---

## Results & Performance Highlights

Evaluated on an untouched holdout test set (491 customer accounts, 80 true churners):

| Metric (Churn = 1) | Holdout Test Score | Operational Interpretation |
| :--- | :---: | :--- |
| **Recall** | **80.0%** | Successfully intercepts **4 out of every 5** customers who intend to defect (64 / 80) |
| **Precision** | **70.3%** | 7 out of 10 targeted retention offers reach genuinely at-risk customers |
| **F1-Score** | **74.9%** | Optimal harmonic trade-off between targeting reach and budget efficiency |
| **PR-AUC** | **0.832** | Area under Precision-Recall curve ($5.1\times$ above the 0.163 naive random baseline) |
| **ROC-AUC** | **0.934** | Outstanding discriminative capability across decision thresholds |
| **Accuracy** | **91.2%** | High overall fidelity across the active customer portfolio (448 / 491 correct) |

### Confusion Matrix Breakdown (Test Set, $n = 491$)

```
                        Predicted Retained (0)    Predicted Churned (1)
Actual Retained (0)              384                       27           (93.4% specificity)
Actual Churned (1)                16                       64           (80.0% sensitivity)
```

- **True Positives (64):** Churners caught in advance for automated retention offers.
- **False Positives (27):** Loyal accounts flagged; minimal operational harm when paired with value-add service.
- **False Negatives (16):** Missed departures (only 20% of churners slipped through undetected).
- **True Negatives (384):** Satisfied, loyal customers left undisturbed.

---

## Dataset & Data Quality Audit

- **Raw Population:** 3,941 customer rows, 11 primary operational features.
- **Deduplication:** Identified and eliminated **671 exact duplicate records (17.03%)**, stabilizing the training distribution to **3,269 unique customer profiles**.
- **Class Imbalance:** Churn represents **16.3%** of the cleaned population (`scale_pos_weight = 5.13` applied).
- **Key Behavioral Dimensions:**
  - **Customer Lifecycle:** Platform tenure (months) and recency (`DaySinceLastOrder`).
  - **Friction & Satisfaction:** Customer satisfaction rating and formal complaint logs.
  - **Account & Shopping Behaviors:** Cashback incentives earned, registered devices, delivery addresses, marital status, and preferred purchase categories.

---

## Machine Learning Approach

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               MACHINE LEARNING PIPELINE                                │
├─────────────────────────┬──────────────────────────┬───────────────────────────────────┤
│ 1. Data Cleaning        │ 2. Feature Engineering   │ 3. Pipeline Architecture          │
├─────────────────────────┼──────────────────────────┼───────────────────────────────────┤
│ • Deduplicate 671 rows  │ • cashback_per_month     │ • Imputation: Median + Indicators │
│ • Missing indicator on  │ • days_since_order_p_m   │ • Categorical: OneHotEncoder      │
│   Tenure (major signal) │ • complain_and_unhappy   │ • Estimator: XGBoost Classifier   │
│ • Median imputation     │ • tenure_band (5 bins)   │   (scale_pos_weight = 5.13)       │
└─────────────────────────┴──────────────────────────┴───────────────────────────────────┘
```

1. **Feature Engineering:** Captured non-linear lifecycle thresholds via binned tenure bands (`0-3m`, `4-6m`, `7-12m`, `13-24m`, `25m+`) and interaction flags (`complain_and_unhappy`).
2. **Model Selection:** **XGBoost** outperformed Random Forest and Logistic Regression on validation PR-AUC (**0.832** vs. 0.788 and 0.682) by effectively capturing non-linear interactions between service complaints and incentive decay.

---

## What's Inside

### Interactive Streamlit Dashboard

Hosted in [`app/streamlit_app.py`](app/streamlit_app.py):
- **Real-Time Customer Scoring:** Instant probability calculation, risk tier badge, and explanation.
- **Dynamic Decision Threshold:** Custom slider to adjust precision and recall trade-offs.
- **Batch CSV Processing:** Upload entire marketing lists and export churn-scored CSVs.

### Production REST API (FastAPI)

Hosted in [`api/main.py`](api/main.py):
- `POST /predict`: Real-time single customer risk prediction.
- `POST /predict/batch`: High-throughput batch inference.
- `GET /health`: Model status and loaded configuration.

---

## Actionable Retention Playbook

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PROACTIVE RETENTION PLAYBOOK                          │
├──────────────────────┬──────────────────────┬───────────────────────────────┤
│ At-Risk Segment      │ Trigger Condition    │ Business Action Plan          │
├──────────────────────┼──────────────────────┼───────────────────────────────┤
│ 1. Early Onboarding  │ Tenure <= 3 months   │ • Dedicated onboarding assist │
│    Friction          │                      │ • First-order follow-up call  │
│                      │                      │ • 15% 2nd-order discount      │
├──────────────────────┼──────────────────────┼───────────────────────────────┤
│ 2. Unresolved        │ Complain = 1         │ • VIP CS escalation call <24h │
│    Complaints        │ & Satisfaction <= 2  │ • Automated apology credit    │
│                      │                      │ • Senior account manager ping │
├──────────────────────┼──────────────────────┼───────────────────────────────┤
│ 3. Incentive-Lapsed  │ Cashback / Month     │ • Personalized loyalty tier   │
│    Shoppers          │ is bottom 25%        │ • Accelerated reward points   │
│                      │                      │ • Exclusive weekend cashback  │
└──────────────────────┴──────────────────────┴───────────────────────────────┘
```

---

## Tech Stack

- **ML Frameworks:** Python, scikit-learn, XGBoost, pandas, NumPy
- **Web & Serving:** FastAPI, Streamlit, Uvicorn, Pydantic
- **Visualization:** Matplotlib, Seaborn
