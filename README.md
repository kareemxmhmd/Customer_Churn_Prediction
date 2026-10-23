# E-Commerce Customer Churn Prediction

**Predict which customers are about to leave, and give retention teams a ready-to-use tool to act on it.**

[Live Demo (Streamlit)](https://customerchurnprediction-ose.streamlit.app/) · [Full Evaluation Report](reports/results.md) ·  REST API (FastAPI, docs at `/docs`)
---

## The Business Problem

Acquiring a new customer costs several times more than keeping an existing one. This project flags customers at risk of churning so a retention team can target them with offers or support before they leave.

## Results

Evaluated once on an untouched, stratified holdout test set (491 customers, 80 churners).

| Metric (Churn = 1) | Test Score | What it means |
|---|:---:|---|
| **Recall** | **80.0%** | Catches 64 of the 80 customers who churned |
| **Precision** | **70.3%** | 7 in 10 flagged customers are real churners |
| **F1-Score** | **74.9%** | Balance of the two |
| **Accuracy** | **91.2%** | Overall (less meaningful here: only 16.3% of customers churn) |

Model selection was done on a separate validation set (XGBoost PR-AUC **0.832** vs. 0.163 for a random baseline).

**Model comparison (validation set):**

| Model | Precision | Recall | F1 | PR-AUC |
|---|:---:|:---:|:---:|:---:|
| **XGBoost (class-balanced)** | 0.670 | 0.863 | 0.754 | **0.832** |
| Random Forest (balanced) | 0.757 | 0.662 | 0.707 | 0.788 |
| Logistic Regression (balanced) | 0.462 | 0.900 | 0.610 | 0.682 |

XGBoost was selected because churn is cost-asymmetric: missing a churner costs more than sending an unnecessary offer, so **recall and PR-AUC** were the main criteria.

## Key Insights

- **The first 3 months are the danger zone:** customers with 0-3 months of tenure churn at **41.1%**; after 25 months, churn is close to 0%.
- **Complaints are a strong warning sign:** customers who complained churn at **31.8%** vs. **10.2%** for those who didn't.
- **Unhappy + complained = highest risk:** churn reaches roughly **45%** for customers with a satisfaction score of 2 or lower who also filed a complaint.
- **Single customers churn at 25.8%**, more than double married customers (10.9%).
- **Mobile-category buyers churn most (26.9%)**; Grocery (4.1%) and Laptop & Accessory (9.8%) are the most loyal.

![Churn rate by tenure band](reports/figures/07_churn_rate_by_tenure_band.png)

## What's Inside

- **Interactive dashboard (Streamlit):** score a single customer (probability, Low/Medium/High risk badge, retention advice) or upload a CSV to score many customers and download the results. A threshold slider lets you trade precision against recall to fit a campaign budget.
- **REST API (FastAPI):** `/predict`, `/predict/batch`, `/health`, with automatic Swagger docs at `/docs`.
- **Reproducible pipeline:** cleaning, feature engineering, training, and inference as scripts in `src/`.

### API example

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Tenure": 2,
    "WarehouseToHome": 15,
    "NumberOfDeviceRegistered": 3,
    "PreferedOrderCat": "Mobile",
    "SatisfactionScore": 2,
    "MaritalStatus": "Single",
    "NumberOfAddress": 2,
    "Complain": 1,
    "DaySinceLastOrder": 7,
    "CashbackAmount": 120
  }'
```

```json
{
  "churn_probability": 0.87,
  "prediction": 1,
  "risk_level": "High"
}
```
<!-- TODO: replace the request/response above with a real call to your API so the field names and output match exactly -->

## Approach

1. **Data cleaning:** removed 671 duplicate rows (17%), merged inconsistent labels (`Mobile` / `Mobile Phone`), and handled missing values with median imputation plus missing-value indicators (missing `Tenure` turned out to be a useful signal).
2. **Feature engineering:** `cashback_per_month`, `days_since_order_per_month`, `complain_and_unhappy`, `tenure_band`.
3. **Imbalance handling:** class weighting (`scale_pos_weight = 5.13`) and evaluation by recall and PR-AUC instead of accuracy alone.
4. **No data leakage:** stratified 70/15/15 train/validation/test split; all imputation, scaling, and encoding live inside one scikit-learn pipeline fit on training data only.
5. **Model selection:** compared Logistic Regression, Random Forest, and XGBoost on the validation set; the test set was used once, at the end.

## Dataset

Public e-commerce customer churn dataset (3,941 rows, 11 columns; 3,269 unique customers after deduplication; 16.3% churn rate).
<!-- TODO: add the dataset source link (e.g., Kaggle) -->

## Suggested Retention Actions

Based on the findings above (illustrative, not tested in a live campaign):

| Segment | Trigger | Action |
|---|---|---|
| Early-stage customers | Tenure ≤ 3 months | Onboarding help, first-order follow-up, second-order voucher |
| Unresolved complaints | Complaint filed and satisfaction ≤ 2 | Fast customer-service escalation within 24h |
| Low incentive value | Cashback per month in bottom 25% | Loyalty tier or bonus rewards |

## Run It Locally

```bash
git clone <your-repo-url>
cd Customer_Churn_Prediction
pip install -r requirements.txt

python -m src.preprocess     # clean data + build features
python -m src.train          # train and save models/best_model.pkl
python -m src.predict        # test inference on sample customers

streamlit run app/streamlit_app.py            # dashboard
uvicorn api.main:app --port 8000 --reload     # API
```

## Project Structure

```
├── api/            FastAPI app (/predict, /predict/batch)
├── app/            Streamlit dashboard
├── data/           raw and processed data
├── models/         trained pipeline (preprocessing + XGBoost)
├── notebooks/      01_eda.ipynb, 02_modeling.ipynb
├── reports/        results.md and figures
├── src/            preprocess.py, train.py, predict.py
└── requirements.txt
```

## Limitations & Next Steps

- The test set is small (80 churners), so metrics carry some uncertainty.
- The data has no timestamps, so there is no time-based validation.
- Next: SHAP explanations per customer, probability calibration, and drift monitoring.

## Tech Stack

Python · pandas · scikit-learn · XGBoost · FastAPI · Streamlit
