import io
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
for directory in [PROJECT_ROOT, SRC_DIR]:
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from src.predict import load_model, predict_batch, predict_single

@st.cache_resource(show_spinner="Loading machine learning model pipeline...")
def get_model():
    """Cache loaded model in memory to avoid redundant disk I/O on every interaction."""
    return load_model()

st.set_page_config(
    page_title="E-commerce Churn Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .metric-card {
        padding: 1.2rem;
        border-radius: 0.8rem;
        background-color: #f8f9fa;
        border-left: 5px solid #007bff;
        margin-bottom: 1rem;
    }
    .badge-high {
        background-color: #dc3545;
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 0.4rem;
        font-weight: bold;
    }
    .badge-med {
        background-color: #ffc107;
        color: #212529;
        padding: 0.4rem 0.8rem;
        border-radius: 0.4rem;
        font-weight: bold;
    }
    .badge-low {
        background-color: #28a745;
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 0.4rem;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.subheader("System Status")
try:
    _ = get_model()
    st.sidebar.success("Model: Loaded & Ready (Standalone)")
except Exception as e:
    st.sidebar.error(f"Model load error: {e}")

st.sidebar.subheader("Prediction Settings")
threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
    help="Customers with predicted churn probability >= threshold are flagged as 'Churn'.",
)

# Header
st.title("Customer Churn Prediction Dashboard")
st.markdown(
    "Predict which e-commerce customers are likely to churn before they leave, allowing proactive retention campaigns."
)

tab_single, tab_batch = st.tabs(["Single Customer Prediction", "Batch CSV Prediction"])

# ==========================================================
# TAB 1: SINGLE CUSTOMER PREDICTION
# ==========================================================
with tab_single:
    st.subheader("Customer Profile Details")

    with st.form("customer_form"):
        col1, col2 = st.columns(2)

        with col1:
            tenure = st.number_input(
                "Tenure (months with company)",
                min_value=0.0,
                max_value=120.0,
                value=4.0,
                step=1.0,
                help="Duration of customer relationship in months.",
            )
            warehouse_to_home = st.number_input(
                "Distance: Warehouse to Home (km)",
                min_value=0.0,
                max_value=200.0,
                value=14.0,
                step=1.0,
            )
            devices_registered = st.number_input(
                "Number of Devices Registered",
                min_value=1,
                max_value=10,
                value=3,
                step=1,
            )
            order_cat = st.selectbox(
                "Preferred Order Category",
                ["Laptop & Accessory", "Mobile", "Fashion", "Grocery", "Others"],
                index=1,
            )
            satisfaction = st.slider(
                "Customer Satisfaction Score",
                min_value=1,
                max_value=5,
                value=2,
                help="1 = Very Unsatisfied, 5 = Highly Satisfied",
            )

        with col2:
            marital_status = st.selectbox(
                "Marital Status",
                ["Single", "Married", "Divorced"],
                index=0,
            )
            num_addresses = st.number_input(
                "Number of Addresses Added",
                min_value=1,
                max_value=25,
                value=2,
                step=1,
            )
            complain_flag = st.selectbox(
                "Has Filed a Complaint Recently?",
                ["No (0)", "Yes (1)"],
                index=1,
            )
            days_since_order = st.number_input(
                "Days Since Last Order",
                min_value=0.0,
                max_value=365.0,
                value=4.0,
                step=1.0,
            )
            cashback = st.number_input(
                "Average Cashback Received ($)",
                min_value=0.0,
                max_value=1000.0,
                value=145.0,
                step=5.0,
            )

        submitted = st.form_submit_button("Predict Churn Risk", use_container_width=True)

    if submitted:
        customer_payload = {
            "Tenure": float(tenure),
            "WarehouseToHome": float(warehouse_to_home),
            "NumberOfDeviceRegistered": int(devices_registered),
            "PreferedOrderCat": str(order_cat),
            "SatisfactionScore": int(satisfaction),
            "MaritalStatus": str(marital_status),
            "NumberOfAddress": int(num_addresses),
            "Complain": 1 if "Yes" in complain_flag else 0,
            "DaySinceLastOrder": float(days_since_order),
            "CashbackAmount": float(cashback),
        }

        result = None
        with st.spinner("Analyzing customer risk..."):
            try:
                model = get_model()
                result = predict_single(customer_payload, model=model, threshold=threshold)
            except Exception as ex:
                st.error(f"Prediction failed: {ex}")

        if result:
            st.markdown("---")
            st.subheader("Prediction Results")

            p_col1, p_col2, p_col3 = st.columns(3)
            proba_pct = result["churn_probability"] * 100
            risk = result["risk_level"]
            is_churn = result["churn_prediction"] == 1

            with p_col1:
                st.metric(label="Churn Probability", value=f"{proba_pct:.1f}%")

            with p_col2:
                if is_churn:
                    st.error(f"Predicted Outcome: **{result['churn_label']}**")
                else:
                    st.success(f"Predicted Outcome: **{result['churn_label']}**")

            with p_col3:
                if risk == "High":
                    st.markdown("<p style='font-size:1.1rem;'>Risk Level: <span class='badge-high'>HIGH RISK</span></p>", unsafe_allow_html=True)
                elif risk == "Medium":
                    st.markdown("<p style='font-size:1.1rem;'>Risk Level: <span class='badge-med'>MEDIUM RISK</span></p>", unsafe_allow_html=True)
                else:
                    st.markdown("<p style='font-size:1.1rem;'>Risk Level: <span class='badge-low'>LOW RISK</span></p>", unsafe_allow_html=True)

            # Visual progress bar for churn probability
            st.progress(min(max(result["churn_probability"], 0.0), 1.0))

            # Strategic retention advice
            st.markdown("### Recommended Business Action")
            if is_churn:
                advice = []
                if tenure <= 3:
                    advice.append("• **Early-Stage Attrition:** Customer is in first 3 months. Dispatch targeted onboarding assistance and check experience.")
                if "Yes" in complain_flag:
                    advice.append("• **Unresolved Complaint:** High risk driven by recent complaint. Trigger urgent VIP customer care follow-up call.")
                if satisfaction <= 2:
                    advice.append("• **Low Satisfaction:** Offer personalized apology voucher / discount on next purchase.")
                if not advice:
                    advice.append("• Enroll customer in proactive loyalty discount campaign.")
                for line in advice:
                    st.warning(line)
            else:
                st.info("• Customer shows strong retention indicators. Standard engagement and newsletter promotions are recommended.")


# ==========================================================
# TAB 2: BATCH PREDICTION VIA CSV
# ==========================================================
with tab_batch:
    st.subheader("Bulk Customer Churn Scoring")
    st.markdown("Upload a CSV file with customer features to score thousands of records in bulk.")

    # 10 realistic sample customer rows
    sample_df = pd.DataFrame([
        {"Tenure": 15.0, "WarehouseToHome": 29.0, "NumberOfDeviceRegistered": 4, "PreferedOrderCat": "Laptop & Accessory", "SatisfactionScore": 3, "MaritalStatus": "Single", "NumberOfAddress": 2, "Complain": 0, "DaySinceLastOrder": 7.0, "CashbackAmount": 143.32},
        {"Tenure": 7.0, "WarehouseToHome": 25.0, "NumberOfDeviceRegistered": 4, "PreferedOrderCat": "Mobile", "SatisfactionScore": 1, "MaritalStatus": "Married", "NumberOfAddress": 2, "Complain": 0, "DaySinceLastOrder": 7.0, "CashbackAmount": 129.29},
        {"Tenure": 27.0, "WarehouseToHome": 13.0, "NumberOfDeviceRegistered": 3, "PreferedOrderCat": "Laptop & Accessory", "SatisfactionScore": 5, "MaritalStatus": "Married", "NumberOfAddress": 5, "Complain": 0, "DaySinceLastOrder": 7.0, "CashbackAmount": 168.54},
        {"Tenure": 1.0, "WarehouseToHome": 18.0, "NumberOfDeviceRegistered": 4, "PreferedOrderCat": "Mobile", "SatisfactionScore": 1, "MaritalStatus": "Single", "NumberOfAddress": 3, "Complain": 1, "DaySinceLastOrder": 1.0, "CashbackAmount": 135.50},
        {"Tenure": 20.0, "WarehouseToHome": 25.0, "NumberOfDeviceRegistered": 4, "PreferedOrderCat": "Fashion", "SatisfactionScore": 4, "MaritalStatus": "Married", "NumberOfAddress": 7, "Complain": 0, "DaySinceLastOrder": 8.0, "CashbackAmount": 230.27},
        {"Tenure": 30.0, "WarehouseToHome": 15.0, "NumberOfDeviceRegistered": 4, "PreferedOrderCat": "Others", "SatisfactionScore": 4, "MaritalStatus": "Single", "NumberOfAddress": 8, "Complain": 0, "DaySinceLastOrder": 8.0, "CashbackAmount": 322.17},
        {"Tenure": 1.0, "WarehouseToHome": 22.0, "NumberOfDeviceRegistered": 5, "PreferedOrderCat": "Mobile", "SatisfactionScore": 2, "MaritalStatus": "Single", "NumberOfAddress": 4, "Complain": 1, "DaySinceLastOrder": 2.0, "CashbackAmount": 140.00},
        {"Tenure": 12.0, "WarehouseToHome": 16.0, "NumberOfDeviceRegistered": 4, "PreferedOrderCat": "Grocery", "SatisfactionScore": 4, "MaritalStatus": "Married", "NumberOfAddress": 2, "Complain": 0, "DaySinceLastOrder": 11.0, "CashbackAmount": 152.81},
        {"Tenure": 24.0, "WarehouseToHome": 10.0, "NumberOfDeviceRegistered": 3, "PreferedOrderCat": "Laptop & Accessory", "SatisfactionScore": 4, "MaritalStatus": "Married", "NumberOfAddress": 3, "Complain": 0, "DaySinceLastOrder": 9.0, "CashbackAmount": 210.00},
        {"Tenure": 0.0, "WarehouseToHome": 15.0, "NumberOfDeviceRegistered": 4, "PreferedOrderCat": "Mobile", "SatisfactionScore": 1, "MaritalStatus": "Single", "NumberOfAddress": 4, "Complain": 1, "DaySinceLastOrder": 1.0, "CashbackAmount": 125.00},
    ])
    csv_sample_buffer = io.StringIO()
    sample_df.to_csv(csv_sample_buffer, index=False)
    st.download_button(
        "Download Sample CSV Template",
        data=csv_sample_buffer.getvalue(),
        file_name="sample_customers.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload Customers CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_input_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded **{len(batch_input_df)}** customer records:")
            st.dataframe(batch_input_df.head(), use_container_width=True)

            if st.button("Score Entire Batch", use_container_width=True):
                scored_df = None
                with st.spinner("Scoring customer records..."):
                    try:
                        model = get_model()
                        scored_df = predict_batch(batch_input_df, model=model, threshold=threshold)
                    except Exception as ex:
                        st.error(f"Batch scoring failed: {ex}")

                if scored_df is not None:
                    total_cust = len(scored_df)
                    churn_count = int(scored_df["churn_prediction"].sum())
                    churn_pct = (churn_count / total_cust) * 100 if total_cust > 0 else 0.0

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Customers Evaluated", total_cust)
                    m2.metric("Flagged Churners", churn_count)
                    m3.metric("Projected Churn Rate", f"{churn_pct:.1f}%")

                    st.markdown("### Scored Customer Dataset")
                    st.dataframe(scored_df, use_container_width=True)

                    # Export predictions
                    out_csv = io.StringIO()
                    scored_df.to_csv(out_csv, index=False)
                    st.download_button(
                        "Download Scored Results (CSV)",
                        data=out_csv.getvalue(),
                        file_name=f"scored_churn_{uploaded_file.name}",
                        mime="text/csv",
                        use_container_width=True,
                    )
        except Exception as e:
            st.error(f"Failed to process uploaded file: {e}")
