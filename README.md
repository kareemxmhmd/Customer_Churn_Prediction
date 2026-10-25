# E-Commerce Customer Churn Prediction

🔗 **Live App:** [customerchurnprediction-ose.streamlit.app](https://customerchurnprediction-ose.streamlit.app/)

## Business Problem
Acquiring new customers costs 5 to 7 times more than retaining existing ones. In e-commerce, customer attrition quietly drains recurring revenue, diminishes customer lifetime value, and wastes acquisition investments. Most companies only detect churn after a customer has stopped purchasing, leaving retention teams reacting too late.

## Solution
This project provides a proactive customer retention system powered by machine learning. Deployed as an interactive web application, it predicts customer churn risk in real time, classifies accounts into clear risk tiers (High, Medium, Low), and delivers targeted retention recommendations so marketing and customer support teams can intervene before customers defect.

## Data
The model is built on an e-commerce customer dataset of 3,269 unique profiles with a 16.3% historical churn rate. The data captures key relationship and engagement signals:
- **Customer Lifecycle:** Platform tenure and days elapsed since the last order.
- **Satisfaction & Friction:** Customer satisfaction ratings and formal complaint history.
- **Account & Shopping Behaviors:** Cashback incentives earned, registered devices, delivery addresses, marital status, and preferred purchase categories.

These attributes capture the critical behavioral shifts—such as service dissatisfaction and declining order frequency—that precede churn.

## Business Value
- **Identifying Risks Early:** Flags at-risk accounts before they defect, with special attention to vulnerable onboarding stages (0–3 months tenure) and unresolved customer complaints.
- **Reducing Retention Costs:** With 70.3% precision, 7 out of 10 flagged customers are true churners, preventing wasteful discount spend on loyal customers who would have stayed anyway.
- **Improving Decisions:** Replaces broad, untargeted promotions with segment-specific playbooks (e.g., immediate customer support escalation for unhappy complainers, onboarding perks for early-tenure shoppers).
- **Increasing Revenue Retention:** Intercepts 80% of departing customers, directly safeguarding recurring revenue and extending customer lifetime value.
- **Optimizing Operations:** Enables retention teams to score single customer accounts on demand or upload batch customer lists for targeted retention campaigns.

## Machine Learning Approach
This solution is formulated as a binary classification problem trained on customer retention outcomes. An **XGBoost** classification pipeline was implemented with class weighting to handle customer churn imbalance. The model was optimized for **Recall** and **Precision-Recall AUC (PR-AUC)**, reflecting the business reality that failing to catch a churning customer is far more costly than sending an extra retention offer.

## Result
Evaluated on an untouched holdout test set:
- **80.0% Recall:** Successfully intercepts 4 out of every 5 churners (64 of 80) in time for retention action.
- **70.3% Precision:** Ensures marketing budgets are focused on genuinely at-risk customers.
- **91.2% Overall Accuracy** and **0.832 PR-AUC** (compared to a 0.163 random baseline).

From a business standpoint, this allows the retention team to protect up to 80% of at-risk customer revenue while keeping campaign operational costs efficient and controlled.
