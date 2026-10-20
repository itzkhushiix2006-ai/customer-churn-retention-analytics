
# Customer Churn Prediction & Retention Analytics
https://customer-churn-retention-analytics-vpgyhj3bgcwvoru3z5d6kt.streamlit.app/

## Project Overview
This project uses Machine Learning to predict customer churn in the telecommunications industry. It helps identify customers who may leave the service and supports data-driven customer retention strategies.

## Problem Statement
Customer churn can lead to revenue loss for telecom companies. The objective is to predict churn probability using customer demographics, account information, contract details, and subscribed services.

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Streamlit
- Joblib
- JupyterLab

## Dataset
The project uses the IBM Telco Customer Churn dataset, containing 7,043 customer records and 21 columns.

## Project Workflow
1. Data loading and inspection
2. Data cleaning and missing-value handling
3. Exploratory Data Analysis (EDA)
4. Feature and target separation
5. Train-test split with stratification
6. Categorical encoding and numerical scaling
7. Logistic Regression model training
8. Evaluation using accuracy, precision, recall, F1-score, and ROC-AUC
9. Churn probability and risk-level estimation
10. Interactive prediction through Streamlit

## Model
The project uses Logistic Regression with class weighting to address the imbalance between churn and non-churn customers.

### Model Evaluation
- Accuracy: 73.81%
- Churn Precision: 50.43%
- Churn Recall: 78.34%
- Churn F1-score: 61.36%
- ROC-AUC: 0.8416

The model prioritizes identifying more potential churners, while also producing some false-positive predictions.

## Business Insights
- Month-to-month contract customers have a higher observed churn rate than customers on longer contracts.
- Customers who churned had lower average tenure than customers who stayed.
- Customers who churned had higher average monthly charges in the analyzed dataset.

These findings are associations in the dataset and do not establish causation.

## Streamlit Application
The application allows users to enter customer details and view:
- Predicted churn status
- Estimated churn probability
- Low, Medium, or High risk category

The risk categories use these illustrative probability thresholds:
- Low: below 30%
- Medium: 30% to below 60%
- High: 60% or above

## How to Run
1. Install the required packages:
   `pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib`

2. Keep these files in the same project directory:
   - `app.py`
   - `churn_model.pkl`
   - `churn_preprocessor.pkl`

3. Start the application:
   `streamlit run app.py`

## Future Improvements
- Compare additional classification algorithms.
- Tune the decision threshold using validation data.
- Improve churn-risk explanations.
- Add a retention recommendation module.
- Deploy the application online.

## Disclaimer
This project is for educational and demonstration purposes. Predictions are estimates based on historical telecom data and should not be treated as guarantees of customer behavior.
