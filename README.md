# Telco Customer Churn Analysis and Prediction System

## Overview

The Telco Customer Churn Analysis and Prediction System is a data mining project that analyzes customer behavior and predicts whether a telecommunications customer is likely to leave the company.

The project uses descriptive analysis, association rule mining, K-Means clustering, and machine learning classification to discover churn patterns, segment customers, and support customer retention decisions.

## Business Problem

Customer churn can affect a telecommunications company's revenue and long-term customer relationships. This project explores customer characteristics associated with churn and develops a model to help identify customers who may be at risk of leaving.

The analysis identifies associations and patterns; it does not establish that any individual feature directly causes churn.

## Objectives

- Analyze customer data and understand the overall churn rate.
- Explore churn patterns across customer characteristics.
- Discover associations between customer attributes and churn using Apriori.
- Segment customers using K-Means clustering.
- Compare Logistic Regression, Decision Tree, and Random Forest classification models.
- Evaluate class imbalance handling using class weighting and SMOTE.
- Build an interactive Streamlit application for data visualization and churn prediction.

## Dataset

The project uses the IBM Telco Customer Churn dataset from Kaggle.

- Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- Records: 7,043 customers
- Attributes: 21 columns
- Target variable: `Churn` (Yes/No)

## Methods

### 1. Data Preprocessing
- Inspect data types, missing values, and duplicate records.
- Convert `TotalCharges` to numeric.
- Handle blank `TotalCharges` values after examining their context.
- Exclude identifier and target-leakage columns from model predictors.
- Encode categorical variables for machine learning.

### 2. Descriptive Mining
Explore churn rates by contract type, tenure, technical support, internet service, payment method, and other customer characteristics.

### 3. Association Rule Mining
Use the Apriori algorithm to identify combinations of customer characteristics associated with churn. Interpret support, confidence, and lift.

### 4. Customer Segmentation
Use K-Means clustering with standardized customer features to identify customer segments and examine their observed churn rates.

### 5. Predictive Mining
Compare the following classification models:
- Logistic Regression
- Decision Tree
- Random Forest

### 6. Class Imbalance Experiments
Compare a baseline model with class weighting and SMOTE. SMOTE is applied to training data only.

### 7. Model Evaluation
Evaluate models using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

## System Features

- Customer churn overview dashboard
- Churn analysis and visualizations
- Association pattern analysis
- Customer segmentation insights
- Churn prediction for customer inputs
- Model evaluation and business recommendations

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib / Seaborn
- Scikit-learn
- MLxtend
- Imbalanced-learn
- Streamlit
- Jupyter Notebook

## Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_PROJECT_FOLDER>
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Run the Application

Run the Streamlit application:

```bash
python -m streamlit run <YOUR_STREAMLIT_APP_FILE>.py
```

Replace the filename with the actual name of your Streamlit application.

## Project Structure

```text
project/
├── README.md
├── requirements.txt
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/
│   └── telco_churn_analysis.ipynb
├── app.py
└── models/
    └── logistic_regression_model.pkl
```

Adjust this structure to match the files actually included in your repository.

## Limitations

- The model is based on the available dataset and may not represent every telecommunications business.
- Model performance may change when applied to new customer populations.
- Association rules and cluster-level churn rates describe patterns and do not prove causation.
- Predictions should support, not replace, business judgment.

## Future Improvements

- Test the system on updated customer data.
- Improve model calibration and churn-risk thresholds.
- Explore additional features and alternative models.
- Evaluate the system with real-world retention outcomes.

## Author

**[Bhone Min Khant]**

Data Mining Project — Telco Customer Churn Analysis and Prediction System
