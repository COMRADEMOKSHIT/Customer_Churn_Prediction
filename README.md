# Customer_Churn_Prediction
A classification model that predicts customer churn, enabling the business to target retention offers at the customers most likely to leave and utilizes a user-friendly interface built with Streamlit.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Project Structure](#project-structure)
4. [License](#license)

---

## Project Overview

**Objective**:
To build an interactive web application that allows users to upload and explore the Telco Customer Churn dataset, visualize customer behavior patterns across demographic, service, and billing attributes, and prepare the data for downstream machine learning (churn prediction) tasks.

**Problem Statement**:
Customer churn — when a customer stops using a company's service — is a major concern for telecom companies since acquiring new customers is far more expensive than retaining existing ones. Understanding which factors (contract type, tenure, payment method, internet service, etc.) correlate with churn helps businesses take proactive retention measures.

**Dataset**:
WA_Fn-UseC_-Telco-Customer-Churn.csv — a public dataset (commonly sourced from IBM/Kaggle) containing ~7,000 customer records with features like:

**Demographics**: gender, SeniorCitizen, Partner, Dependents
Account info: tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
Services subscribed: PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
Target variable: Churn (Yes/No)

**Tech Stack**:

Component	Tool
App framework	Streamlit
Data handling	Pandas, NumPy
Visualization	Seaborn, Matplotlib
Preprocessing	scikit-learn (LabelEncoder)
Deployment	Streamlit Community Cloud

## Features

Data Upload/Loading — users can upload their own CSV or use a bundled sample file
Data Cleaning — automatically converts TotalCharges to numeric, drops nulls, removes the non-predictive customerID column
Data Preview — displays column types and missing-value summary
Categorical Distribution Plots — count plots across 15 categorical features (gender, contract type, services, etc.)
Numeric Distribution Plots — KDE plots for tenure, MonthlyCharges, TotalCharges
Label Encoding — converts categorical columns to numeric form, ready for ML modeling
Downloadable Output — encoded dataset available as a CSV download

Current Scope: Exploratory Data Analysis (EDA) and data preprocessing only — no predictive model is trained yet.

---

## Technologies Used

The project utilizes the following technologies and libraries:
- **Python**: Programming language for backend and model development.
- **Streamlit**: Web framework for frontend.
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical computations.
- **Scikit-learn**: Machine learning utilities.
- **TensorFlow**: Deep learning framework for model training.
- **Matplotlib**: Data visualization.
