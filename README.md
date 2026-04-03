# Loan Approval Prediction System

## 📌 Overview
This project predicts loan approval using advanced machine learning techniques and an ensemble model. It combines multiple models to improve prediction accuracy and robustness.

The system is deployed using Flask and hosted on AWS EC2 for real-time predictions.

---

## ⚙️ Tech Stack
- Python
- Scikit-learn
- XGBoost
- Flask
- AWS EC2

---

## 🔄 Data Processing
- Handled missing values using imputation
- Applied log transformation for skewed features
- Performed feature scaling and encoding using ColumnTransformer
- Built reusable preprocessing pipeline

---

## 🧠 Feature Engineering
Created meaningful features:
- TotalIncome
- EMI
- Income_to_Loan
- Loan_to_Term
- Income_per_person
- Credit_TotalIncome

---

## 🤖 Models Used
- Random Forest
- Gradient Boosting
- XGBoost
- Logistic Regression

### 🔥 Final Model
Voting Classifier (Ensemble Learning)

---

## 📊 Model Evaluation
- Stratified Train-Test Split
- 5-Fold Cross Validation
- Accuracy, Confusion Matrix, Classification Report

---

## 🚀 Deployment
- Backend: Flask
- Hosted on AWS EC2
- Real-time prediction through web interface

---

## 💡 Key Highlights
- End-to-end ML pipeline
- Ensemble learning for better performance
- Handles imbalanced data
- Deployed production-ready system

---

## 👨‍💻 Author
Farhan Ahmed Mohammad
