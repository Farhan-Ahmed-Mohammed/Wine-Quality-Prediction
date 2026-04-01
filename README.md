import numpy as np
import pandas as pd
import pickle
import sys

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer

!{sys.executable} -m pip install xgboost -q
from xgboost import XGBClassifier

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv("Loan_data.csv")
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

# ── Feature engineering ──────────────────────────────────────────────────────
df["TotalIncome"]           = df["ApplicantIncome"] + df["CoapplicantIncome"]
df["EMI"]                   = df["LoanAmount"] / df["Loan_Amount_Term"]
df["Income_to_Loan"]        = df["TotalIncome"] / df["LoanAmount"]
df["Credit_TotalIncome"]    = df["Credit_History"] * df["TotalIncome"]
df["Loan_to_Term"]          = df["LoanAmount"] / df["Loan_Amount_Term"]
df["Income_per_person"]     = df["TotalIncome"] / (df["Dependents"].replace("3+", 3).astype(float) + 1)
df["Dependents"]            = df["Dependents"].replace("3+", 3).astype(float)

# ── Split X and y ────────────────────────────────────────────────────────────
y = df["Loan_Status"]
x = df.drop(columns=["Loan_Status", "Loan_ID"])

# ── Column groups ────────────────────────────────────────────────────────────
categorical_cols = x.select_dtypes(include=['object']).columns.tolist()
numerical_cols   = x.select_dtypes(include=['int64', 'float64']).columns.tolist()
log_cols         = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "TotalIncome", "Income_per_person"]
other_num        = [c for c in numerical_cols if c not in log_cols]

print("Categorical columns:", categorical_cols)
print("Numerical columns:  ", numerical_cols)

# ── Preprocessing pipelines ──────────────────────────────────────────────────
log_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("log",     FunctionTransformer(np.log1p)),
    ("scaler",  StandardScaler())
])

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  OneHotEncoder(handle_unknown="ignore"))
])

preprocessing = ColumnTransformer([
    ("log", log_pipeline, log_cols),
    ("num", num_pipeline, other_num),
    ("cat", cat_pipeline, categorical_cols)
])

# ── Models with class_weight to handle 2:1 imbalance ────────────────────────
# No SMOTE — class_weight handles imbalance cleanly on small datasets

xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.5,
    reg_lambda=1.5,
    scale_pos_weight=192/422,  # ratio of negative/positive
    random_state=42,
    eval_metric='logloss'
)

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=5,
    min_samples_leaf=3,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

gb_model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    min_samples_leaf=3,
    random_state=42
)

lr_model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    C=0.1,
    random_state=42
)

ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('rf',  rf_model),
        ('gb',  gb_model),
        ('lr',  lr_model)
    ],
    voting='soft'
)

model = Pipeline([
    ("preprocessing", preprocessing),
    ("ensemble",      ensemble)
])

# ── Train ────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

model.fit(X_train, y_train)

# ── Cross-validation ─────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, x, y, cv=cv, scoring="accuracy")
print(f"\nCross-val accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── Evaluate ─────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)

print("\nModel Evaluation on test set")
print("-----------------------------")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ── Sample prediction ────────────────────────────────────────────────────────
sample = pd.DataFrame([{
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": 1.0,
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 4500,
    "CoapplicantIncome": 3000,
    "LoanAmount": 128,
    "Loan_Amount_Term": 360,
    "Credit_History": 1,
    "Property_Area": "Urban",
    "TotalIncome": 7500,
    "EMI": 128/360,
    "Income_to_Loan": 7500/128,
    "Credit_TotalIncome": 1 * 7500,
    "Loan_to_Term": 128/360,
    "Income_per_person": 7500 / (1 + 1)
}])

print("\nPrediction:", model.predict(sample)[0])
print("Probability:", model.predict_proba(sample)[0][1])

# ── Save ─────────────────────────────────────────────────────────────────────
with open("loan_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved to loan_model.pkl")
