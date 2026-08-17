"""
preprocessing.py
================
Member Churn Prediction and Retention Advisor
Healthcare / Health-Plan Domain

Purpose
-------
Load, inspect, engineer features, and preprocess the patient_churn_dataset.
Does NOT train any model. Outputs train/test arrays ready for modelling.

Dataset facts (confirmed by inspection):
  - 2,000 rows, 21 columns
  - Target: Churned (0 = retained, 1 = churned) -- 68.35 % positive
  - No missing values, no duplicate rows
  - Columns: PatientID (ID), Last_Interaction_Date (date str),
    4 nominal categoricals, 14 numerical / binary features

Author  : CTS NPN Project
Created : 2026-08-13
"""

# --------------------------------------------------
# 0. Imports
# --------------------------------------------------
import pandas as pd
import numpy as np
import warnings
import os

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------
DATA_PATH = "patient_churn_dataset.csv.xls"   # actual filename on disk
SNAPSHOT_DATE = pd.Timestamp("2026-08-13")    # fixed reference date for recency
RANDOM_STATE = 42
TEST_SIZE = 0.20

print("=" * 65)
print(" MEMBER CHURN PREDICTION -- PREPROCESSING PIPELINE")
print("=" * 65)

df = pd.read_csv(DATA_PATH)
print(f"\n[1] Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

# --------------------------------------------------
# 2. Quick Sanity Checks
# --------------------------------------------------
print("\n[2] Sanity checks:")
print(f"    Missing values : {df.isnull().sum().sum()}")
print(f"    Duplicate rows : {df.duplicated().sum()}")

assert "Churned" in df.columns, "Target column 'Churned' not found!"
assert df["Churned"].isin([0, 1]).all(), "Target contains values other than 0/1!"

target_dist = df["Churned"].value_counts()
print(f"\n[3] Target distribution (Churned):")
print(f"    Churned  (1) : {target_dist[1]:,}  ({target_dist[1]/len(df)*100:.1f}%)")
print(f"    Retained (0) : {target_dist[0]:,}  ({target_dist[0]/len(df)*100:.1f}%)")
print("    Class imbalance ~2.16:1 -- use class_weight='balanced' in model")

# --------------------------------------------------
# 3. Drop ID Column
# --------------------------------------------------
# PatientID is a pure row identifier (2000 unique values = every row unique).
# Including it would cause data leakage and prevent generalisation.
ID_COLS = ["PatientID"]
df.drop(columns=ID_COLS, inplace=True)
print(f"\n[4] Dropped ID columns: {ID_COLS}")

# --------------------------------------------------
# 4. Date Feature Engineering -- Last_Interaction_Date
# --------------------------------------------------
# The raw date string cannot be used directly by ML models.
# We derive two numeric features and then drop the original column.

df["Last_Interaction_Date"] = pd.to_datetime(df["Last_Interaction_Date"])

# 4a. Recency: how many days ago was the last interaction?
#     More days -> less engaged -> higher churn risk.
df["days_since_interaction"] = (SNAPSHOT_DATE - df["Last_Interaction_Date"]).dt.days

# 4b. Month seasonality: health-plan renewals cluster at certain months.
df["interaction_month"] = df["Last_Interaction_Date"].dt.month

# Drop the raw date column -- now captured by the two numeric features above.
df.drop(columns=["Last_Interaction_Date"], inplace=True)

print("\n[5] Date engineering (Last_Interaction_Date -> 2 features):")
print(f"    days_since_interaction : min={df['days_since_interaction'].min()}, "
      f"max={df['days_since_interaction'].max()}, "
      f"mean={df['days_since_interaction'].mean():.0f}")
print(f"    interaction_month      : values 1-12")

# --------------------------------------------------
# 5. Feature Engineering (only columns that exist in the dataset)
# --------------------------------------------------

# 5a. Missed appointment rate
#     Raw missed count is skewed by total visits; the rate is fairer.
df["missed_appt_rate"] = df["Missed_Appointments"] / df["Visits_Last_Year"].clip(lower=1)

# 5b. Composite satisfaction score
#     Aggregates four correlated satisfaction columns into one signal.
#     Reduces multicollinearity while preserving overall sentiment.
sat_cols = [
    "Overall_Satisfaction",
    "Wait_Time_Satisfaction",
    "Staff_Satisfaction",
    "Provider_Rating",
]
df["composite_satisfaction"] = df[sat_cols].mean(axis=1)

# 5c. Member engagement score
#     Weights portal usage more heavily (indicates digital engagement).
df["engagement_score"] = (
    df["Visits_Last_Year"]
    + df["Portal_Usage"] * 2
    + df["Referrals_Made"]
)

# 5d. Monthly cost burden
#     High out-of-pocket cost in early tenure is a strong churn signal.
df["cost_per_tenure_month"] = (
    df["Avg_Out_Of_Pocket_Cost"] / df["Tenure_Months"].clip(lower=1)
)

# 5e. Low satisfaction flag
#     Flags critically dissatisfied members (below midpoint of 1.5-5 scale).
df["is_low_satisfaction"] = (df["Overall_Satisfaction"] < 2.5).astype(int)

# 5f. Distance barrier flag
#     Members > 40 miles away face access barriers that predict churn.
df["is_distant"] = (df["Distance_To_Facility_Miles"] > 40).astype(int)

# 5g. Monthly visit rate
#     Normalises visit frequency to a per-month basis.
df["monthly_visit_rate"] = df["Visits_Last_Year"] / 12

print("\n[6] Engineered features created:")
engineered = [
    "days_since_interaction", "interaction_month",
    "missed_appt_rate", "composite_satisfaction",
    "engagement_score", "cost_per_tenure_month",
    "is_low_satisfaction", "is_distant", "monthly_visit_rate",
]
for f in engineered:
    print(f"    + {f}")

# --------------------------------------------------
# 6. Define Feature Groups for Pipeline
# --------------------------------------------------

TARGET = "Churned"
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Nominal categoricals -- no ordinal relationship -> OneHotEncoder
NOMINAL_CATS = ["Gender", "State", "Specialty", "Insurance_Type"]

# All numeric columns (continuous, binary, and engineered)
NUM_COLS = [c for c in X.columns if c not in NOMINAL_CATS]

print(f"\n[7] Feature groups:")
print(f"    Nominal categoricals ({len(NOMINAL_CATS)}) : {NOMINAL_CATS}")
print(f"    Numerical / binary  ({len(NUM_COLS)}) :")
for c in NUM_COLS:
    print(f"      - {c}")

# --------------------------------------------------
# 7. Train / Test Split (stratified to preserve class ratio)
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,            # preserves 68/32 split in both sets
)

print(f"\n[8] Train/test split (stratified, test_size={TEST_SIZE}):")
print(f"    Train: {X_train.shape[0]:,} rows  |  Test: {X_test.shape[0]:,} rows")
print(f"    Train churn rate: {y_train.mean()*100:.1f}%  "
      f"| Test churn rate: {y_test.mean()*100:.1f}%")

# --------------------------------------------------
# 8. Build sklearn ColumnTransformer Pipeline
# --------------------------------------------------

# Numeric pipeline:
#   StandardScaler -- centres and scales to unit variance.
#   Required for linear models / SVMs; harmless for tree-based models.
numeric_pipeline = Pipeline([
    ("scaler", StandardScaler()),
])

# Categorical pipeline:
#   OneHotEncoder -- creates k binary columns per category.
#   handle_unknown='ignore' makes the pipeline robust to unseen categories
#   at inference time (e.g., a new State in production).
#   drop='first' removes one dummy per feature to avoid multicollinearity
#   in linear models (irrelevant for trees but keeps dimensions tidy).
categorical_pipeline = Pipeline([
    ("ohe", OneHotEncoder(
        handle_unknown="ignore",
        drop="first",
        sparse_output=False,
    )),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, NUM_COLS),
        ("cat", categorical_pipeline, NOMINAL_CATS),
    ],
    remainder="drop",    # drops any columns not explicitly listed (safety net)
    verbose_feature_names_out=True,
)

# --------------------------------------------------
# 9. Fit on Train, Transform Both Sets
# --------------------------------------------------
print("\n[9] Fitting preprocessor on training data ...")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)   # uses train statistics only

# Recover readable feature names after OHE expansion
feature_names = preprocessor.get_feature_names_out()
feature_names = [
    n.replace("num__", "").replace("cat__", "")
    for n in feature_names
]

print(f"    Preprocessed train shape : {X_train_processed.shape}")
print(f"    Preprocessed test shape  : {X_test_processed.shape}")
print(f"    Total features after OHE : {X_train_processed.shape[1]}")

# --------------------------------------------------
# 10. Convert to DataFrames for Inspection / Export
# --------------------------------------------------
X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)

print("\n[10] Final feature list (post-preprocessing):")
for i, name in enumerate(feature_names, 1):
    print(f"    {i:2d}. {name}")

# --------------------------------------------------
# 11. Save Preprocessed Artefacts
# --------------------------------------------------
os.makedirs("outputs", exist_ok=True)

X_train_df.to_csv("outputs/X_train.csv", index=False)
X_test_df.to_csv("outputs/X_test.csv", index=False)
y_train.reset_index(drop=True).to_csv("outputs/y_train.csv", index=False)
y_test.reset_index(drop=True).to_csv("outputs/y_test.csv", index=False)
pd.Series(feature_names).to_csv("outputs/feature_names.csv", index=False, header=["feature"])

# Save preprocessor bundle
import pickle
preprocessor_bundle = {
    "preprocessor": preprocessor,
    "num_cols": NUM_COLS,
    "nominal_cats": NOMINAL_CATS
}
with open("outputs/preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor_bundle, f)

print("\n[11] Saved to outputs/:")
print("    X_train.csv, X_test.csv, y_train.csv, y_test.csv, feature_names.csv, preprocessor.pkl")

# --------------------------------------------------
# 12. Final Summary
# --------------------------------------------------
print("\n" + "=" * 65)
print(" PREPROCESSING COMPLETE -- SUMMARY")
print("=" * 65)
print(f"  Original features      : {df.shape[1] - 1}")
print(f"  Features after OHE     : {X_train_processed.shape[1]}")
print(f"  Training samples       : {X_train_processed.shape[0]:,}")
print(f"  Test samples           : {X_test_processed.shape[0]:,}")
print(f"  Target class balance   : {y_train.mean()*100:.1f}% churn (train)")
print(f"  Missing values in X    : {X_train_df.isnull().sum().sum()}")
print("=" * 65)
print("  Ready for model training.")
print("  Remember: use class_weight='balanced' or SMOTE in the model step.")
print("=" * 65)
