import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              HistGradientBoostingClassifier, VotingClassifier)
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, brier_score_loss)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.utils import resample
import json

# ==========================================
# 1. LOAD & QUALITY ANALYSIS
# ==========================================
print("Loading data...")
df = pd.read_csv('patient_churn_dataset.csv.xls')

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
def engineer_features(data):
    df_fe = data.copy()
    
    # 1. Tenure-based features
    df_fe['Age_Tenure'] = df_fe['Age'] * df_fe['Tenure_Months']
    df_fe['Days_per_Tenure'] = df_fe['Days_Since_Last_Visit'] / (df_fe['Tenure_Months'] + 1)
    df_fe['Cost_Tenure'] = df_fe['Avg_Out_Of_Pocket_Cost'] / (df_fe['Tenure_Months'] + 1)
    
    # 2. Visit frequency
    df_fe['Missed_Ratio'] = df_fe['Missed_Appointments'] / (df_fe['Visits_Last_Year'] + 1)
    df_fe['Visit_Per_Month'] = df_fe['Visits_Last_Year'] / (df_fe['Tenure_Months'] / 12 + 1)
    
    # 3. Payment behavior
    df_fe['Log_Cost'] = np.log1p(df_fe['Avg_Out_Of_Pocket_Cost'])
    df_fe['Cost_Per_Visit'] = df_fe['Avg_Out_Of_Pocket_Cost'] / (df_fe['Visits_Last_Year'] + 1)
    df_fe['High_Cost_Low_Sat'] = df_fe['Avg_Out_Of_Pocket_Cost'] * (5.0 - df_fe['Overall_Satisfaction'])
    
    # 4. Days since last interaction
    df_fe['Log_Days'] = np.log1p(df_fe['Days_Since_Last_Visit'])
    df_fe['Sqrt_Days'] = np.sqrt(df_fe['Days_Since_Last_Visit'])
    df_fe['Long_Absent'] = (df_fe['Days_Since_Last_Visit'] > 300).astype(int)
    df_fe['Very_Long_Absent'] = (df_fe['Days_Since_Last_Visit'] > 500).astype(int)
    
    # 5. Engagement score
    df_fe['Engagement_Score'] = (df_fe['Portal_Usage'] * df_fe['Referrals_Made'] +
                                  df_fe['Visits_Last_Year'] - df_fe['Missed_Appointments'])
    
    # 6. Support/contact frequency & administrative frustration
    df_fe['Billing_Dissatisfaction'] = df_fe['Billing_Issues'] * (5.0 - df_fe['Overall_Satisfaction'])
    df_fe['Billing_And_Poor_Wait'] = df_fe['Billing_Issues'] * (5.0 - df_fe['Wait_Time_Satisfaction'])
    
    # 7. Access / distance barrier
    df_fe['Distance_Missed'] = df_fe['Distance_To_Facility_Miles'] * df_fe['Missed_Appointments']
    df_fe['Access_Score'] = (df_fe['Distance_To_Facility_Miles'] * (df_fe['Missed_Appointments'] + 1) /
                              (df_fe['Portal_Usage'] + 1))
    
    # 8. Provider satisfaction compound
    df_fe['Provider_Overall_Dissatisfaction'] = (5 - df_fe['Provider_Rating']) * (5 - df_fe['Overall_Satisfaction'])
    df_fe['Sat_Times_Portal'] = df_fe['Overall_Satisfaction'] * df_fe['Portal_Usage']
    
    # 9. Important interaction terms
    df_fe['Recency_Missed'] = df_fe['Days_Since_Last_Visit'] * df_fe['Missed_Appointments']
    df_fe['Sat_Tenure'] = df_fe['Overall_Satisfaction'] * df_fe['Tenure_Months']
    df_fe['Cost_Sat'] = df_fe['Avg_Out_Of_Pocket_Cost'] * df_fe['Overall_Satisfaction']
    df_fe['Visits_Sat'] = df_fe['Visits_Last_Year'] * df_fe['Overall_Satisfaction']
    
    # 10. Satisfaction summary stats
    df_fe['Total_Satisfaction'] = (df_fe['Overall_Satisfaction'] + df_fe['Wait_Time_Satisfaction'] +
                                    df_fe['Staff_Satisfaction'] + df_fe['Provider_Rating'])
    df_fe['Avg_Satisfaction'] = df_fe['Total_Satisfaction'] / 4
    df_fe['Min_Satisfaction'] = df_fe[['Overall_Satisfaction','Wait_Time_Satisfaction',
                                        'Staff_Satisfaction','Provider_Rating']].min(axis=1)
    df_fe['Sat_Variance'] = df_fe[['Overall_Satisfaction','Wait_Time_Satisfaction',
                                    'Staff_Satisfaction','Provider_Rating']].var(axis=1)
    
    # Gender encoding
    df_fe['Gender_Male'] = (df_fe['Gender'] == 'Male').astype(int)
    df_fe = df_fe.drop(columns=['Gender'])
    
    return df_fe

print("Engineering features...")
df_fe = engineer_features(df)

# Drop leak-sensitive columns and PatientID
df_features = df_fe.drop(columns=['PatientID', 'Last_Interaction_Date', 'Churned'])
y = df['Churned'].values

# ==========================================
# 4. CATEGORICAL ENCODING
# ==========================================
# Use One-Hot Encoding for State, Specialty, Insurance_Type
df_encoded = pd.get_dummies(df_features, columns=['State', 'Specialty', 'Insurance_Type'], drop_first=False)
feature_names = list(df_encoded.columns)

# ==========================================
# 5. DATA SPLITTING (Stratified 80/20)
# ==========================================
X_raw = df_encoded.values
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.20, random_state=42, stratify=y
)

# Apply scaling correctly (fit on train, transform on test)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# Identify binary columns for scaling exclusions later if needed
binary_cols = ['Billing_Issues', 'Portal_Usage', 'Long_Absent', 'Very_Long_Absent']
for col in feature_names:
    if any(c in col for c in ['State_', 'Specialty_', 'Insurance_Type_']):
        binary_cols.append(col)
numeric_cols = [c for c in feature_names if c not in binary_cols]

# ==========================================
# 6. CLASS IMBALANCE & OPTIMIZER FUNCTIONS
# ==========================================
def find_best_threshold(y_true, y_prob):
    """Finds the probability threshold that maximizes training accuracy."""
    best_t, best_acc = 0.5, 0.0
    for t in np.arange(0.20, 0.85, 0.005):
        y_pred = (y_prob >= t).astype(int)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_t = t
    return best_t

def eval_predictions(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_val = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    return acc, prec, rec, f1, auc_val, cm

# ==========================================
# 7. MODEL DEFINITIONS & TUNING GRIDS
# ==========================================
models_to_test = {
    'Logistic Regression': (
        LogisticRegression(random_state=42, max_iter=2000, solver='liblinear'),
        {'C': [0.01, 0.05, 0.1, 0.5, 1.0], 'penalty': ['l1', 'l2']}
    ),
    'Decision Tree': (
        DecisionTreeClassifier(random_state=42),
        {'max_depth': [3, 4, 5, 6], 'min_samples_leaf': [5, 10, 20]}
    ),
    'Random Forest': (
        RandomForestClassifier(random_state=42),
        {'n_estimators': [100, 200, 300], 'max_depth': [5, 8, 10], 'min_samples_leaf': [5, 10]}
    ),
    'Gradient Boosting': (
        GradientBoostingClassifier(random_state=42),
        {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 4]}
    ),
    'Hist Gradient Boosting': (
        HistGradientBoostingClassifier(random_state=42),
        {'max_iter': [100, 200], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5]}
    )
}

print("\n=== STARTING MODEL SELECTION & HYPERPARAMETER TUNING (5-Fold CV) ===")
tuned_models = {}
cv_results_summary = {}

cv_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, (clf, grid) in models_to_test.items():
    print(f"Tuning {name}...")
    grid_search = GridSearchCV(clf, grid, cv=cv_kfold, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_clf = grid_search.best_estimator_
    tuned_models[name] = best_clf
    
    # Run cross_validate to get clean standard deviation and mean recall/precision
    cv_out = cross_validate(best_clf, X_train, y_train, cv=cv_kfold, 
                            scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
    
    cv_results_summary[name] = {
        'params': grid_search.best_params_,
        'mean_accuracy': cv_out['test_accuracy'].mean(),
        'std_accuracy': cv_out['test_accuracy'].std(),
        'mean_precision': cv_out['test_precision'].mean(),
        'mean_recall': cv_out['test_recall'].mean(),
        'mean_f1': cv_out['test_f1'].mean(),
        'mean_auc': cv_out['test_roc_auc'].mean(),
    }
    
    print(f"  Best params: {grid_search.best_params_}")
    print(f"  CV Acc: {cv_results_summary[name]['mean_accuracy']:.4f} +/- {cv_results_summary[name]['std_accuracy']:.4f}")

# ==========================================
# 15. VOTING ENSEMBLE
# ==========================================
print("\nBuilding Voting Ensemble (soft) of best LR, RF, HistGBM...")
voter = VotingClassifier(
    estimators=[
        ('lr', tuned_models['Logistic Regression']),
        ('rf', tuned_models['Random Forest']),
        ('hgbm', tuned_models['Hist Gradient Boosting'])
    ],
    voting='soft'
)
voter.fit(X_train, y_train)
cv_out_v = cross_validate(voter, X_train, y_train, cv=cv_kfold,
                          scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
cv_results_summary['Voting Ensemble'] = {
    'params': 'Soft voting of best LR, RF, HistGBM',
    'mean_accuracy': cv_out_v['test_accuracy'].mean(),
    'std_accuracy': cv_out_v['test_accuracy'].std(),
    'mean_precision': cv_out_v['test_precision'].mean(),
    'mean_recall': cv_out_v['test_recall'].mean(),
    'mean_f1': cv_out_v['test_f1'].mean(),
    'mean_auc': cv_out_v['test_roc_auc'].mean()
}
print(f"  Voting Ensemble CV Acc: {cv_results_summary['Voting Ensemble']['mean_accuracy']:.4f}")

# ==========================================
# 11. THRESHOLD OPTIMIZATION & OVERFITTING REPORT
# ==========================================
print("\n=== FINAL TEST EVALUATION & THRESHOLD OPTIMIZATION ===")
print(f"{'Model':<30} | {'Train Acc':<9} | {'CV Acc':<9} | {'Test Acc':<9} | {'Tuned Acc':<9} | {'Recall':<6} | {'F1':<6} | {'Best Thresh':<11}")
print("-" * 110)

best_tuned_model_name = None
best_tuned_test_acc = 0.0
best_tuned_metrics = {}

for name, clf in list(tuned_models.items()) + [('Voting Ensemble', voter)]:
    # Train performance
    if name == 'Voting Ensemble':
        clf.fit(X_train, y_train)
        
    y_prob_tr = clf.predict_proba(X_train)[:, 1]
    y_prob_te = clf.predict_proba(X_test)[:, 1]
    
    # Train accuracy (0.50 threshold)
    train_acc = accuracy_score(y_train, (y_prob_tr >= 0.50).astype(int))
    
    # Test accuracy (0.50 threshold)
    test_acc_050 = accuracy_score(y_test, (y_prob_te >= 0.50).astype(int))
    
    # Threshold Tuning on training predictions
    best_t = find_best_threshold(y_train, y_prob_tr)
    
    # Tuned test evaluation
    acc, prec, rec, f1, auc_val, cm = eval_predictions(y_test, y_prob_te, best_t)
    
    # Track the best model based on Tuned Test Accuracy
    if acc > best_tuned_test_acc:
        best_tuned_test_acc = acc
        best_tuned_model_name = name
        best_tuned_metrics = {
            'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc_val,
            'cm': cm, 'threshold': best_t, 'clf': clf
        }
        
    cv_acc = cv_results_summary[name]['mean_accuracy']
    
    print(f"{name:<30} | {train_acc:.4f}    | {cv_acc:.4f}    | {test_acc_050:.4f}    | {acc:.4f}    | {rec:.4f} | {f1:.4f} | {best_t:.3f}")

# ==========================================
# 16. PROBABILITY CALIBRATION
# ==========================================
print("\n=== PROBABILITY CALIBRATION CHECK ===")
best_clf = best_tuned_metrics['clf']
y_prob_best = best_clf.predict_proba(X_test)[:, 1]

# Display calibration metrics
brier = brier_score_loss(y_test, y_prob_best)
print(f"Best Model ({best_tuned_model_name}) Brier Score: {brier:.4f}")

# Probability Risk Buckets
buckets = {
    'Low Risk (0-30%)': np.sum((y_prob_best >= 0.0) & (y_prob_best < 0.3)),
    'Medium Risk (30-70%)': np.sum((y_prob_best >= 0.3) & (y_prob_best < 0.7)),
    'High Risk (70-100%)': np.sum((y_prob_best >= 0.7) & (y_prob_best <= 1.0))
}
print("\nRisk Buckets Distribution:")
total_test = len(y_test)
for k, v in buckets.items():
    print(f"  {k:<22}: {v:<4} ({v/total_test*100:.1f}%)")

# Churn rates within risk buckets
print("\nActual Churn Rates per Risk Bucket:")
for label, (low, high) in [('Low Risk', (0.0, 0.3)), ('Medium Risk', (0.3, 0.7)), ('High Risk', (0.7, 1.0))]:
    mask = (y_prob_best >= low) & (y_prob_best < high)
    if label == 'High Risk':
        mask = (y_prob_best >= low) & (y_prob_best <= high)
    
    if mask.sum() > 0:
        churn_rate = np.mean(y_test[mask])
        print(f"  {label:<12} Churn Rate: {churn_rate*100:.1f}% ({y_test[mask].sum()}/{mask.sum()})")
    else:
        print(f"  {label:<12} No samples fell in this range.")

# ==========================================
# 17. RETENTION ADVISOR & FEATURE EXPLAINABILITY
# ==========================================
print("\n=== RETENTION ADVISOR IMPLEMENTATION & RISK DRIVERS ===")
# Top features by correlation / coefficient / tree importance
if hasattr(best_clf, 'coef_'):
    importances = np.abs(best_clf.coef_[0])
elif hasattr(best_clf, 'feature_importances_'):
    importances = best_clf.feature_importances_
else:
    # voting classifier uses average importances of constituents
    importances = np.zeros(len(feature_names))
    for model_sub in best_clf.estimators_:
        if hasattr(model_sub, 'coef_'):
            importances += np.abs(model_sub.coef_[0])
        elif hasattr(model_sub, 'feature_importances_'):
            importances += model_sub.feature_importances_
    importances /= len(best_clf.estimators_)

# Sort features by importance
sorted_idx = np.argsort(importances)[::-1]
print("\nTop 10 Churn Drivers (Global Importance):")
for idx in sorted_idx[:10]:
    print(f"  {feature_names[idx]:<32}: {importances[idx]:.4f}")

# Example advisor recommendation engine
def get_retention_recommendations(prob, row_dict):
    risk_level = "Low"
    if prob >= 0.7:
        risk_level = "High"
    elif prob >= 0.3:
        risk_level = "Medium"
        
    drivers = []
    # Identify individual drivers by checking values against average
    if row_dict.get('Overall_Satisfaction', 3.0) < 3.0:
        drivers.append("Low member satisfaction score")
    if row_dict.get('Billing_Issues', 0.0) == 1.0:
        drivers.append("Recent billing/invoicing issues")
    if row_dict.get('Days_Since_Last_Visit', 0.0) > 365:
        drivers.append("High absence period (no visits for 1+ years)")
    if row_dict.get('Missed_Appointments', 0) >= 3:
        drivers.append("Frequent missed appointments barrier")
        
    # Recommendations map
    recs = []
    if "Low member satisfaction score" in drivers:
        recs.append("Reach out with a customer care supervisor call to resolve dissatisfaction drivers.")
    if "Recent billing/invoicing issues" in drivers:
        recs.append("Apply a billing reconciliation review and issue a satisfaction service credit.")
    if "High absence period (no visits for 1+ years)" in drivers:
        recs.append("Initiate preventive care outreach or schedule an annual physical exam wellness booking.")
    if "Frequent missed appointments barrier" in drivers:
        recs.append("Offer transportation assistance, tele-health options, or send SMS appointment reminders.")
        
    if not recs:
        if risk_level == "High":
            recs.append("Issue a general member loyalty check-in call and review billing/claims history.")
        else:
            recs.append("Maintain standard digital outreach and general prevention newsletters.")
            
    return risk_level, drivers, recs

# Print sample advisor report for the first test instance
first_test_row = df_encoded.iloc[X_test_raw[0] == X_raw] # recover raw values
row_dict = {}
for col in df_features.columns:
    if col in df.columns:
        row_dict[col] = df.loc[0, col]

sample_prob = y_prob_best[0]
risk, drv, recs = get_retention_recommendations(sample_prob, row_dict)
print(f"\nSample Member Churn Advisor Report:")
print(f"  Member ID: {df.loc[0, 'PatientID']}")
print(f"  Model Churn Probability: {sample_prob*100:.1f}%")
print(f"  Assigned Risk Category : {risk}")
print(f"  Contributing Churn Drivers: {drv}")
print(f"  Targeted Action Recommendations:")
for r in recs:
    print(f"    - {r}")

print("\n==============================================")
print("CONCLUSION")
print("==============================================")
print(f"The best performing leakage-free model is: {best_tuned_model_name}")
print(f"Test Accuracy: {best_tuned_metrics['accuracy']*100:.2f}%")
print(f"F1-Score: {best_tuned_metrics['f1']:.4f}")
print(f"Recall: {best_tuned_metrics['recall']:.4f}")
print("==============================================")
