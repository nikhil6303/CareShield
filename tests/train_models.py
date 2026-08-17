import pandas as pd
import numpy as np
import json
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc, confusion_matrix)

warnings.filterwarnings('ignore')


def find_best_threshold(y_true, y_prob):
    """Find the probability threshold that maximises accuracy on the training set."""
    best_thresh, best_acc = 0.5, 0.0
    for thresh in np.arange(0.20, 0.85, 0.005):
        y_pred = (y_prob >= thresh).astype(int)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc, best_thresh = acc, thresh
    return best_thresh


def build_features(df):
    df_fe = df.copy()

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

    # 6. Support/contact frequency & frustration
    df_fe['Billing_Dissatisfaction'] = df_fe['Billing_Issues'] * (5.0 - df_fe['Overall_Satisfaction'])
    df_fe['Billing_And_Poor_Wait'] = df_fe['Billing_Issues'] * (5.0 - df_fe['Wait_Time_Satisfaction'])

    # 7. Access / distance
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
    df_fe['Min_Satisfaction'] = df_fe[['Overall_Satisfaction', 'Wait_Time_Satisfaction',
                                        'Staff_Satisfaction', 'Provider_Rating']].min(axis=1)
    df_fe['Sat_Variance'] = df_fe[['Overall_Satisfaction', 'Wait_Time_Satisfaction',
                                    'Staff_Satisfaction', 'Provider_Rating']].var(axis=1)

    # Gender encoding
    df_fe['Gender_Male'] = (df_fe['Gender'] == 'Male').astype(int)
    df_fe = df_fe.drop(columns=['Gender'])

    # One-hot encode categoricals
    categorical_cols = ['State', 'Specialty', 'Insurance_Type']
    categories = {}
    for col in categorical_cols:
        unique_vals = sorted(df_fe[col].unique())
        categories[col] = unique_vals
        for val in unique_vals:
            df_fe[f'{col}_{val}'] = (df_fe[col] == val).astype(int)
        df_fe = df_fe.drop(columns=[col])

    # Private/Self-Pay cost interaction
    df_fe['Private_SelfPay_Cost'] = df_fe['Avg_Out_Of_Pocket_Cost'] * (
        df_fe['Insurance_Type_Private'] + df_fe['Insurance_Type_Self-Pay']
    )

    feature_names = list(df_fe.columns)

    binary_cols = ['Billing_Issues', 'Portal_Usage', 'Gender_Male',
                   'Long_Absent', 'Very_Long_Absent']
    for col in feature_names:
        if any(c in col for c in ['State_', 'Specialty_', 'Insurance_Type_']):
            binary_cols.append(col)
    numeric_cols = [c for c in feature_names if c not in binary_cols]

    return df_fe, feature_names, binary_cols, numeric_cols, categories


def evaluate(clf, X_train, X_test, y_train, y_test):
    y_prob_train = clf.predict_proba(X_train)[:, 1]
    y_prob_test = clf.predict_proba(X_test)[:, 1]
    threshold = find_best_threshold(y_train, y_prob_train)
    y_pred = (y_prob_test >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y_test, y_prob_test)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_test, y_pred)
    step = max(1, len(fpr) // 20)
    return {
        'accuracy':  float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall':    float(recall_score(y_test, y_pred)),
        'f1':        float(f1_score(y_test, y_pred)),
        'auc':       float(roc_auc),
        'cm':        [[int(v) for v in row] for row in cm],
        'threshold': float(threshold),
        'roc_curve': {
            'fpr': [float(x) for x in fpr[::step]],
            'tpr': [float(x) for x in tpr[::step]]
        }
    }


def train_and_export():
    print("Step 1: Loading dataset...")
    df = pd.read_csv("patient_churn_dataset.csv.xls")

    print("Step 2: Feature engineering...")
    df_fe, feature_names, binary_cols, numeric_cols, categories = build_features(df)
    X_raw = df_fe.drop(columns=['PatientID', 'Last_Interaction_Date', 'Churned']).values
    y = df_fe['Churned'].values

    feature_names = [f for f in feature_names if f not in ['PatientID', 'Last_Interaction_Date', 'Churned']]
    print(f"         Total features: {len(feature_names)}")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y)

    # Scaling correctly fit on train only (leakage-free!)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # Save scaler parameters mapped to feature names (for simulator use)
    scaler_params = {}
    for idx, col in enumerate(feature_names):
        if col in numeric_cols:
            scaler_params[col] = {
                'mean': float(scaler.mean_[idx]),
                'std': float(scaler.scale_[idx])
            }

    # ── LOGISTIC REGRESSION ──────────────────────────────────────────────────
    print("Step 3: Training Logistic Regression (L1, C=0.05)...")
    lr = LogisticRegression(C=0.05, penalty='l1', solver='liblinear', random_state=42)
    lr.fit(X_train, y_train)
    lr_metrics = evaluate(lr, X_train, X_test, y_train, y_test)
    print(f"         Acc={lr_metrics['accuracy']:.4f}  Recall={lr_metrics['recall']:.4f}  F1={lr_metrics['f1']:.4f}  Th={lr_metrics['threshold']:.3f}")

    lr_model_data = {
        'intercept': float(lr.intercept_[0]),
        'coefficients': {n: float(c) for n, c in zip(feature_names, lr.coef_[0])}
    }

    # ── DECISION TREE ─────────────────────────────────────────────────────────
    print("Step 4: Training Decision Tree (entropy, depth=3, leaf=20)...")
    dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, criterion='entropy', random_state=42)
    dt.fit(X_train, y_train)
    dt_metrics = evaluate(dt, X_train, X_test, y_train, y_test)
    print(f"         Acc={dt_metrics['accuracy']:.4f}  Recall={dt_metrics['recall']:.4f}  F1={dt_metrics['f1']:.4f}  Th={dt_metrics['threshold']:.3f}")

    tree = dt.tree_
    tree_nodes = []
    for i in range(tree.node_count):
        left  = int(tree.children_left[i])
        right = int(tree.children_right[i])
        fi    = int(tree.feature[i])
        val   = tree.value[i][0]
        tot   = float(np.sum(val))
        tree_nodes.append({
            'id': i,
            'is_leaf': bool(left == -1),
            'left_child': left,
            'right_child': right,
            'feature': feature_names[fi] if fi != -2 else None,
            'threshold': float(tree.threshold[i]),
            'churn_probability': float(val[1] / tot) if tot > 0 else 0.0,
            'samples': int(tot)
        })

    # ── ADABOOST ──────────────────────────────────────────────────────────────
    print("Step 5: Training AdaBoost (depth=2, n=300, lr=0.3)...")
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=2),
        n_estimators=300, learning_rate=0.3, random_state=42)
    ada.fit(X_train, y_train)
    ada_metrics = evaluate(ada, X_train, X_test, y_train, y_test)
    print(f"         Acc={ada_metrics['accuracy']:.4f}  Recall={ada_metrics['recall']:.4f}  F1={ada_metrics['f1']:.4f}  Th={ada_metrics['threshold']:.3f}")

    ada_importances = {n: float(v) for n, v in zip(feature_names, ada.feature_importances_)}

    # ── Compile output ────────────────────────────────────────────────────────
    output_data = {
        'feature_names': feature_names,
        'binary_cols':   binary_cols,
        'numeric_cols':  numeric_cols,
        'categorical_source': categories,
        'scaler_params': scaler_params,
        'logistic_regression': {
            'metrics': lr_metrics,
            'model': lr_model_data
        },
        'decision_tree': {
            'metrics': dt_metrics,
            'nodes': tree_nodes,
            'importances': {n: float(v) for n, v in zip(feature_names, dt.feature_importances_)}
        },
        'adaboost': {
            'metrics': ada_metrics,
            'importances': ada_importances
        }
    }

    print("\nStep 6: Writing model_data.js...")
    js_content = (
        "// Auto-generated by train_models.py - do not edit manually.\n"
        f"const MODEL_DATA = {json.dumps(output_data, indent=2)};\n"
    )
    with open("model_data.js", "w") as f:
        f.write(js_content)
    print("Done! model_data.js updated successfully.")


if __name__ == "__main__":
    train_and_export()
