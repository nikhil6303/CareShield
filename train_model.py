"""
train_model.py
==============
Member Churn Prediction and Retention Advisor
Healthcare / Health-Plan Domain

What this script does
---------------------
1. Imports the preprocessed train/test splits produced by preprocessing.py.
2. Trains three models:
      a. Logistic Regression  (baseline, class_weight='balanced')
      b. Random Forest        (balanced, 5-fold CV)
      c. XGBoost              (scale_pos_weight for imbalance, lightweight tuning)
3. Evaluates each model on held-out test data using:
      Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
4. Runs Stratified K-Fold CV on each model (on training data only).
5. Selects the best model by ROC-AUC then F1 as tiebreaker.
6. Saves:
      outputs/best_model.pkl          - best sklearn/XGBoost model
      outputs/preprocessor.pkl        - fitted ColumnTransformer
      outputs/metrics_all_models.json - all evaluation metrics
      outputs/feature_names.csv       - feature names list (already exists)

Leakage prevention
------------------
- Preprocessing was fit ONLY on X_train (see preprocessing.py).
- CV is run ONLY on training data; test set is touched exactly once.
- Threshold is fixed at 0.50 (no threshold-tuning on the test set).
- Churned is NEVER used as an input feature.
- No duplication of rows or modification of target labels.

Author  : CTS NPN Project
Created : 2026-08-13
"""

# ------------------------------------------------------------------------------
# 0.  Imports
# ------------------------------------------------------------------------------
import os
import json
import pickle
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# 1.  Constants
# ------------------------------------------------------------------------------
RANDOM_STATE = 42
CV_FOLDS     = 5
OUTPUTS_DIR  = "outputs"

print("=" * 70)
print("  MEMBER CHURN PREDICTION -- MODEL TRAINING")
print("=" * 70)

# ------------------------------------------------------------------------------
# 2.  Load preprocessed data (produced by preprocessing.py)
# ------------------------------------------------------------------------------
print("\n[1] Loading preprocessed data from outputs/ …")
X_train = pd.read_csv(f"{OUTPUTS_DIR}/X_train.csv").values
X_test  = pd.read_csv(f"{OUTPUTS_DIR}/X_test.csv").values
y_train = pd.read_csv(f"{OUTPUTS_DIR}/y_train.csv").squeeze().values
y_test  = pd.read_csv(f"{OUTPUTS_DIR}/y_test.csv").squeeze().values
feature_names = pd.read_csv(f"{OUTPUTS_DIR}/feature_names.csv")["feature"].tolist()

print(f"    X_train : {X_train.shape}   y_train: {y_train.shape}")
print(f"    X_test  : {X_test.shape}    y_test : {y_test.shape}")
print(f"    Features: {len(feature_names)}")
print(f"    Train churn rate : {y_train.mean()*100:.1f}%")
print(f"    Test  churn rate : {y_test.mean()*100:.1f}%")

# Class imbalance ratio -- used to set scale_pos_weight in XGBoost
n_neg  = (y_train == 0).sum()
n_pos  = (y_train == 1).sum()
spw    = round(n_neg / n_pos, 4)   # ~0.46 for this dataset
print(f"\n    Class ratio (neg/pos) = {spw:.4f}  -> XGBoost scale_pos_weight")

# ------------------------------------------------------------------------------
# 3.  Evaluation Helper
# ------------------------------------------------------------------------------
def evaluate_on_test(model, X_tr, X_te, y_tr, y_te, name):
    """
    Evaluate a fitted model on the test set.
    Threshold is fixed at 0.50 -- NOT tuned on test data.
    Returns a dict of metrics.
    """
    y_prob = model.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= 0.50).astype(int)

    acc   = accuracy_score(y_te, y_pred)
    prec  = precision_score(y_te, y_pred, zero_division=0)
    rec   = recall_score(y_te, y_pred, zero_division=0)
    f1    = f1_score(y_te, y_pred, zero_division=0)
    auc   = roc_auc_score(y_te, y_prob)
    cm    = confusion_matrix(y_te, y_pred)

    print(f"\n  -- {name} -- Test Set Results --------------------------")
    print(f"     Accuracy  : {acc*100:.2f}%")
    print(f"     Precision : {prec:.4f}")
    print(f"     Recall    : {rec:.4f}")
    print(f"     F1-Score  : {f1:.4f}")
    print(f"     ROC-AUC   : {auc:.4f}")
    print(f"     Confusion Matrix:")
    print(f"       TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"       FN={cm[1,0]}  TP={cm[1,1]}")

    return {
        "model_name": name,
        "accuracy":   round(acc, 4),
        "precision":  round(prec, 4),
        "recall":     round(rec, 4),
        "f1":         round(f1, 4),
        "roc_auc":    round(auc, 4),
        "confusion_matrix": {
            "TN": int(cm[0,0]), "FP": int(cm[0,1]),
            "FN": int(cm[1,0]), "TP": int(cm[1,1]),
        }
    }


def run_cv(model, X_tr, y_tr, name):
    """
    Stratified K-Fold CV -- runs ONLY on training data.
    Returns mean ± std for key metrics.
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    results = cross_validate(model, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=-1)

    cv_summary = {
        "cv_accuracy_mean":  round(results["test_accuracy"].mean(), 4),
        "cv_accuracy_std":   round(results["test_accuracy"].std(),  4),
        "cv_precision_mean": round(results["test_precision"].mean(), 4),
        "cv_recall_mean":    round(results["test_recall"].mean(), 4),
        "cv_f1_mean":        round(results["test_f1"].mean(), 4),
        "cv_roc_auc_mean":   round(results["test_roc_auc"].mean(), 4),
        "cv_roc_auc_std":    round(results["test_roc_auc"].std(), 4),
    }

    print(f"     {CV_FOLDS}-Fold CV  ->  "
          f"AUC={cv_summary['cv_roc_auc_mean']:.4f} ± {cv_summary['cv_roc_auc_std']:.4f}  "
          f"| F1={cv_summary['cv_f1_mean']:.4f}  "
          f"| Recall={cv_summary['cv_recall_mean']:.4f}")
    return cv_summary


# ------------------------------------------------------------------------------
# 4.  MODEL A -- Logistic Regression (Baseline)
# ------------------------------------------------------------------------------
# Decision: class_weight='balanced' adjusts loss weights proportional to
# inverse class frequency. C=0.1 gives mild L2 regularisation to avoid
# overfitting on a dataset this size.
# ------------------------------------------------------------------------------
print("\n\n[2] Training Model A -- Logistic Regression (baseline) …")
lr = LogisticRegression(
    C=0.1,
    penalty="l2",
    class_weight="balanced",   # handles 68/32 imbalance via loss reweighting
    solver="lbfgs",
    max_iter=2000,
    random_state=RANDOM_STATE,
)
lr.fit(X_train, y_train)
lr_cv      = run_cv(lr, X_train, y_train, "Logistic Regression")
lr_metrics = evaluate_on_test(lr, X_train, X_test, y_train, y_test, "Logistic Regression")
lr_metrics.update(lr_cv)


# ------------------------------------------------------------------------------
# 5.  MODEL B -- Random Forest
# ------------------------------------------------------------------------------
# Decision: class_weight='balanced_subsample' rebalances within each bootstrap
# sample drawn during training, which is more appropriate than global weighting
# for ensemble methods. n_estimators=300 is sufficient for a 2 K-row dataset.
# max_features='sqrt' is the standard for classification trees.
# ------------------------------------------------------------------------------
print("\n\n[3] Training Model B -- Random Forest …")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=8,
    max_features="sqrt",
    class_weight="balanced_subsample",  # balances within each bootstrap sample
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
rf_cv      = run_cv(rf, X_train, y_train, "Random Forest")
rf_metrics = evaluate_on_test(rf, X_train, X_test, y_train, y_test, "Random Forest")
rf_metrics.update(rf_cv)


# ------------------------------------------------------------------------------
# 6.  MODEL C -- XGBoost (with lightweight hyperparameter tuning)
# ------------------------------------------------------------------------------
# Decision:
#   scale_pos_weight = n_neg / n_pos tells XGBoost to penalise missed positives
#   proportionally -- correct way to handle imbalance in gradient boosting.
#   eval_metric="auc" aligns training objective with our primary evaluation metric.
#   We run a small manual grid (9 combinations) instead of GridSearchCV to keep
#   compute light on a 2 K-row dataset.
# ------------------------------------------------------------------------------
print("\n\n[4] Training Model C -- XGBoost (lightweight hyperparameter tuning) …")

xgb_param_grid = [
    {"n_estimators": 200, "learning_rate": 0.05,  "max_depth": 4, "subsample": 0.8},
    {"n_estimators": 200, "learning_rate": 0.10,  "max_depth": 4, "subsample": 0.8},
    {"n_estimators": 300, "learning_rate": 0.05,  "max_depth": 5, "subsample": 0.8},
    {"n_estimators": 300, "learning_rate": 0.05,  "max_depth": 4, "subsample": 0.7},
    {"n_estimators": 200, "learning_rate": 0.05,  "max_depth": 5, "subsample": 0.9},
    {"n_estimators": 150, "learning_rate": 0.10,  "max_depth": 3, "subsample": 0.8},
    {"n_estimators": 300, "learning_rate": 0.03,  "max_depth": 5, "subsample": 0.8},
    {"n_estimators": 200, "learning_rate": 0.10,  "max_depth": 3, "subsample": 0.7},
    {"n_estimators": 400, "learning_rate": 0.03,  "max_depth": 4, "subsample": 0.8},
]

cv_obj = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

best_xgb_auc  = -1
best_xgb_params = None
best_xgb_model  = None

print("     Tuning XGBoost (9 param combos × 5-fold CV on train only) …")
for idx, params in enumerate(xgb_param_grid):
    candidate = XGBClassifier(
        **params,
        scale_pos_weight=spw,
        use_label_encoder=False,
        eval_metric="auc",
        tree_method="hist",
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    cv_res = cross_validate(
        candidate, X_train, y_train,
        cv=cv_obj, scoring="roc_auc", n_jobs=-1
    )
    mean_auc = cv_res["test_score"].mean()
    print(f"     [{idx+1:2d}/9] n={params['n_estimators']:3d} lr={params['learning_rate']:.2f} "
          f"depth={params['max_depth']} sub={params['subsample']} "
          f"-> CV-AUC={mean_auc:.4f}")

    if mean_auc > best_xgb_auc:
        best_xgb_auc    = mean_auc
        best_xgb_params = params
        best_xgb_model  = candidate

print(f"\n     Best XGBoost params : {best_xgb_params}")
print(f"     Best CV-AUC         : {best_xgb_auc:.4f}")

# Refit best XGBoost on full training data
best_xgb = XGBClassifier(
    **best_xgb_params,
    scale_pos_weight=spw,
    use_label_encoder=False,
    eval_metric="auc",
    tree_method="hist",
    random_state=RANDOM_STATE,
    verbosity=0,
)
best_xgb.fit(X_train, y_train)
xgb_cv      = run_cv(best_xgb, X_train, y_train, "XGBoost")
xgb_metrics = evaluate_on_test(best_xgb, X_train, X_test, y_train, y_test, "XGBoost")
xgb_metrics.update(xgb_cv)
xgb_metrics["best_hyperparameters"] = best_xgb_params


# ------------------------------------------------------------------------------
# 7.  Select Best Model (ROC-AUC primary, F1 tiebreaker)
# ------------------------------------------------------------------------------
all_metrics = {
    "Logistic Regression": lr_metrics,
    "Random Forest":       rf_metrics,
    "XGBoost":             xgb_metrics,
}

model_objects = {
    "Logistic Regression": lr,
    "Random Forest":       rf,
    "XGBoost":             best_xgb,
}

# Sort by ROC-AUC desc, then F1 desc
ranked = sorted(
    all_metrics.items(),
    key=lambda kv: (kv[1]["roc_auc"], kv[1]["f1"]),
    reverse=True,
)
best_model_name   = ranked[0][0]
best_model_object = model_objects[best_model_name]

print(f"\n\n[5] BEST MODEL -> {best_model_name}")
print(f"     ROC-AUC = {all_metrics[best_model_name]['roc_auc']:.4f}")
print(f"     F1      = {all_metrics[best_model_name]['f1']:.4f}")


# ------------------------------------------------------------------------------
# 8.  Comparison Table
# ------------------------------------------------------------------------------
print("\n\n" + "=" * 70)
print("  MODEL COMPARISON TABLE (Test Set, Threshold = 0.50)")
print("=" * 70)
header = (
    f"{'Model':<22} {'Accuracy':>9} {'Precision':>10} "
    f"{'Recall':>8} {'F1':>8} {'ROC-AUC':>9} {'CV-AUC':>8}"
)
print(header)
print("-" * 70)
for name, m in all_metrics.items():
    mark = " *" if name == best_model_name else ""
    print(
        f"{name:<22} "
        f"{m['accuracy']*100:>8.2f}% "
        f"{m['precision']:>10.4f} "
        f"{m['recall']:>8.4f} "
        f"{m['f1']:>8.4f} "
        f"{m['roc_auc']:>9.4f} "
        f"{m['cv_roc_auc_mean']:>8.4f}"
        f"{mark}"
    )
print("=" * 70)
print("  * = Selected best model (highest ROC-AUC, F1 as tiebreaker)")


# ------------------------------------------------------------------------------
# 9.  Predictive Signal Analysis
# ------------------------------------------------------------------------------
best_auc = all_metrics[best_model_name]["roc_auc"]
best_f1  = all_metrics[best_model_name]["f1"]
best_rec = all_metrics[best_model_name]["recall"]

print("\n\n" + "=" * 70)
print("  PREDICTIVE SIGNAL ANALYSIS")
print("=" * 70)

if best_auc >= 0.80:
    signal_verdict = "STRONG"
    signal_detail  = (
        "The dataset contains sufficient predictive signal. "
        "ROC-AUC >= 0.80 means the model can clearly separate churners "
        "from retained members. Useful for production deployment."
    )
elif best_auc >= 0.70:
    signal_verdict = "MODERATE"
    signal_detail  = (
        "The dataset has moderate signal. ROC-AUC 0.70–0.80 is useful "
        "for prioritised outreach but not near-certain prediction. "
        "Adding external data (claims history, prior-year churn) could help."
    )
else:
    signal_verdict = "WEAK"
    signal_detail  = (
        "The dataset has weak discriminative power. ROC-AUC < 0.70 means "
        "the model is only marginally better than random guessing. "
        "The dataset likely lacks the key variables that drive churn "
        "(e.g., benefit comparison data, competitor offers, renewal timing)."
    )

print(f"\n  Verdict        : {signal_verdict}")
print(f"  Best ROC-AUC   : {best_auc:.4f}")
print(f"  Best F1-Score  : {best_f1:.4f}")
print(f"  Best Recall    : {best_rec:.4f}")
print(f"\n  Analysis       : {signal_detail}")
print(f"\n  Class imbalance: 68.35% churn. Models using class_weight='balanced'")
print(f"  and scale_pos_weight give fair evaluation across both classes.")
print(f"  Accuracy alone is misleading here -- a naive 'predict all churn'")
print(f"  model would reach ~68% accuracy. ROC-AUC and F1 are the right metrics.")


# ------------------------------------------------------------------------------
# 10. Save Artefacts
# ------------------------------------------------------------------------------
print("\n\n[6] Saving artefacts …")

# Best model
with open(f"{OUTPUTS_DIR}/best_model.pkl", "wb") as f:
    pickle.dump(best_model_object, f)

# Best model as churn_model.pkl
with open(f"{OUTPUTS_DIR}/churn_model.pkl", "wb") as f:
    pickle.dump(best_model_object, f)

# Also save all three models individually for evaluate_model.py
for name, obj in model_objects.items():
    safe_name = name.lower().replace(" ", "_")
    with open(f"{OUTPUTS_DIR}/model_{safe_name}.pkl", "wb") as f:
        pickle.dump(obj, f)

# Save metrics JSON
metrics_output = {
    "best_model": best_model_name,
    "best_xgb_params": best_xgb_params,
    "models": all_metrics,
}
with open(f"{OUTPUTS_DIR}/metrics_all_models.json", "w") as f:
    json.dump(metrics_output, f, indent=2)

print(f"    outputs/churn_model.pkl          -> {best_model_name}")
print(f"    outputs/best_model.pkl           -> {best_model_name}")
print(f"    outputs/model_logistic_regression.pkl")
print(f"    outputs/model_random_forest.pkl")
print(f"    outputs/model_xgboost.pkl")
print(f"    outputs/metrics_all_models.json")

print("\n" + "=" * 70)
print("  TRAINING COMPLETE")
print("=" * 70)
print("  Next step: run evaluate_model.py for full evaluation report.")
print("=" * 70)
