"""
evaluate_model.py
=================
Member Churn Prediction and Retention Advisor
Healthcare / Health-Plan Domain

What this script does
---------------------
Loads the three saved models and the preprocessed test set, then produces:
  1. Full classification report for each model
  2. Confusion matrix (labelled)
  3. ROC-AUC per model
  4. Cross-validated metrics (loaded from JSON, no data re-touch)
  5. Final comparison table
  6. Predictive signal verdict with explanation
  7. Prints a short analysis of whether the dataset limits performance

This script NEVER touches training data -- it only reads from outputs/.

Usage
-----
    python evaluate_model.py

Requires
--------
    Run preprocessing.py first, then train_model.py.
    All required files must exist in outputs/.

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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve,
)

warnings.filterwarnings("ignore")

OUTPUTS_DIR = "outputs"

print("=" * 72)
print("  MEMBER CHURN PREDICTION -- MODEL EVALUATION REPORT")
print("=" * 72)


# ------------------------------------------------------------------------------
# 1.  Load test data and saved models
# ------------------------------------------------------------------------------
print("\n[1] Loading test data and models ")

X_test        = pd.read_csv(f"{OUTPUTS_DIR}/X_test.csv").values
y_test        = pd.read_csv(f"{OUTPUTS_DIR}/y_test.csv").squeeze().values
feature_names = pd.read_csv(f"{OUTPUTS_DIR}/feature_names.csv")["feature"].tolist()

with open(f"{OUTPUTS_DIR}/model_logistic_regression.pkl", "rb") as f:
    lr_model = pickle.load(f)
with open(f"{OUTPUTS_DIR}/model_random_forest.pkl", "rb") as f:
    rf_model = pickle.load(f)
with open(f"{OUTPUTS_DIR}/model_xgboost.pkl", "rb") as f:
    xgb_model = pickle.load(f)

# Load saved metrics JSON (produced by train_model.py)
with open(f"{OUTPUTS_DIR}/metrics_all_models.json", "r") as f:
    saved_metrics = json.load(f)

print(f"    Test set: {X_test.shape[0]} rows x {X_test.shape[1]} features")
print(f"    Test churn rate: {y_test.mean()*100:.1f}%")
print(f"    Best model (by train_model.py): {saved_metrics['best_model']}")

models = {
    "Logistic Regression": lr_model,
    "Random Forest":       rf_model,
    "XGBoost":             xgb_model,
}


# ------------------------------------------------------------------------------
# 2.  Per-model detailed evaluation
# ------------------------------------------------------------------------------
def full_eval(model, X_te, y_te, name):
    """Full evaluation at threshold 0.50 -- no test-set tuning."""
    y_prob = model.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= 0.50).astype(int)

    acc   = accuracy_score(y_te, y_pred)
    prec  = precision_score(y_te, y_pred, zero_division=0)
    rec   = recall_score(y_te, y_pred, zero_division=0)
    f1    = f1_score(y_te, y_pred, zero_division=0)
    auc   = roc_auc_score(y_te, y_prob)
    cm    = confusion_matrix(y_te, y_pred)
    report = classification_report(
        y_te, y_pred,
        target_names=["Retained (0)", "Churned (1)"],
        digits=4,
    )

    tn, fp, fn, tp = cm.ravel()
    sensitivity  = tp / (tp + fn) if (tp + fn) > 0 else 0   # = Recall
    specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv          = tp / (tp + fp) if (tp + fp) > 0 else 0   # = Precision
    npv          = tn / (tn + fn) if (tn + fn) > 0 else 0

    print(f"\n{'-'*72}")
    print(f"  MODEL: {name}")
    print(f"{'-'*72}")
    print(f"\n  Core Metrics (threshold = 0.50)")
    print(f"    Accuracy       : {acc*100:.2f}%")
    print(f"    Precision      : {prec:.4f}")
    print(f"    Recall         : {rec:.4f}")
    print(f"    F1-Score       : {f1:.4f}")
    print(f"    ROC-AUC        : {auc:.4f}")
    print(f"\n  Extended Metrics")
    print(f"    Sensitivity    : {sensitivity:.4f}  (same as Recall)")
    print(f"    Specificity    : {specificity:.4f}  (True Negative Rate)")
    print(f"    PPV            : {ppv:.4f}  (same as Precision)")
    print(f"    NPV            : {npv:.4f}  (Negative Predictive Value)")
    print(f"\n  Confusion Matrix")
    print(f"    {'':20s}  Predicted No Churn   Predicted Churn")
    print(f"    {'Actual No Churn':20s}  TN={tn:<6}              FP={fp}")
    print(f"    {'Actual Churn':20s}  FN={fn:<6}              TP={tp}")
    print(f"\n  Classification Report\n")
    for line in report.split("\n"):
        print(f"    {line}")

    return {
        "model_name":  name,
        "accuracy":    round(acc,  4),
        "precision":   round(prec, 4),
        "recall":      round(rec,  4),
        "f1":          round(f1,   4),
        "roc_auc":     round(auc,  4),
        "specificity": round(specificity, 4),
        "npv":         round(npv, 4),
        "cm": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "y_prob":      y_prob,
    }


print("\n[2] Running full evaluation on test set ")
eval_results = {}
for name, model in models.items():
    eval_results[name] = full_eval(model, X_test, y_test, name)


# ------------------------------------------------------------------------------
# 3.  Feature importances (top 15 per tree-based model)
# ------------------------------------------------------------------------------
print(f"\n\n{'-'*72}")
print("  FEATURE IMPORTANCES (Top 15)")
print(f"{'-'*72}")

for name in ["Random Forest", "XGBoost"]:
    model = models[name]
    imp   = model.feature_importances_
    idx   = np.argsort(imp)[::-1][:15]
    print(f"\n  {name}:")
    for rank, i in enumerate(idx, 1):
        print(f"    {rank:2d}. {feature_names[i]:<35} {imp[i]:.4f}")

# LR coefficients
lr_coef = np.abs(lr_model.coef_[0])
lr_idx  = np.argsort(lr_coef)[::-1][:15]
print(f"\n  Logistic Regression (|coefficients|):")
for rank, i in enumerate(lr_idx, 1):
    print(f"    {rank:2d}. {feature_names[i]:<35} {lr_coef[i]:.4f}")


# ------------------------------------------------------------------------------
# 4.  Cross-validated metrics (from train_model.py saved JSON)
# ------------------------------------------------------------------------------
print(f"\n\n{'-'*72}")
print("  CROSS-VALIDATION RESULTS (5-Fold, train data only)")
print(f"{'-'*72}")
print(f"\n  {'Model':<22} {'CV-AUC (mean)':>14} {'CV-AUC (+/-std)':>14} "
      f"{'CV-F1':>8} {'CV-Recall':>10}")
print("  " + "-" * 68)
for name, m in saved_metrics["models"].items():
    print(f"  {name:<22} {m['cv_roc_auc_mean']:>14.4f} "
          f"{m['cv_roc_auc_std']:>14.4f} "
          f"{m['cv_f1_mean']:>8.4f} "
          f"{m['cv_recall_mean']:>10.4f}")


# ------------------------------------------------------------------------------
# 5.  Final Comparison Table
# ------------------------------------------------------------------------------
best_name = saved_metrics["best_model"]

print(f"\n\n{'='*72}")
print("  FINAL MODEL COMPARISON TABLE (Test Set)")
print(f"{'='*72}")
print(f"  {'Model':<22} {'Acc%':>6} {'Prec':>7} {'Recall':>7} "
      f"{'F1':>7} {'AUC':>7} {'Specificity':>12} {'CV-AUC':>8}")
print("  " + "-" * 72)

for name, r in eval_results.items():
    cv_auc = saved_metrics["models"][name]["cv_roc_auc_mean"]
    mark   = " *" if name == best_name else ""
    print(
        f"  {name:<22} "
        f"{r['accuracy']*100:>5.1f}% "
        f"{r['precision']:>7.4f} "
        f"{r['recall']:>7.4f} "
        f"{r['f1']:>7.4f} "
        f"{r['roc_auc']:>7.4f} "
        f"{r['specificity']:>12.4f} "
        f"{cv_auc:>8.4f}"
        f"{mark}"
    )

print(f"{'='*72}")
print("  * = Best model by ROC-AUC (F1 as tiebreaker)")


# ------------------------------------------------------------------------------
# 6.  Predictive Signal & Dataset Limitation Analysis
# ------------------------------------------------------------------------------
best_auc = eval_results[best_name]["roc_auc"]
best_f1  = eval_results[best_name]["f1"]
best_rec = eval_results[best_name]["recall"]
best_acc = eval_results[best_name]["accuracy"]

print(f"\n\n{'='*72}")
print("  PREDICTIVE SIGNAL & DATASET LIMITATION ANALYSIS")
print(f"{'='*72}")

if best_auc >= 0.80:
    verdict = "STRONG SIGNAL"
    color   = "[OK]"
elif best_auc >= 0.70:
    verdict = "MODERATE SIGNAL"
    color   = "[!] "
else:
    verdict = "WEAK SIGNAL"
    color   = "[X]"

print(f"\n  {color} Signal Verdict : {verdict}  (ROC-AUC = {best_auc:.4f})")
print(f"\n  Best Model     : {best_name}")
print(f"  Test Accuracy  : {best_acc*100:.2f}%  "
      f"<- Note: naive 'all-churn' baseline = 68.35%")
print(f"  ROC-AUC        : {best_auc:.4f}  <- primary metric")
print(f"  F1-Score       : {best_f1:.4f}  <- secondary metric")
print(f"  Recall         : {best_rec:.4f}  <- critical for churn use-case")

print("""
  INTERPRETATION
  --------------
  1. ROC-AUC tells us how well the model separates churners from non-churners
     regardless of the threshold chosen. A score above 0.70 means the model
     adds meaningful value over a random outreach campaign.

  2. The dataset has a 68/32 class imbalance. All three models use balanced
     weighting, so Recall (catching actual churners) is prioritised over
     Precision. In healthcare retention, a missed churner (FN) costs more
     than a false alarm (FP).

  3. F1-score and ROC-AUC were used as the selection criteria -- NOT accuracy,
     which would be misleadingly inflated by the majority class.

  POTENTIAL DATASET LIMITATIONS
  ------------------------------
  The following factors may cap performance at the current level:

  a) Small dataset (2,000 rows): Tree-based models with many features may
     not fully generalise. More data (5 K+) would allow deeper tuning.

  b) No temporal features: The dataset lacks plan renewal dates, claims
     history, or competitor pricing signals -- all strong churn predictors
     in a real health-plan setting.

  c) Synthetic/bounded ranges: If the dataset was generated with constrained
     distributions (all states have equal samples, costs are uniformly spread),
     the model will learn a less nuanced boundary than it would from real data.

  d) Satisfaction scores: If these were measured simultaneously with or after
     the churn decision, they carry soft leakage that inflates apparent signal
     but degrades real-world performance.

  RECOMMENDATION
  --------------
  Use this model for risk stratification (High/Medium/Low buckets) rather
  than binary prediction. Focus intervention resources on the top 30% by
  churn probability. Track lift vs. random outreach to validate real value.
""")

print("=" * 72)
print("  EVALUATION COMPLETE")
print("=" * 72)
