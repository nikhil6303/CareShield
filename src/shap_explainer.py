"""
shap_explainer.py
=================
Member Churn Prediction and Retention Advisor
Healthcare / Health-Plan Domain

What this module does
---------------------
1. Loads the saved XGBoost model and the fitted preprocessing pipeline.
2. Runs SHAP TreeExplainer on the full test set for global importance.
3. Saves global SHAP plots to outputs/shap_plots/.
4. Exposes get_member_explanation(member_id) which returns a structured JSON
   dict with churn probability, risk level, and the top feature drivers.

Language policy (IMPORTANT)
----------------------------
All language in the output is EVIDENCE-BASED only:
  - Features are described by their values and measured association with risk.
  - No psychological, medical, emotional, or financial assumptions are made.
  - Correct:  "Billing issues are associated with increased predicted risk."
  - Incorrect: "The member is unhappy." / "The member cannot afford the plan."

Risk level thresholds
---------------------
  0.00 - 0.39  ->  Low
  0.40 - 0.69  ->  Medium
  0.70 - 0.84  ->  High
  0.85 - 1.00  ->  Critical

Usage
-----
    # Run the full SHAP analysis and save all plots:
    python shap_explainer.py

    # Import and call the API function:
    from shap_explainer import get_member_explanation
    result = get_member_explanation("C20000")
    print(result)

Author  : CTS NPN Project
Created : 2026-08-13
"""

# ──────────────────────────────────────────────────────────────────────────────
# 0.  Imports
# ──────────────────────────────────────────────────────────────────────────────
import os
import json
import pickle
import warnings
import sys

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Configuration
# ──────────────────────────────────────────────────────────────────────────────
OUTPUTS_DIR   = "outputs"
PLOTS_DIR     = os.path.join(OUTPUTS_DIR, "shap_plots")
DATA_PATH     = "data/dataset.csv" if os.path.exists("data/dataset.csv") else "patient_churn_dataset.csv.xls"
SNAPSHOT_DATE = pd.Timestamp("2026-08-13")
RANDOM_STATE  = 42
TEST_SIZE     = 0.20

# Risk level thresholds (lower bound inclusive)
RISK_THRESHOLDS = [
    (0.85, "Critical"),
    (0.70, "High"),
    (0.40, "Medium"),
    (0.00, "Low"),
]

# Number of top drivers to include in individual explanations
TOP_N_DRIVERS = 5

os.makedirs(PLOTS_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Preprocessing — mirrors preprocessing.py exactly
# ──────────────────────────────────────────────────────────────────────────────
NOMINAL_CATS = ["Gender", "State", "Specialty", "Insurance_Type"]
SAT_COLS     = [
    "Overall_Satisfaction", "Wait_Time_Satisfaction",
    "Staff_Satisfaction", "Provider_Rating",
]

# Human-readable labels for engineered and raw features
FEATURE_LABELS = {
    "Age":                      "Age (years)",
    "Tenure_Months":            "Membership tenure (months)",
    "Visits_Last_Year":         "Visits in last year",
    "Missed_Appointments":      "Missed appointments (count)",
    "Days_Since_Last_Visit":    "Days since last clinical visit",
    "Overall_Satisfaction":     "Overall satisfaction score",
    "Wait_Time_Satisfaction":   "Wait-time satisfaction score",
    "Staff_Satisfaction":       "Staff satisfaction score",
    "Provider_Rating":          "Provider rating",
    "Avg_Out_Of_Pocket_Cost":   "Average out-of-pocket cost (USD)",
    "Billing_Issues":           "Billing issue recorded (0/1)",
    "Portal_Usage":             "Patient portal usage (0/1)",
    "Referrals_Made":           "Referrals made (count)",
    "Distance_To_Facility_Miles": "Distance to facility (miles)",
    "days_since_interaction":   "Days since last recorded interaction",
    "interaction_month":        "Month of last interaction (1-12)",
    "missed_appt_rate":         "Missed appointment rate",
    "composite_satisfaction":   "Composite satisfaction score",
    "engagement_score":         "Engagement score (weighted)",
    "cost_per_tenure_month":    "Out-of-pocket cost per tenure month (USD)",
    "is_low_satisfaction":      "Low satisfaction flag (score < 2.5)",
    "is_distant":               "Distance barrier flag (> 40 miles)",
    "monthly_visit_rate":       "Monthly visit rate",
}

# Impact phrase templates — evidence-based only
DIRECTION_PHRASES = {
    "increases_risk": "is associated with increased predicted churn risk",
    "decreases_risk": "is associated with decreased predicted churn risk",
}


def _apply_feature_engineering(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same feature engineering steps as preprocessing.py.
    Works on a copy; does NOT modify the original dataframe.
    """
    df = df_raw.copy()

    # Date features
    df["Last_Interaction_Date"] = pd.to_datetime(df["Last_Interaction_Date"])
    df["days_since_interaction"] = (SNAPSHOT_DATE - df["Last_Interaction_Date"]).dt.days
    df["interaction_month"]      = df["Last_Interaction_Date"].dt.month
    df.drop(columns=["Last_Interaction_Date"], inplace=True)

    # Engineered features
    df["missed_appt_rate"]      = df["Missed_Appointments"] / df["Visits_Last_Year"].clip(lower=1)
    df["composite_satisfaction"]= df[SAT_COLS].mean(axis=1)
    df["engagement_score"]      = df["Visits_Last_Year"] + df["Portal_Usage"] * 2 + df["Referrals_Made"]
    df["cost_per_tenure_month"] = df["Avg_Out_Of_Pocket_Cost"] / df["Tenure_Months"].clip(lower=1)
    df["is_low_satisfaction"]   = (df["Overall_Satisfaction"] < 2.5).astype(int)
    df["is_distant"]            = (df["Distance_To_Facility_Miles"] > 40).astype(int)
    df["monthly_visit_rate"]    = df["Visits_Last_Year"] / 12

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Loaders
# ──────────────────────────────────────────────────────────────────────────────
def load_model():
    """Load the saved best model (XGBClassifier)."""
    path = os.path.join("model", "model.pkl")
    if not os.path.exists(path):
        path = os.path.join(OUTPUTS_DIR, "churn_model.pkl")
    if not os.path.exists(path):
        path = os.path.join(OUTPUTS_DIR, "best_model.pkl")
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def load_preprocessor():
    """Load the saved fitted ColumnTransformer and column lists."""
    path = os.path.join("model", "preprocessor.pkl")
    if not os.path.exists(path):
        path = os.path.join(OUTPUTS_DIR, "preprocessor.pkl")
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["preprocessor"], bundle["num_cols"], bundle["nominal_cats"]


def load_feature_names():
    """Load the 42 post-OHE feature names."""
    return pd.read_csv(
        os.path.join(OUTPUTS_DIR, "feature_names.csv")
    )["feature"].tolist()


def load_raw_dataset():
    """Load the original raw CSV with PatientID intact."""
    return pd.read_csv(DATA_PATH)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Risk Level Helper
# ──────────────────────────────────────────────────────────────────────────────
def assign_risk_level(probability: float) -> str:
    """
    Map a predicted probability to a named risk level.

    0.00 - 0.39  -> Low
    0.40 - 0.69  -> Medium
    0.70 - 0.84  -> High
    0.85 - 1.00  -> Critical
    """
    for threshold, label in RISK_THRESHOLDS:
        if probability >= threshold:
            return label
    return "Low"


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Preprocessing a single raw row
# ──────────────────────────────────────────────────────────────────────────────
def preprocess_member_row(raw_row: pd.Series, preprocessor) -> np.ndarray:
    """
    Apply feature engineering and the fitted ColumnTransformer to a single
    raw member row (as loaded from the original CSV, including PatientID).

    Returns a (1, 42) numpy array ready for model.predict_proba().
    """
    # Build a single-row DataFrame, drop ID and target columns that may exist
    cols_to_drop = [c for c in ["PatientID", "Churned"] if c in raw_row.index]
    row_df = pd.DataFrame([raw_row]).drop(columns=cols_to_drop, errors="ignore")

    # Apply feature engineering
    row_fe = _apply_feature_engineering(row_df)

    # Apply fitted ColumnTransformer (scales numerics, OHEs categoricals)
    return preprocessor.transform(row_fe)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  SHAP Background and Explainer
# ──────────────────────────────────────────────────────────────────────────────
def build_shap_explainer(model, X_background: np.ndarray):
    """
    Build a SHAP TreeExplainer for the XGBoost model.

    Mode: tree_path_dependent
    -------------------------
    Because the model was trained on already-encoded (OHE + scaled) data,
    we use tree_path_dependent perturbation.  This mode does NOT require a
    background dataset — SHAP derives the marginal baseline from the tree
    structure itself, making it fast and exact for XGBoost.

    TreeExplainer is orders of magnitude faster than KernelExplainer and
    gives exact (not approximate) Shapley values for tree-based models.
    """
    print("[SHAP] Building TreeExplainer (tree_path_dependent mode) ...")
    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent",
        model_output="raw",   # log-odds; we convert to probability later
    )
    ev = explainer.expected_value
    ev_scalar = float(ev[0]) if hasattr(ev, '__len__') else float(ev)
    print(f"[SHAP] Expected value (log-odds baseline): {ev_scalar:.4f}")
    return explainer


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Global SHAP Plots
# ──────────────────────────────────────────────────────────────────────────────
def generate_global_plots(shap_values, X_test_df: pd.DataFrame, feature_names: list):
    """
    Generate and save:
      a. Beeswarm summary plot  — shows distribution of SHAP values per feature
      b. Bar chart              — mean |SHAP| per feature (global importance)
    """
    print("[SHAP] Generating global plots ...")

    # ── a. Beeswarm summary plot ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 9))
    plt.sca(ax)
    shap.summary_plot(
        shap_values,
        X_test_df,
        feature_names=feature_names,
        plot_type="dot",
        max_display=20,
        show=False,
        color_bar_label="Feature value (normalised)",
    )
    plt.title(
        "SHAP Summary Plot — Feature Impact on Churn Prediction\n"
        "(Red = high feature value, Blue = low feature value; "
        "right of 0 = increases predicted churn risk)",
        fontsize=10, pad=14,
    )
    plt.tight_layout()
    path_a = os.path.join(PLOTS_DIR, "shap_beeswarm_summary.png")
    plt.savefig(path_a, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path_a}")

    # ── b. Bar chart (mean |SHAP|) ────────────────────────────────────────────
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx    = np.argsort(mean_abs_shap)[::-1][:20]
    top_features  = [feature_names[i] for i in sorted_idx]
    top_vals      = mean_abs_shap[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#E84040" if v > 0 else "#4A90D9" for v in top_vals]
    bars   = ax.barh(range(len(top_features)), top_vals[::-1], color=colors[::-1], edgecolor="white", height=0.7)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels([FEATURE_LABELS.get(f, f) for f in top_features[::-1]], fontsize=9)
    ax.set_xlabel("Mean |SHAP Value|  (average impact on predicted churn probability)", fontsize=9)
    ax.set_title(
        "Global Feature Importance (SHAP)\n"
        "Higher bar = larger average contribution to churn probability",
        fontsize=11, fontweight="bold", pad=12,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path_b = os.path.join(PLOTS_DIR, "shap_global_importance_bar.png")
    plt.savefig(path_b, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path_b}")

    return mean_abs_shap, sorted_idx


def generate_individual_waterfall(
    explainer,
    shap_values_row: np.ndarray,
    processed_row: np.ndarray,
    feature_names: list,
    member_id: str,
    churn_prob: float,
    risk_level: str,
):
    """
    Save a SHAP waterfall plot for a single member showing feature contributions.
    """
    ev = explainer.expected_value
    ev_scalar = float(ev[0]) if hasattr(ev, '__len__') else float(ev)
    explanation = shap.Explanation(
        values=shap_values_row,
        base_values=ev_scalar,
        data=processed_row,
        feature_names=feature_names,
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.sca(ax)
    shap.waterfall_plot(explanation, max_display=15, show=False)
    plt.title(
        f"SHAP Waterfall — Member {member_id}\n"
        f"Predicted churn probability: {churn_prob:.1%}  |  Risk level: {risk_level}",
        fontsize=10, pad=14,
    )
    plt.tight_layout()
    safe_id  = member_id.replace("/", "_").replace("\\", "_")
    path_out = os.path.join(PLOTS_DIR, f"shap_waterfall_{safe_id}.png")
    plt.savefig(path_out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Waterfall saved: {path_out}")
    return path_out


# ──────────────────────────────────────────────────────────────────────────────
# 8.  Core API — get_member_explanation(member_id)
# ──────────────────────────────────────────────────────────────────────────────

# Module-level cache so repeated calls don't reload from disk
_cache = {}


def _get_loaded_assets():
    """
    Lazily load and cache model, preprocessor, feature names, raw dataset,
    SHAP explainer, and full SHAP values.  Subsequent calls return instantly.
    """
    if "ready" in _cache:
        return _cache

    print("[SHAP] Loading assets ...")
    model        = load_model()
    preprocessor, num_cols, nominal_cats = load_preprocessor()
    feature_names = load_feature_names()
    raw_df       = load_raw_dataset()

    # Build background matrix from training data (same split as preprocessing.py)
    X_train = pd.read_csv(os.path.join(OUTPUTS_DIR, "X_train.csv")).values

    # Build explainer
    explainer = build_shap_explainer(model, None)

    # Pre-compute SHAP values for the full test set (used for global plots)
    X_test = pd.read_csv(os.path.join(OUTPUTS_DIR, "X_test.csv")).values
    print("[SHAP] Computing SHAP values for full test set ...")
    shap_values_test = explainer.shap_values(X_test)
    print(f"[SHAP] SHAP values shape: {shap_values_test.shape}")

    _cache.update({
        "ready":          True,
        "model":          model,
        "preprocessor":   preprocessor,
        "feature_names":  feature_names,
        "raw_df":         raw_df,
        "explainer":      explainer,
        "shap_values_test": shap_values_test,
        "X_test":         X_test,
    })
    return _cache


def get_member_explanation(
    member_id: str,
    top_n: int = TOP_N_DRIVERS,
    save_waterfall: bool = True,
) -> dict:
    """
    Generate a structured SHAP explanation for a single member.

    Parameters
    ----------
    member_id : str
        The PatientID from the raw dataset (e.g., "C20000").
    top_n : int, optional
        Number of top feature drivers to include (default 5).
    save_waterfall : bool, optional
        Whether to save a waterfall PNG for this member (default True).

    Returns
    -------
    dict with keys:
        member_id   : str
        risk_score  : float  (churn probability, rounded to 4dp)
        risk_level  : str    (Low / Medium / High / Critical)
        drivers     : list of dicts:
            feature    : str   (internal feature name)
            label      : str   (human-readable label)
            value      : str   (actual value of the feature, formatted)
            shap_value : float (raw SHAP contribution to probability)
            impact     : str   (evidence-based sentence)
            direction  : str   ("increases_risk" or "decreases_risk")
        waterfall_plot : str | None   (path to saved PNG, or None)
        error          : str | None   (None if successful)

    Language policy
    ---------------
    All driver descriptions use evidence-based language only.
    No psychological, emotional, financial, or medical assumptions.
    """
    assets = _get_loaded_assets()
    model         = assets["model"]
    preprocessor  = assets["preprocessor"]
    feature_names = assets["feature_names"]
    raw_df        = assets["raw_df"]
    explainer     = assets["explainer"]

    # ── Locate member row ─────────────────────────────────────────────────────
    match = raw_df[raw_df["PatientID"] == member_id]
    if match.empty:
        return {
            "member_id":      member_id,
            "risk_score":     None,
            "risk_level":     None,
            "drivers":        [],
            "waterfall_plot": None,
            "error":          f"Member ID '{member_id}' not found in dataset.",
        }

    raw_row = match.iloc[0]

    # ── Preprocess ────────────────────────────────────────────────────────────
    X_member = preprocess_member_row(raw_row, preprocessor)   # shape (1, 42)

    # ── Predict ───────────────────────────────────────────────────────────────
    churn_prob = float(model.predict_proba(X_member)[0, 1])
    risk_level = assign_risk_level(churn_prob)

    # ── SHAP values for this member ───────────────────────────────────────────
    shap_vals = explainer.shap_values(X_member)[0]   # shape (42,) -- raw log-odds contributions

    # ── Build feature value lookup (pre-engineering raw values) ───────────────
    # We store the raw values for display, falling back to processed for engineered features
    raw_values = raw_row.to_dict()

    # Also compute engineered feature values for display
    cols_to_drop = [c for c in ["PatientID", "Churned"] if c in raw_row.index]
    row_df = pd.DataFrame([raw_row]).drop(columns=cols_to_drop, errors="ignore")
    row_fe = _apply_feature_engineering(row_df).iloc[0].to_dict()

    def _format_value(feat_name, feat_idx):
        """Return a human-readable string for the feature value."""
        # OHE binary features — decode back to the category name
        if "_" in feat_name:
            for cat in NOMINAL_CATS:
                if feat_name.startswith(cat + "_"):
                    category_val = feat_name[len(cat) + 1:]
                    raw_cat = raw_values.get(cat, "?")
                    # The feature value = 1 if this category is the member's, else 0
                    return f"{raw_cat} ({category_val}: {'yes' if raw_cat == category_val else 'no'})"
        # Engineered features
        if feat_name in row_fe:
            v = row_fe[feat_name]
            if isinstance(v, float):
                return f"{v:.3f}"
            return str(v)
        # Raw numeric features
        if feat_name in raw_values:
            v = raw_values[feat_name]
            if isinstance(v, float):
                return f"{v:.2f}"
            return str(v)
        return "n/a"

    # ── Rank drivers by |SHAP value| ─────────────────────────────────────────
    abs_shap   = np.abs(shap_vals)
    top_idx    = np.argsort(abs_shap)[::-1][:top_n]

    drivers = []
    for i in top_idx:
        feat_name = feature_names[i]
        sv        = float(shap_vals[i])
        direction = "increases_risk" if sv > 0 else "decreases_risk"
        label     = FEATURE_LABELS.get(feat_name, feat_name)
        value_str = _format_value(feat_name, i)

        # Evidence-based impact sentence — no assumptions
        impact = (
            f"{label} {DIRECTION_PHRASES[direction]} "
            f"(SHAP contribution: {sv:+.4f})."
        )

        drivers.append({
            "feature":    feat_name,
            "label":      label,
            "value":      value_str,
            "shap_value": round(sv, 4),
            "impact":     impact,
            "direction":  direction,
        })

    # ── Optional: waterfall plot ──────────────────────────────────────────────
    waterfall_path = None
    if save_waterfall:
        waterfall_path = generate_individual_waterfall(
            explainer     = explainer,
            shap_values_row = shap_vals,
            processed_row   = X_member[0],
            feature_names   = feature_names,
            member_id       = member_id,
            churn_prob      = churn_prob,
            risk_level      = risk_level,
        )

    return {
        "member_id":      member_id,
        "risk_score":     round(churn_prob, 4),
        "risk_level":     risk_level,
        "drivers":        drivers,
        "waterfall_plot": waterfall_path,
        "error":          None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 9.  Global SHAP Analysis (run when script is called directly)
# ──────────────────────────────────────────────────────────────────────────────
def run_global_analysis():
    """
    Run the full global SHAP analysis:
      - Compute SHAP values on the test set
      - Save beeswarm summary + global bar chart
      - Print top 15 global feature drivers
    """
    assets = _get_loaded_assets()
    shap_values_test = assets["shap_values_test"]
    X_test           = assets["X_test"]
    feature_names    = assets["feature_names"]

    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    mean_abs_shap, sorted_idx = generate_global_plots(
        shap_values_test, X_test_df, feature_names
    )

    print("\n" + "=" * 65)
    print("  GLOBAL SHAP FEATURE IMPORTANCE (Test Set)")
    print("=" * 65)
    print(f"  {'Rank':<5} {'Feature':<40} {'Mean |SHAP|':>12}")
    print("  " + "-" * 60)
    for rank, i in enumerate(sorted_idx[:15], 1):
        label = FEATURE_LABELS.get(feature_names[i], feature_names[i])
        print(f"  {rank:<5} {label:<40} {mean_abs_shap[i]:>12.5f}")
    print("=" * 65)

    # Summary of how to read the beeswarm plot
    print("""
  HOW TO READ THE BEESWARM SUMMARY PLOT
  --------------------------------------
  - Each dot represents one member in the test set.
  - X-axis: SHAP value (positive = pushes prediction toward churn;
            negative = pushes toward retention).
  - Colour:  Red = high raw feature value; Blue = low raw feature value.
  - Width:   Distribution of impact across the test population.

  Example interpretation (evidence-based):
    - "Overall satisfaction score" dots to the RIGHT with BLUE colour
      -> Low satisfaction scores are associated with increased churn risk.
    - "Tenure_Months" dots to the LEFT with RED colour
      -> Higher tenure (longer membership) is associated with lower predicted
         churn probability.

  NOTE: SHAP values reflect the model's predictions, not clinical ground truth.
""")


# ──────────────────────────────────────────────────────────────────────────────
# 10.  Demo Explanations (run when called directly)
# ──────────────────────────────────────────────────────────────────────────────
def run_demo_explanations():
    """Print formatted SHAP explanations for a small set of sample members."""
    raw_df = load_raw_dataset()

    # Pick 5 members: 1 from each risk zone + 1 random
    # We'll sample from the full dataset by predicted probability
    assets       = _get_loaded_assets()
    model        = assets["model"]
    preprocessor = assets["preprocessor"]

    # Preprocess all members quickly for probability ranking
    all_probs = []
    for _, row in raw_df.iterrows():
        try:
            X = preprocess_member_row(row, preprocessor)
            p = float(model.predict_proba(X)[0, 1])
        except Exception:
            p = -1.0
        all_probs.append(p)

    raw_df = raw_df.copy()
    raw_df["_pred_prob"] = all_probs

    demo_ids = []
    for label, lo, hi in [
        ("Low",      0.00, 0.39),
        ("Medium",   0.40, 0.69),
        ("High",     0.70, 0.84),
        ("Critical", 0.85, 1.00),
    ]:
        subset = raw_df[(raw_df["_pred_prob"] >= lo) & (raw_df["_pred_prob"] < hi)]
        if not subset.empty:
            mid_idx = len(subset) // 2
            pid = subset.iloc[mid_idx]["PatientID"]
            demo_ids.append((pid, label))
            print(f"  Demo member ({label} zone): {pid}  prob={subset.iloc[mid_idx]['_pred_prob']:.3f}")

    print("\n" + "=" * 65)
    print("  INDIVIDUAL MEMBER EXPLANATIONS")
    print("=" * 65)

    all_explanations = []
    for member_id, zone in demo_ids:
        print(f"\n  Processing: {member_id}  (expected zone: {zone})")
        result = get_member_explanation(member_id, top_n=TOP_N_DRIVERS, save_waterfall=True)
        all_explanations.append(result)

        if result["error"]:
            print(f"  ERROR: {result['error']}")
            continue

        print(f"\n  Member ID   : {result['member_id']}")
        print(f"  Risk Score  : {result['risk_score']:.4f}  ({result['risk_score']*100:.1f}%)")
        print(f"  Risk Level  : {result['risk_level']}")
        print(f"  Top {TOP_N_DRIVERS} Feature Drivers:")
        for d in result["drivers"]:
            arrow = "  [+]" if d["direction"] == "increases_risk" else "  [-]"
            print(f"{arrow} {d['label']}")
            print(f"       Value  : {d['value']}")
            print(f"       Impact : {d['impact']}")
        print()

    # Save all explanations to JSON
    out_path = os.path.join(OUTPUTS_DIR, "shap_member_explanations.json")
    with open(out_path, "w") as f:
        # Remove non-serialisable y_prob field
        safe = [{k: v for k, v in e.items() if k != "y_prob"} for e in all_explanations]
        json.dump(safe, f, indent=2)
    print(f"  Saved all demo explanations: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 11.  Main Entry Point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("  SHAP EXPLAINABILITY — MEMBER CHURN PREDICTION")
    print("=" * 65)

    # Step 1: Global analysis + plots
    print("\n[STEP 1] Running global SHAP analysis on test set ...")
    run_global_analysis()

    # Step 2: Demo individual explanations
    print("\n[STEP 2] Generating individual member explanations ...")
    run_demo_explanations()

    print("\n" + "=" * 65)
    print("  SHAP ANALYSIS COMPLETE")
    print(f"  Plots saved to : {PLOTS_DIR}/")
    print(f"  JSON saved to  : {OUTPUTS_DIR}/shap_member_explanations.json")
    print("=" * 65)
