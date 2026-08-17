"""
app.py
======
Flask REST API for Member Churn Prediction and Retention Advisor.
Provides endpoints for member lookup, risk prediction, SHAP drivers,
advisory recommendations, and global analytics.

Endpoints:
----------
  - GET  /health
  - GET  /members
  - GET  /member/<member_id>
  - GET  /analytics
  - POST /predict

Performance:
------------
Loads the saved preprocessor, XGBoost model, and SHAP explainer once at startup.
Pre-computes and caches predictions & SHAP values for all 2,000 members in the dataset
to ensure sub-millisecond response times for lookup and analytics.

Author  : CTS NPN Project
Created : 2026-08-13
"""

import io
import os
import json
import logging
import re
import warnings
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

import sys
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(APP_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Initialize Flask app
app = Flask(__name__, static_folder=None)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
# Configure CORS for local development and deployed frontend URLs
frontend_env = os.environ.get("FRONTEND_URL", "").strip()
allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
if frontend_env:
    if frontend_env == "*":
        allowed_origins = "*"
    else:
        for url in frontend_env.split(","):
            cleaned = url.strip().rstrip("/")
            if cleaned and cleaned not in allowed_origins:
                allowed_origins.append(cleaned)
                if cleaned.startswith("https://"):
                    allowed_origins.append(cleaned.replace("https://", "http://"))
                elif cleaned.startswith("http://"):
                    allowed_origins.append(cleaned.replace("http://", "https://"))

if allowed_origins == "*":
    CORS(app, resources={r"/*": {"origins": "*"}})
else:
    CORS(app, resources={
        r"/*": {
            "origins": allowed_origins,
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })

@app.after_request
def add_cache_control_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    return response


ALLOWED_UPLOAD_EXTENSIONS = {"csv", "xlsx", "xls"}
REQUIRED_FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "State",
    "Tenure_Months",
    "Specialty",
    "Insurance_Type",
    "Visits_Last_Year",
    "Missed_Appointments",
    "Days_Since_Last_Visit",
    "Last_Interaction_Date",
    "Overall_Satisfaction",
    "Wait_Time_Satisfaction",
    "Staff_Satisfaction",
    "Provider_Rating",
    "Avg_Out_Of_Pocket_Cost",
    "Billing_Issues",
    "Portal_Usage",
    "Referrals_Made",
    "Distance_To_Facility_Miles",
]
NUMERIC_FEATURE_COLUMNS = [
    "Age",
    "Tenure_Months",
    "Visits_Last_Year",
    "Missed_Appointments",
    "Days_Since_Last_Visit",
    "Overall_Satisfaction",
    "Wait_Time_Satisfaction",
    "Staff_Satisfaction",
    "Provider_Rating",
    "Avg_Out_Of_Pocket_Cost",
    "Billing_Issues",
    "Portal_Usage",
    "Referrals_Made",
    "Distance_To_Facility_Miles",
]
BINARY_FEATURE_COLUMNS = {"Billing_Issues", "Portal_Usage"}
CATEGORICAL_FEATURE_COLUMNS = ["Gender", "State", "Specialty", "Insurance_Type"]


class DatasetValidationError(ValueError):
    """Raised when an uploaded member dataset cannot be safely analyzed."""

# Serve React SPA (frontend/dist), Flask templates, or static assets
FRONTEND_DIST = os.path.join(APP_DIR, 'frontend', 'dist')
TEMPLATES_DIR = os.path.join(APP_DIR, 'templates')
STATIC_DIR = os.path.join(APP_DIR, 'static')

@app.route('/')
def serve_index():
    if os.path.exists(os.path.join(FRONTEND_DIST, 'index.html')):
        return send_from_directory(FRONTEND_DIST, 'index.html', max_age=0)
    if os.path.exists(os.path.join(TEMPLATES_DIR, 'index.html')):
        return send_from_directory(TEMPLATES_DIR, 'index.html', max_age=0)
    return send_from_directory(APP_DIR, 'index.html', max_age=0)

@app.route('/assets/<path:path>')
def serve_dist_assets(path):
    dist_assets = os.path.join(FRONTEND_DIST, 'assets')
    if os.path.exists(dist_assets):
        return send_from_directory(dist_assets, path, max_age=0)
    return ("Asset not found", 404)

@app.route('/static/<path:filename>')
def serve_static(filename):
    if os.path.exists(STATIC_DIR):
        return send_from_directory(STATIC_DIR, filename, max_age=0)
    return ("Static file not found", 404)

@app.route('/app.js')
@app.route('/main_app.js')
def serve_app_js():
    if os.path.exists(os.path.join(STATIC_DIR, 'app.js')):
        return send_from_directory(STATIC_DIR, 'app.js', max_age=0)
    return send_from_directory(APP_DIR, 'app.js', max_age=0)

@app.route('/styles.css')
@app.route('/style.css')
def serve_styles_css():
    if os.path.exists(os.path.join(STATIC_DIR, 'style.css')):
        return send_from_directory(STATIC_DIR, 'style.css', max_age=0)
    if os.path.exists(os.path.join(STATIC_DIR, 'styles.css')):
        return send_from_directory(STATIC_DIR, 'styles.css', max_age=0)
    return send_from_directory(APP_DIR, 'styles.css', max_age=0)

@app.route('/model_data.js')
def serve_model_data_js():
    if os.path.exists(os.path.join(STATIC_DIR, 'model_data.js')):
        return send_from_directory(STATIC_DIR, 'model_data.js', max_age=0)
    return send_from_directory(APP_DIR, 'model_data.js', max_age=0)

# ------------------------------------------------------------------------------
# 1.  Asset Loading & Pre-caching
# ------------------------------------------------------------------------------
logger.info("Initializing API and loading models/explainers...")

try:
    # Import functions and assets from shap_explainer and retention_advisor
    from shap_explainer import (
        _get_loaded_assets,
        _apply_feature_engineering,
        preprocess_member_row,
        assign_risk_level,
        FEATURE_LABELS
    )
    from retention_advisor import get_retention_recommendations

    # Get cached assets (loads model, preprocessor, explainer, raw_df)
    assets = _get_loaded_assets()
    model = assets["model"]
    preprocessor = assets["preprocessor"]
    explainer = assets["explainer"]
    raw_df = assets["raw_df"]
    feature_names = assets["feature_names"]
    
    logger.info("Machine learning models, preprocessing pipelines, and SHAP TreeExplainer loaded successfully.")

    # --------------------------------------------------------------------------
    # Pre-compute & Cache All Predictions and SHAP Values for Dataset
    # --------------------------------------------------------------------------
    logger.info("Pre-computing predictions and SHAP values for all 2,000 members...")
    
    # Preprocess the entire raw dataset in one batch
    cols_to_drop = [c for c in ["PatientID", "Churned"] if c in raw_df.columns]
    raw_df_fe = raw_df.drop(columns=cols_to_drop)
    df_fe = _apply_feature_engineering(raw_df_fe)
    X_all_processed = preprocessor.transform(df_fe)
    
    # Batch predict probabilities
    all_probs = model.predict_proba(X_all_processed)[:, 1]
    
    # Batch compute SHAP values (extremely fast in TreeExplainer tree_path_dependent mode)
    all_shap_values = explainer.shap_values(X_all_processed)
    
    # Index the pre-computed data by PatientID
    MEMBER_CACHE = {}
    RISK_COUNTS = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0}
    
    for i, row in raw_df.iterrows():
        member_id = row["PatientID"]
        prob = float(all_probs[i])
        risk_lvl = assign_risk_level(prob)
        RISK_COUNTS[risk_lvl] += 1
        
        # Extract SHAP drivers for this member
        shap_row = all_shap_values[i]
        abs_shap = np.abs(shap_row)
        top_indices = np.argsort(abs_shap)[::-1]
        
        # Primary driver is the feature with largest absolute SHAP value
        primary_driver_idx = top_indices[0]
        primary_driver_feat = feature_names[primary_driver_idx]
        primary_driver_label = FEATURE_LABELS.get(primary_driver_feat, primary_driver_feat)
        primary_driver_dir = "increases_risk" if shap_row[primary_driver_idx] > 0 else "decreases_risk"
        primary_driver_text = f"{primary_driver_label} ({'increases' if primary_driver_dir == 'increases_risk' else 'decreases'} risk)"
        
        # Store top drivers details
        drivers = []
        for idx in top_indices[:5]:  # Top 5 drivers
            feat = feature_names[idx]
            sv = float(shap_row[idx])
            direction = "increases_risk" if sv > 0 else "decreases_risk"
            label = FEATURE_LABELS.get(feat, feat)
            
            # Format actual value for display
            raw_val = row.get(feat, None)
            if raw_val is None:
                # Might be an engineered feature
                raw_val = df_fe.iloc[i].get(feat, "N/A")
            
            if isinstance(raw_val, float):
                val_str = f"{raw_val:.2f}"
            else:
                val_str = str(raw_val)
                
            drivers.append({
                "feature": feat,
                "label": label,
                "value": val_str,
                "shap_value": round(sv, 4),
                "direction": direction,
                "impact": f"{label} {'is associated with increased predicted churn risk' if direction == 'increases_risk' else 'is associated with decreased predicted churn risk'} (SHAP contribution: {sv:+.4f})."
            })
            
        # Get retention recommendations (rule-based)
        raw_member_dict = row.to_dict()
        raw_member_dict["risk_score"] = prob
        raw_member_dict["risk_level"] = risk_lvl
        
        recommendations = get_retention_recommendations(
            member_data=raw_member_dict,
            shap_drivers=drivers,
            risk_level=risk_lvl
        )
        
        # Clean date values for JSON serialization
        member_info = {}
        for k, v in row.to_dict().items():
            if isinstance(v, pd.Timestamp):
                member_info[k] = v.isoformat()
            elif isinstance(v, (np.int64, np.int32)):
                member_info[k] = int(v)
            elif isinstance(v, (np.float64, np.float32)):
                member_info[k] = float(v)
            else:
                member_info[k] = v

        MEMBER_CACHE[member_id] = {
            "member_info": member_info,
            "churn_probability": round(prob, 4),
            "risk_level": risk_lvl,
            "primary_driver": primary_driver_text,
            "drivers": drivers,
            "recommendations": recommendations
        }
        
    logger.info(f"Pre-computation complete. Cached {len(MEMBER_CACHE)} members.")
    
    # Calculate global analytics
    avg_risk = float(all_probs.mean())
    
    # Calculate top global risk drivers
    mean_abs_shap = np.abs(all_shap_values).mean(axis=0)
    top_global_indices = np.argsort(mean_abs_shap)[::-1][:5]
    GLOBAL_TOP_DRIVERS = [
        {
            "feature": feature_names[idx],
            "label": FEATURE_LABELS.get(feature_names[idx], feature_names[idx]),
            "mean_importance": round(float(mean_abs_shap[idx]), 5)
        }
        for idx in top_global_indices
    ]
    
    # Calculate risk trend by tenure months cohort (e.g. 0-12m, 13-24m, etc.)
    tenure_vals = raw_df["Tenure_Months"].values
    cohort_bins = [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 121]
    cohort_labels = ["0-12m", "13-24m", "25-36m", "37-48m", "49-60m", "61-72m", "73-84m", "85-96m", "97-108m", "109-120m"]
    
    TENURE_RISK_TREND = []
    for idx in range(len(cohort_bins) - 1):
        low, high = cohort_bins[idx], cohort_bins[idx+1]
        mask = (tenure_vals >= low) & (tenure_vals < high)
        if mask.any():
            mean_p = float(all_probs[mask].mean())
        else:
            mean_p = 0.0
        TENURE_RISK_TREND.append({
            "cohort": cohort_labels[idx],
            "avg_probability": round(mean_p, 4)
        })

    
except Exception as e:
    logger.error(f"Critical error during API startup: {str(e)}", exc_info=True)
    raise e


# ------------------------------------------------------------------------------
# 2.  API Endpoints
# ------------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


@app.route("/members", methods=["GET"])
@app.route("/api/members", methods=["GET"])
def get_members():
    """
    Returns a list of all members with summary risk details, plan, tenure and actions.
    Uses cached pre-computed values for instant delivery.
    """
    members_list = []
    for member_id, cached in MEMBER_CACHE.items():
        top_rec = cached["recommendations"][0]["action"] if cached["recommendations"] else "Monitor (Routine)"
        members_list.append({
            "member_id": member_id,
            "plan": cached["member_info"]["Insurance_Type"],
            "tenure": cached["member_info"]["Tenure_Months"],
            "churn_probability": cached["churn_probability"],
            "risk_level": cached["risk_level"],
            "primary_driver": cached["primary_driver"],
            "recommended_action": top_rec
        })
    return jsonify(members_list)


@app.route("/member/<member_id>", methods=["GET"])
@app.route("/api/member/<member_id>", methods=["GET"])
def get_member_detail(member_id):
    """
    Returns detailed risk profile and rule-based recommendations for a single member.
    Uses cached pre-computed values.
    """
    cached = MEMBER_CACHE.get(member_id)
    if not cached:
        return jsonify({"error": f"Member with ID '{member_id}' not found."}), 404
    return jsonify(cached)


@app.route("/analytics", methods=["GET"])
@app.route("/api/analytics", methods=["GET"])
def get_analytics():
    """
    Returns global dataset statistics and risk distribution analytics.
    """
    analytics_data = {
        "total_members": len(MEMBER_CACHE),
        "risk_distribution": RISK_COUNTS,
        "average_risk": round(avg_risk, 4),
        "top_risk_drivers": GLOBAL_TOP_DRIVERS,
        "tenure_risk_trend": TENURE_RISK_TREND
    }
    return jsonify(analytics_data)


@app.route("/predict", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Accepts member feature data via POST and returns predictions, SHAP drivers,
    and retention recommendations. Runs dynamically using the saved pipeline.
    
    Input JSON structure example:
    {
        "Age": 45,
        "Gender": "Male",
        "State": "MI",
        "Tenure_Months": 60,
        "Specialty": "Family Medicine",
        "Insurance_Type": "Private",
        "Visits_Last_Year": 6,
        "Missed_Appointments": 1,
        "Days_Since_Last_Visit": 120,
        "Last_Interaction_Date": "2025-06-03",
        "Overall_Satisfaction": 3.5,
        "Wait_Time_Satisfaction": 3.5,
        "Staff_Satisfaction": 3.5,
        "Provider_Rating": 3.5,
        "Avg_Out_Of_Pocket_Cost": 450,
        "Billing_Issues": 0,
        "Portal_Usage": 1,
        "Referrals_Made": 1,
        "Distance_To_Facility_Miles": 12.5
    }
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "Missing input JSON payload"}), 400
            
        # Handle default fallbacks for missing columns to ensure API robustness
        DEFAULTS = {
            "Age": 54, "Gender": "Male", "State": "MI", "Tenure_Months": 60,
            "Specialty": "Family Medicine", "Insurance_Type": "Private",
            "Visits_Last_Year": 8, "Missed_Appointments": 2,
            "Days_Since_Last_Visit": 363, "Last_Interaction_Date": "2025-06-03",
            "Overall_Satisfaction": 3.2, "Wait_Time_Satisfaction": 3.3,
            "Staff_Satisfaction": 3.5, "Provider_Rating": 3.8,
            "Avg_Out_Of_Pocket_Cost": 716, "Billing_Issues": 0,
            "Portal_Usage": 0, "Referrals_Made": 1, "Distance_To_Facility_Miles": 25.0
        }
        
        # Merge input data with defaults
        inputs = {}
        for k, default_val in DEFAULTS.items():
            inputs[k] = data.get(k, default_val)
            
        # Represent as a Pandas Series (excluding PatientID/Churned)
        row_series = pd.Series(inputs)
        
        # Preprocess using pipeline
        X_member = preprocess_member_row(row_series, preprocessor)
        
        # Predict probability
        prob = float(model.predict_proba(X_member)[0, 1])
        risk_lvl = assign_risk_level(prob)
        
        # Compute SHAP values for this custom input
        shap_vals = explainer.shap_values(X_member)[0]
        abs_shap = np.abs(shap_vals)
        top_indices = np.argsort(abs_shap)[::-1]
        
        # Extract engineered features for OHE feature checks
        cols_to_drop = [c for c in ["PatientID", "Churned"] if c in row_series.index]
        row_df = pd.DataFrame([row_series]).drop(columns=cols_to_drop, errors="ignore")
        row_fe = _apply_feature_engineering(row_df).iloc[0].to_dict()
        
        # Format top 5 drivers
        drivers = []
        for idx in top_indices[:5]:
            feat = feature_names[idx]
            sv = float(shap_vals[idx])
            direction = "increases_risk" if sv > 0 else "decreases_risk"
            label = FEATURE_LABELS.get(feat, feat)
            
            # Format actual value for display
            if "_" in feat:
                # OHE decoder
                val_str = str(inputs.get(feat, "N/A"))
            elif feat in row_fe:
                v = row_fe[feat]
                val_str = f"{v:.3f}" if isinstance(v, float) else str(v)
            else:
                v = inputs.get(feat, "N/A")
                val_str = f"{v:.2f}" if isinstance(v, float) else str(v)
                
            drivers.append({
                "feature": feat,
                "label": label,
                "value": val_str,
                "shap_value": round(sv, 4),
                "direction": direction,
                "impact": f"{label} {'is associated with increased predicted churn risk' if direction == 'increases_risk' else 'is associated with decreased predicted churn risk'} (SHAP contribution: {sv:+.4f})."
            })
            
        # Get recommendations
        inputs["risk_score"] = prob
        inputs["risk_level"] = risk_lvl
        recommendations = get_retention_recommendations(
            member_data=inputs,
            shap_drivers=drivers,
            risk_level=risk_lvl
        )
        
        return jsonify({
            "churn_probability": round(prob, 4),
            "risk_level": risk_lvl,
            "drivers": drivers,
            "recommendations": recommendations
        })
        
    except Exception as e:
        logger.error(f"Error during dynamic prediction: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal prediction failure: {str(e)}"}), 500


def _json_error(message, status_code=400, **extra):
    payload = {"success": False, "status": "error", "error": message}
    payload.update(extra)
    return jsonify(payload), status_code


def _column_key(column_name):
    return re.sub(r"[^a-z0-9]", "", str(column_name).strip().lower())


COLUMN_ALIASES = {
    "patientid": "PatientID",
    "memberid": "PatientID",
    "subscriberid": "PatientID",
    "id": "PatientID",
    "age": "Age",
    "gender": "Gender",
    "state": "State",
    "tenuremonths": "Tenure_Months",
    "tenure": "Tenure_Months",
    "membertenure": "Tenure_Months",
    "specialty": "Specialty",
    "providerSpecialty".lower(): "Specialty",
    "insurancetype": "Insurance_Type",
    "insurance": "Insurance_Type",
    "plan": "Insurance_Type",
    "plantype": "Insurance_Type",
    "visitslastyear": "Visits_Last_Year",
    "visits": "Visits_Last_Year",
    "annualvisits": "Visits_Last_Year",
    "missedappointments": "Missed_Appointments",
    "missed": "Missed_Appointments",
    "dayssincelastvisit": "Days_Since_Last_Visit",
    "dayssincevisit": "Days_Since_Last_Visit",
    "lastinteractiondate": "Last_Interaction_Date",
    "lastvisitdate": "Last_Interaction_Date",
    "overallsatisfaction": "Overall_Satisfaction",
    "satisfaction": "Overall_Satisfaction",
    "waittimesatisfaction": "Wait_Time_Satisfaction",
    "staffsatisfaction": "Staff_Satisfaction",
    "providerrating": "Provider_Rating",
    "avgoutofpocketcost": "Avg_Out_Of_Pocket_Cost",
    "averageoutofpocketcost": "Avg_Out_Of_Pocket_Cost",
    "outofpocketcost": "Avg_Out_Of_Pocket_Cost",
    "outofpocket": "Avg_Out_Of_Pocket_Cost",
    "billingissues": "Billing_Issues",
    "billing": "Billing_Issues",
    "portalusage": "Portal_Usage",
    "portal": "Portal_Usage",
    "referralsmade": "Referrals_Made",
    "referrals": "Referrals_Made",
    "distancetofacilitymiles": "Distance_To_Facility_Miles",
    "distance": "Distance_To_Facility_Miles",
    "churned": "Churned",
}


def _first_problem_rows(mask, max_rows=5):
    return [int(i) + 2 for i in mask[mask].index[:max_rows]]


def _normalize_columns(df):
    rename_map = {}
    for col in df.columns:
        canonical = COLUMN_ALIASES.get(_column_key(col))
        if canonical:
            rename_map[col] = canonical
    df = df.rename(columns=rename_map)

    duplicated = df.columns[df.columns.duplicated()].tolist()
    if duplicated:
        dupes = sorted(set(duplicated))
        raise DatasetValidationError(
            "Duplicate columns after normalization: " + ", ".join(dupes)
        )
    return df


def _validate_no_missing(df, column):
    values = df[column]
    blank_mask = values.isna() | values.astype(str).str.strip().eq("")
    if blank_mask.any():
        rows = _first_problem_rows(blank_mask)
        raise DatasetValidationError(
            f"Column '{column}' has missing or blank value(s) at row(s): {rows}."
        )


def _coerce_binary_values(series):
    yes_no_map = {
        "yes": 1, "y": 1, "true": 1, "t": 1, "present": 1, "active": 1,
        "no": 0, "n": 0, "false": 0, "f": 0, "none": 0, "absent": 0, "inactive": 0,
    }
    text = series.astype(str).str.strip()
    mapped = text.str.lower().map(yes_no_map)
    return series.where(mapped.isna(), mapped)


def _coerce_numeric_column(df, column):
    _validate_no_missing(df, column)

    values = df[column]
    if column in BINARY_FEATURE_COLUMNS:
        values = _coerce_binary_values(values)

    if values.dtype == object:
        values = (
            values.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.strip()
        )

    converted = pd.to_numeric(values, errors="coerce")
    bad_mask = converted.isna()
    if bad_mask.any():
        examples = df.loc[bad_mask, column].astype(str).head(3).tolist()
        rows = _first_problem_rows(bad_mask)
        raise DatasetValidationError(
            f"Column '{column}' contains non-numeric value(s) at row(s) {rows}: {examples}."
        )

    if column in BINARY_FEATURE_COLUMNS:
        invalid_binary = ~converted.isin([0, 1])
        if invalid_binary.any():
            rows = _first_problem_rows(invalid_binary)
            raise DatasetValidationError(
                f"Column '{column}' must contain only 0/1 or yes/no values. Problem row(s): {rows}."
            )
        converted = converted.astype(int)

    df[column] = converted


def _coerce_date_column(df, column):
    _validate_no_missing(df, column)
    parsed = pd.to_datetime(df[column], errors="coerce")
    bad_mask = parsed.isna()
    if bad_mask.any():
        examples = df.loc[bad_mask, column].astype(str).head(3).tolist()
        rows = _first_problem_rows(bad_mask)
        raise DatasetValidationError(
            f"Column '{column}' contains invalid date value(s) at row(s) {rows}: {examples}."
        )
    df[column] = parsed.dt.strftime("%Y-%m-%d")


def _normalize_member_ids(df):
    if "PatientID" not in df.columns:
        df["PatientID"] = [f"M{i + 1:05d}" for i in range(len(df))]
        return

    _validate_no_missing(df, "PatientID")
    df["PatientID"] = df["PatientID"].astype(str).str.strip()
    seen = {}
    unique_ids = []
    for member_id in df["PatientID"]:
        if member_id not in seen:
            seen[member_id] = 1
            unique_ids.append(member_id)
        else:
            seen[member_id] += 1
            unique_ids.append(f"{member_id}-{seen[member_id]}")
    df["PatientID"] = unique_ids


def _normalize_and_clean_dataframe(df_raw):
    """
    Normalize aliases and validate required model input columns without leaking
    target data or inventing missing feature values.
    """
    if df_raw is None or df_raw.empty:
        raise DatasetValidationError("Uploaded dataset is empty.")

    df = _normalize_columns(df_raw.copy())

    missing = [col for col in REQUIRED_FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise DatasetValidationError(
            "Missing required column(s): " + ", ".join(missing)
        )

    _normalize_member_ids(df)

    for col in CATEGORICAL_FEATURE_COLUMNS:
        _validate_no_missing(df, col)
        df[col] = df[col].astype(str).str.strip()

    for col in NUMERIC_FEATURE_COLUMNS:
        _coerce_numeric_column(df, col)

    _coerce_date_column(df, "Last_Interaction_Date")

    if "Churned" in df.columns:
        churned = pd.to_numeric(df["Churned"], errors="coerce")
        invalid_target = churned.notna() & ~churned.isin([0, 1])
        if invalid_target.any():
            rows = _first_problem_rows(invalid_target)
            raise DatasetValidationError(
                f"Optional target column 'Churned' must contain only 0/1 values. Problem row(s): {rows}."
            )
        df["Churned"] = churned

    return df


def _process_dataframe_response(df_uploaded):
    """
    Shared processing pipeline used by both /predict-file and /load-demo.
    Validates, preprocesses, predicts, generates SHAP + recommendations,
    updates global caches, and returns a structured Flask JSON response.
    """
    global MEMBER_CACHE, RISK_COUNTS, avg_risk, GLOBAL_TOP_DRIVERS, TENURE_RISK_TREND

    # Clean and normalize uploaded dataset
    df_clean = _normalize_and_clean_dataframe(df_uploaded)

    # Preprocessing & Prediction
    cols_to_drop = [c for c in ["PatientID", "Churned"] if c in df_clean.columns]
    df_features = df_clean.drop(columns=cols_to_drop)
    df_fe = _apply_feature_engineering(df_features)
    X_processed = preprocessor.transform(df_fe)
    probs = model.predict_proba(X_processed)[:, 1]
    shap_vals = explainer.shap_values(X_processed)

    # Build Cache
    new_member_cache = {}
    new_risk_counts = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0}

    for position, (_, row) in enumerate(df_clean.iterrows()):
        member_id = str(row["PatientID"])
        prob = float(probs[position])
        risk_lvl = assign_risk_level(prob)
        new_risk_counts[risk_lvl] += 1

        shap_row = shap_vals[position]
        abs_shap = np.abs(shap_row)
        top_indices = np.argsort(abs_shap)[::-1]

        drivers = []
        for idx in top_indices[:5]:
            feat = feature_names[idx]
            sv = float(shap_row[idx])
            direction = "increases_risk" if sv > 0 else "decreases_risk"
            label = FEATURE_LABELS.get(feat, feat)
            raw_val = row.get(feat, None)
            if raw_val is None:
                raw_val = df_fe.iloc[position].get(feat, "N/A")
            val_str = f"{raw_val:.2f}" if isinstance(raw_val, float) else str(raw_val)
            drivers.append({
                "feature": feat, "label": label, "value": val_str,
                "shap_value": round(sv, 4), "direction": direction,
                "impact": f"{label} {'is associated with increased predicted churn risk' if direction == 'increases_risk' else 'is associated with decreased predicted churn risk'} (SHAP contribution: {sv:+.4f})."
            })

        raw_member_dict = row.to_dict()
        raw_member_dict["risk_score"] = prob
        raw_member_dict["risk_level"] = risk_lvl
        recommendations = get_retention_recommendations(
            member_data=raw_member_dict, shap_drivers=drivers, risk_level=risk_lvl
        )

        member_info = {}
        for k, v in row.to_dict().items():
            if isinstance(v, pd.Timestamp):
                member_info[k] = v.isoformat()
            elif isinstance(v, (np.int64, np.int32)):
                member_info[k] = int(v)
            elif isinstance(v, (np.float64, np.float32)):
                member_info[k] = float(v)
            else:
                member_info[k] = v

        new_member_cache[member_id] = {
            "member_info": member_info,
            "churn_probability": round(prob, 4),
            "risk_level": risk_lvl,
            "primary_driver": f"{drivers[0]['label']} ({'increases' if drivers[0]['direction'] == 'increases_risk' else 'decreases'} risk)" if drivers else "None",
            "drivers": drivers,
            "recommendations": recommendations
        }

    new_avg_risk = float(probs.mean())
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    top_global_indices = np.argsort(mean_abs_shap)[::-1][:5]
    new_global_top_drivers = [
        {"feature": feature_names[idx], "label": FEATURE_LABELS.get(feature_names[idx], feature_names[idx]),
         "mean_importance": round(float(mean_abs_shap[idx]), 5)}
        for idx in top_global_indices
    ]

    tenure_vals = df_clean["Tenure_Months"].values
    cohort_bins = [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 121]
    cohort_labels = ["0-12m","13-24m","25-36m","37-48m","49-60m","61-72m","73-84m","85-96m","97-108m","109-120m"]
    new_tenure_risk_trend = []
    for idx in range(len(cohort_bins) - 1):
        low, high = cohort_bins[idx], cohort_bins[idx+1]
        mask = (tenure_vals >= low) & (tenure_vals < high)
        mean_p = float(probs[mask].mean()) if mask.any() else 0.0
        new_tenure_risk_trend.append({"cohort": cohort_labels[idx], "avg_probability": round(mean_p, 4)})

    MEMBER_CACHE = new_member_cache
    RISK_COUNTS = new_risk_counts
    avg_risk = new_avg_risk
    GLOBAL_TOP_DRIVERS = new_global_top_drivers
    TENURE_RISK_TREND = new_tenure_risk_trend

    members_list = []
    for member_id, cached in MEMBER_CACHE.items():
        top_rec = cached["recommendations"][0]["action"] if cached["recommendations"] else "Monitor (Routine)"
        members_list.append({
            "member_id": member_id,
            "plan": cached["member_info"]["Insurance_Type"],
            "tenure": cached["member_info"]["Tenure_Months"],
            "churn_probability": cached["churn_probability"],
            "risk_level": cached["risk_level"],
            "primary_driver": cached["primary_driver"],
            "recommended_action": top_rec,
            "drivers": cached["drivers"],
            "recommendations": cached["recommendations"]
        })

    risk_summary = {key.lower(): int(value) for key, value in RISK_COUNTS.items()}
    logger.info(f"Pipeline complete. Cached {len(MEMBER_CACHE)} members.")
    return jsonify({
        "success": True,
        "status": "success",
        "total_members": len(MEMBER_CACHE),
        "risk_summary": risk_summary,
        "summary": {
            "total_members": len(MEMBER_CACHE),
            "risk_distribution": RISK_COUNTS,
            "average_risk": round(avg_risk, 4),
            "top_risk_drivers": GLOBAL_TOP_DRIVERS,
            "tenure_risk_trend": TENURE_RISK_TREND
        },
        "members": members_list
    })


def _get_upload_extension(filename):
    safe_name = os.path.basename(filename or "").strip()
    if "." not in safe_name:
        raise DatasetValidationError(
            "Unsupported file type. Please upload a .csv, .xlsx, or .xls file."
        )
    extension = safe_name.rsplit(".", 1)[1].lower()
    if extension not in ALLOWED_UPLOAD_EXTENSIONS:
        raise DatasetValidationError(
            f"Unsupported file type '.{extension}'. Please upload a .csv, .xlsx, or .xls file."
        )
    return extension


def _read_csv_bytes(file_bytes):
    parse_errors = []
    for encoding in ("utf-8", "utf-8-sig", "latin1", "cp1252"):
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)
        except UnicodeDecodeError as exc:
            parse_errors.append(f"{encoding}: {exc}")
        except pd.errors.EmptyDataError:
            raise DatasetValidationError("Uploaded dataset is empty.")
        except Exception as exc:
            parse_errors.append(f"{encoding}: {exc}")

    raise DatasetValidationError(
        "Could not parse CSV file. Please verify it is a valid delimited table."
    )


def _read_excel_bytes(file_bytes, extension):
    parse_errors = []
    engines = ["openpyxl"] if extension == "xlsx" else ["xlrd", "openpyxl"]
    for engine in engines:
        try:
            return pd.read_excel(io.BytesIO(file_bytes), engine=engine)
        except ImportError as exc:
            parse_errors.append(f"{engine}: {exc}")
        except ValueError as exc:
            parse_errors.append(f"{engine}: {exc}")
        except Exception as exc:
            parse_errors.append(f"{engine}: {exc}")

    # Some legacy project files are CSV content with a .xls suffix.
    if extension == "xls":
        try:
            logger.info("Excel parser failed for .xls upload; trying CSV fallback for legacy .csv.xls files.")
            return _read_csv_bytes(file_bytes)
        except DatasetValidationError as exc:
            parse_errors.append(f"csv fallback: {exc}")

    details = "; ".join(parse_errors[-3:])
    if extension == "xls" and any("xlrd" in err.lower() for err in parse_errors):
        raise DatasetValidationError(
            "Could not parse .xls file. Install xlrd>=2.0.1 for legacy Excel .xls support. "
            f"Parser details: {details}"
        )
    raise DatasetValidationError(
        f"Could not parse .{extension} file. Parser details: {details}"
    )


def _read_uploaded_dataframe(file_storage):
    filename = file_storage.filename or ""
    extension = _get_upload_extension(filename)
    file_bytes = file_storage.read()
    if not file_bytes:
        raise DatasetValidationError("Uploaded file is empty.")

    if extension == "csv":
        return _read_csv_bytes(file_bytes)
    return _read_excel_bytes(file_bytes, extension)


@app.route("/predict-file", methods=["POST"])
@app.route("/api/predict-file", methods=["POST"])
@app.route("/upload", methods=["POST"])
@app.route("/api/upload", methods=["POST"])
def predict_file():
    """
    Accepts a CSV or Excel file of member profiles, validates it, preprocesses it,
    predicts churn probabilities, runs SHAP explanations, runs retention rules,
    updates the in-memory global caches, and returns structured JSON.
    """
    try:
        if "file" not in request.files:
            return _json_error("No file was uploaded. Send multipart/form-data with field name 'file'.")

        file = request.files["file"]
        if file.filename == "":
            return _json_error("No file was selected.")

        df_uploaded = _read_uploaded_dataframe(file)
        return _process_dataframe_response(df_uploaded)
    except DatasetValidationError as exc:
        logger.warning(f"Dataset validation failed: {exc}")
        return _json_error(str(exc))
    except Exception as e:
        logger.error(f"Error processing uploaded file: {str(e)}", exc_info=True)
        return _json_error(f"Dataset analysis failed: {str(e)}", status_code=500)



@app.route("/retention-recommendations", methods=["POST"])
@app.route("/api/retention-recommendations", methods=["POST"])
def retention_recommendations():
    """
    Accepts member data, SHAP drivers, and risk level as JSON,
    and returns rule-based recommendations.
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "Missing input JSON payload"}), 400
        
        member_data = data.get("member_data", data)
        shap_drivers = data.get("shap_drivers", [])
        risk_level = data.get("risk_level", None)
        
        recs = get_retention_recommendations(
            member_data=member_data,
            shap_drivers=shap_drivers,
            risk_level=risk_level
        )
        return jsonify(recs)
    except Exception as e:
        logger.error(f"Error in dynamic recommendations endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": f"Failed to compute recommendations: {str(e)}"}), 500


@app.route("/load-demo", methods=["GET"])
@app.route("/api/load-demo", methods=["GET"])
def load_demo():
    """
    Reads the bundled patient_churn_dataset.csv.xls from disk,
    runs it through the full processing pipeline (identical to /predict-file),
    and returns structured analytics + member list.
    Used by the 'Load Demo Data' button so no file dialog is needed.
    """
    import io
    demo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'dataset.csv')
    if not os.path.exists(demo_path):
        demo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'patient_churn_dataset.csv.xls')
    if not os.path.exists(demo_path):
        return jsonify({"error": "Demo dataset not found on server."}), 404
    try:
        with open(demo_path, 'rb') as f:
            file_bytes = f.read()
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as e:
        return jsonify({"error": f"Failed to read demo dataset: {str(e)}"}), 500
    logger.info("Loading demo dataset via /load-demo endpoint...")
    return _process_dataframe_response(df)



# ------------------------------------------------------------------------------
# 3.  Main Entrypoint
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Standard development server port 5000
    # Host 0.0.0.0 enables access from virtual environments / containers
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting CareShield Advisor REST API on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
