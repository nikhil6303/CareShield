"""
retention_advisor.py
====================
Member Churn Prediction and Retention Advisor
Healthcare / Health-Plan Domain

What this module does
---------------------
Implements a fully RULE-BASED retention recommendation engine.

- No machine learning model is trained here.
- No LLM generates recommendations.
- Recommendations are derived exclusively from observable dataset features
  and SHAP-identified risk drivers.
- All rules are declared in RULE_CATALOG (plain dicts) -- easy to modify.

Language policy (MANDATORY)
----------------------------
Every recommendation must be evidence-based. The system:
  NEVER claims the member definitely intends to leave.
  NEVER diagnoses the member medically.
  NEVER infers emotions (unhappy, frustrated, etc.).
  NEVER makes unsupported financial or medical assumptions.

This system is a decision-support tool.
Human review is required before any action is taken.

Action categories
-----------------
  1. Service Recovery
  2. Benefit Education
  3. Care Outreach
  4. Pharmacy Support
  5. General Member Engagement

Usage
-----
    from retention_advisor import get_retention_recommendations

    recommendations = get_retention_recommendations(member_data, shap_drivers)

    # OR: run end-to-end from a member_id
    from retention_advisor import advise_member
    result = advise_member("C20983")

Input contracts
---------------
member_data : dict
    Raw field values from the dataset row. Must contain at minimum:
      PatientID, Age, Tenure_Months, Visits_Last_Year, Missed_Appointments,
      Days_Since_Last_Visit, Overall_Satisfaction, Wait_Time_Satisfaction,
      Staff_Satisfaction, Provider_Rating, Avg_Out_Of_Pocket_Cost,
      Billing_Issues, Portal_Usage, Referrals_Made, Distance_To_Facility_Miles,
      Insurance_Type, Specialty.
    Optional: risk_score (float 0-1), risk_level (str).

shap_drivers : list[dict]
    Each dict contains: feature, label, value, shap_value, direction.
    (Produced by shap_explainer.get_member_explanation.)

Returns
-------
list[dict] -- each item:
    action          : str   (one of the 5 action categories)
    priority        : str   (Critical / High / Medium / Low)
    reason          : str   (evidence-based sentence)
    evidence        : list[str]  (observable signals that triggered the rule)
    suggested_next_step : str

Author  : CTS NPN Project
Created : 2026-08-13
"""

# ==============================================================================
# 0.  Imports
# ==============================================================================
import copy
import json
import os
import warnings
from typing import Any

warnings.filterwarnings("ignore")


# ==============================================================================
# 1.  RULE CATALOG
#     -------------------------------------------------------------------------
#     HOW TO MODIFY RULES
#     -------------------
#     Each rule is a dict with these keys:
#
#       rule_id    : unique string -- used for deduplication
#       category   : one of the 5 action categories (string)
#       action     : short action name shown in output
#       condition  : callable(member_data, shap_drivers) -> bool
#       priority_fn: callable(member_data, risk_level) -> str
#       reason     : static string (evidence-based, no assumptions)
#       evidence_fn: callable(member_data, shap_drivers) -> list[str]
#       next_step  : static string describing the suggested next step
#
#     To add a new rule: append a new dict to RULE_CATALOG.
#     To disable a rule: set "enabled": False in its dict.
#     To change a threshold: update only the condition lambda.
#
#     THRESHOLDS are based on dataset quartiles:
#       Overall_Satisfaction  : scale 1.5-5.0, p25=2.4, median=3.2
#       Wait_Time_Satisfaction: scale 1.5-5.0, p25=2.4, median=3.3
#       Staff_Satisfaction    : scale 2.0-5.0, p25=2.775, median=3.5
#       Provider_Rating       : scale 2.5-5.0, p25=3.1,   median=3.8
#       Days_Since_Last_Visit : p25=180,  median=363, p75=550
#       Missed_Appointments   : median=2, p75=3, max=5
#       Distance_To_Facility  : p25=12.8, median=25, p75=37.2
#       Avg_Out_Of_Pocket_Cost: p25=326,  median=716, p75=1368
#       Visits_Last_Year      : p25=4,    median=8
#       Tenure_Months         : median=60
# ==============================================================================

def _shap_feature_increases_risk(shap_drivers: list, feature_name: str) -> bool:
    """Return True if the named feature appears in shap_drivers as increases_risk."""
    for d in shap_drivers:
        if d.get("feature") == feature_name and d.get("direction") == "increases_risk":
            return True
    return False


def _shap_any_feature_increases(shap_drivers: list, feature_names: list) -> bool:
    """Return True if ANY feature in feature_names increases risk in shap_drivers."""
    return any(_shap_feature_increases_risk(shap_drivers, f) for f in feature_names)


def _priority_by_risk(member_data: dict, risk_level: str) -> str:
    """Map the model risk level to a recommendation priority."""
    return {
        "Critical": "Critical",
        "High":     "High",
        "Medium":   "Medium",
        "Low":      "Low",
    }.get(risk_level, "Medium")


def _elevated_priority(member_data: dict, risk_level: str) -> str:
    """
    Escalate priority by one tier when the underlying signal is severe.
    Critical stays Critical; otherwise bumps up one level.
    """
    mapping = {"Low": "Medium", "Medium": "High", "High": "Critical", "Critical": "Critical"}
    return mapping.get(risk_level, "High")


# -- Helper: satisfaction low check --------------------------------------------
def _is_overall_sat_low(m):        return float(m.get("Overall_Satisfaction", 5)) < 2.5
def _is_wait_sat_low(m):           return float(m.get("Wait_Time_Satisfaction", 5)) < 2.5
def _is_staff_sat_low(m):          return float(m.get("Staff_Satisfaction", 5)) < 2.8
def _is_provider_rating_low(m):    return float(m.get("Provider_Rating", 5)) < 3.1
def _has_billing_issue(m):         return int(m.get("Billing_Issues", 0)) == 1
def _no_portal_usage(m):           return int(m.get("Portal_Usage", 0)) == 0
def _visits_low(m):                return int(m.get("Visits_Last_Year", 99)) <= 3
def _missed_appts_high(m):         return int(m.get("Missed_Appointments", 0)) >= 3
def _days_since_visit_high(m):     return int(m.get("Days_Since_Last_Visit", 0)) >= 365
def _days_since_visit_moderate(m): return int(m.get("Days_Since_Last_Visit", 0)) >= 180
def _distance_high(m):             return float(m.get("Distance_To_Facility_Miles", 0)) > 40
def _distance_moderate(m):         return float(m.get("Distance_To_Facility_Miles", 0)) > 25
def _cost_high(m):                 return float(m.get("Avg_Out_Of_Pocket_Cost", 0)) > 1368
def _cost_moderate(m):             return float(m.get("Avg_Out_Of_Pocket_Cost", 0)) > 716
def _tenure_short(m):              return int(m.get("Tenure_Months", 999)) < 12
def _referrals_zero(m):            return int(m.get("Referrals_Made", 99)) == 0
def _is_self_pay(m):               return str(m.get("Insurance_Type", "")).strip() == "Self-Pay"
def _is_medicaid(m):               return str(m.get("Insurance_Type", "")).strip() == "Medicaid"
def _is_senior(m):                 return int(m.get("Age", 0)) >= 65


# ==============================================================================
# RULE CATALOG -- one dict per rule
# ==============================================================================
RULE_CATALOG: list[dict[str, Any]] = [

    # --------------------------------------------------------------------------
    # CATEGORY 1: Service Recovery
    # --------------------------------------------------------------------------
    {
        "rule_id":   "SR-001",
        "enabled":   True,
        "category":  "Service Recovery",
        "action":    "Service Recovery -- Billing Issue",
        "condition": lambda m, s: _has_billing_issue(m),
        "priority_fn": _elevated_priority,
        "reason": (
            "A billing issue has been recorded for this member. "
            "Billing issues are associated with increased predicted churn risk in this model."
        ),
        "evidence_fn": lambda m, s: [
            "Billing issue flag is set to 1 (issue present)",
            f"Overall satisfaction score: {m.get('Overall_Satisfaction', 'N/A')} "
            f"(population median: 3.2)",
        ] + (["Overall satisfaction is below 2.5 (low range)"] if _is_overall_sat_low(m) else []),
        "next_step": (
            "Review the billing record for any unresolved items. "
            "Consider initiating a member services outreach to clarify billing details. "
            "Human review required before contact."
        ),
    },

    {
        "rule_id":   "SR-002",
        "enabled":   True,
        "category":  "Service Recovery",
        "action":    "Service Recovery -- Low Overall Satisfaction",
        "condition": lambda m, s: (
            _is_overall_sat_low(m) and
            _shap_feature_increases_risk(s, "Overall_Satisfaction")
        ),
        "priority_fn": _elevated_priority,
        "reason": (
            "Overall satisfaction score is in the low range (below 2.5) and is identified "
            "as an increasing-risk driver by the model's SHAP analysis."
        ),
        "evidence_fn": lambda m, s: [
            f"Overall satisfaction score: {m.get('Overall_Satisfaction', 'N/A')} "
            f"(below threshold of 2.5; population median: 3.2)",
        ] + (
            [f"Wait-time satisfaction also low: {m.get('Wait_Time_Satisfaction')}"] if _is_wait_sat_low(m) else []
        ) + (
            [f"Staff satisfaction also low: {m.get('Staff_Satisfaction')}"] if _is_staff_sat_low(m) else []
        ),
        "next_step": (
            "Initiate a service quality review for this member's recent encounters. "
            "Consider a structured member experience check-in call. "
            "Human review required before contact."
        ),
    },

    {
        "rule_id":   "SR-003",
        "enabled":   True,
        "category":  "Service Recovery",
        "action":    "Service Recovery -- Wait Time or Staff Experience Signal",
        "condition": lambda m, s: (
            (_is_wait_sat_low(m) or _is_staff_sat_low(m)) and
            _shap_any_feature_increases(s, ["Wait_Time_Satisfaction", "Staff_Satisfaction"])
        ),
        "priority_fn": _priority_by_risk,
        "reason": (
            "Wait-time satisfaction or staff satisfaction score is below the low-range "
            "threshold and is identified as a model risk driver. "
            "These scores reflect the member's recorded assessment of service experience."
        ),
        "evidence_fn": lambda m, s: [
            c for c in [
                f"Wait-time satisfaction: {m.get('Wait_Time_Satisfaction', 'N/A')}"
                if _is_wait_sat_low(m) else None,
                f"Staff satisfaction: {m.get('Staff_Satisfaction', 'N/A')}"
                if _is_staff_sat_low(m) else None,
            ] if c is not None
        ],
        "next_step": (
            "Flag this member's recent service encounters for quality review. "
            "Human review required before contact."
        ),
    },

    {
        "rule_id":   "SR-004",
        "enabled":   True,
        "category":  "Service Recovery",
        "action":    "Service Recovery -- Provider Rating Signal",
        "condition": lambda m, s: (
            _is_provider_rating_low(m) and
            _shap_feature_increases_risk(s, "Provider_Rating")
        ),
        "priority_fn": _priority_by_risk,
        "reason": (
            "Provider rating is below 3.1 (25th percentile) and is an increasing-risk "
            "driver in the SHAP analysis. This reflects the member's recorded provider rating."
        ),
        "evidence_fn": lambda m, s: [
            f"Provider rating: {m.get('Provider_Rating', 'N/A')} "
            f"(below 3.1; population 25th percentile: 3.1)",
        ],
        "next_step": (
            "Review whether the member has had consistent provider assignment. "
            "Consider whether care continuity options are available. "
            "Human review required before contact."
        ),
    },

    # --------------------------------------------------------------------------
    # CATEGORY 2: Benefit Education
    # --------------------------------------------------------------------------
    {
        "rule_id":   "BE-001",
        "enabled":   True,
        "category":  "Benefit Education",
        "action":    "Benefit Education -- No Portal Enrollment",
        "condition": lambda m, s: _no_portal_usage(m),
        "priority_fn": _priority_by_risk,
        "reason": (
            "Portal usage is recorded as 0, indicating the member has not used the member portal. "
            "Portal non-engagement is a measurable signal in this dataset."
        ),
        "evidence_fn": lambda m, s: [
            "Portal usage: 0 (member portal not used)",
            f"Tenure: {m.get('Tenure_Months', 'N/A')} months "
            f"(portal non-use over the full tenure period)",
        ] + ([
            "No referrals made -- potential indicator of limited plan navigation awareness"
        ] if _referrals_zero(m) else []),
        "next_step": (
            "Consider sharing portal enrollment and feature information through standard "
            "member communication channels. Human review required before contact."
        ),
    },

    {
        "rule_id":   "BE-002",
        "enabled":   True,
        "category":  "Benefit Education",
        "action":    "Benefit Education -- Low Visit Frequency",
        "condition": lambda m, s: (
            _visits_low(m) and
            _shap_feature_increases_risk(s, "Visits_Last_Year")
        ),
        "priority_fn": _priority_by_risk,
        "reason": (
            "Annual visit count is at or below 3 (below the 25th percentile of 4) "
            "and is identified as a risk driver. Low utilisation is an observable signal."
        ),
        "evidence_fn": lambda m, s: [
            f"Visits last year: {m.get('Visits_Last_Year', 'N/A')} "
            f"(population 25th percentile: 4; median: 8)",
        ] + ([
            "Portal usage also absent -- dual low-engagement signal"
        ] if _no_portal_usage(m) else []),
        "next_step": (
            "Consider sharing information on available preventive care services "
            "and appointment scheduling options. Human review required before contact."
        ),
    },

    {
        "rule_id":   "BE-003",
        "enabled":   True,
        "category":  "Benefit Education",
        "action":    "Benefit Education -- New Member Low Engagement",
        "condition": lambda m, s: (
            _tenure_short(m) and
            (_no_portal_usage(m) or _visits_low(m))
        ),
        "priority_fn": _priority_by_risk,
        "reason": (
            "Membership tenure is under 12 months and portal or visit utilisation is low. "
            "Early-tenure members with low engagement represent an observable onboarding signal."
        ),
        "evidence_fn": lambda m, s: [
            f"Tenure: {m.get('Tenure_Months', 'N/A')} months (under 12 months -- early tenure)",
            f"Visits last year: {m.get('Visits_Last_Year', 'N/A')}",
            f"Portal usage: {m.get('Portal_Usage', 'N/A')}",
        ],
        "next_step": (
            "Consider including this member in standard new-member welcome or onboarding "
            "communication workflows. Human review required before contact."
        ),
    },

    {
        "rule_id":   "BE-004",
        "enabled":   True,
        "category":  "Benefit Education",
        "action":    "Benefit Education -- Self-Pay or Medicaid Coverage Awareness",
        "condition": lambda m, s: (
            (_is_self_pay(m) or _is_medicaid(m)) and _cost_moderate(m)
        ),
        "priority_fn": _priority_by_risk,
        "reason": (
            "Insurance type is Self-Pay or Medicaid and out-of-pocket cost is above the "
            "population median. This is a measurable combination that may indicate "
            "relevant benefit information is available."
        ),
        "evidence_fn": lambda m, s: [
            f"Insurance type: {m.get('Insurance_Type', 'N/A')}",
            f"Average out-of-pocket cost: ${m.get('Avg_Out_Of_Pocket_Cost', 'N/A')} "
            f"(population median: $716)",
        ],
        "next_step": (
            "Consider sharing information on available cost-assistance programs, "
            "financial counselling resources, or plan benefit summaries. "
            "This is an informational action only. Human review required before contact."
        ),
    },

    # --------------------------------------------------------------------------
    # CATEGORY 3: Care Outreach
    # --------------------------------------------------------------------------
    {
        "rule_id":   "CO-001",
        "enabled":   True,
        "category":  "Care Outreach",
        "action":    "Care Outreach -- Extended Gap in Clinical Visit",
        "condition": lambda m, s: _days_since_visit_high(m),
        "priority_fn": _elevated_priority,
        "reason": (
            "Days since last clinical visit is 365 or more. "
            "This is an observable care gap signal based on recorded visit history."
        ),
        "evidence_fn": lambda m, s: [
            f"Days since last clinical visit: {m.get('Days_Since_Last_Visit', 'N/A')} days "
            f"(threshold: 365 days; population median: 363 days)",
        ] + ([
            f"Missed appointments: {m.get('Missed_Appointments', 'N/A')} "
            "(additional access signal)"
        ] if _missed_appts_high(m) else []),
        "next_step": (
            "Consider including this member in standard preventive care gap notification "
            "workflows. A care coordinator may review appointment history. "
            "Human review required before contact."
        ),
    },

    {
        "rule_id":   "CO-002",
        "enabled":   True,
        "category":  "Care Outreach",
        "action":    "Care Outreach -- Moderate Visit Gap",
        "condition": lambda m, s: (
            _days_since_visit_moderate(m) and
            not _days_since_visit_high(m) and
            _shap_feature_increases_risk(s, "Days_Since_Last_Visit")
        ),
        "priority_fn": _priority_by_risk,
        "reason": (
            "Days since last clinical visit is between 180 and 364 days "
            "and is identified as an increasing-risk driver in the SHAP analysis."
        ),
        "evidence_fn": lambda m, s: [
            f"Days since last clinical visit: {m.get('Days_Since_Last_Visit', 'N/A')} days "
            f"(population 25th percentile: 180 days)"
        ],
        "next_step": (
            "Consider including this member in routine preventive care reminder workflows. "
            "Human review required before contact."
        ),
    },

    {
        "rule_id":   "CO-003",
        "enabled":   True,
        "category":  "Care Outreach",
        "action":    "Care Outreach -- Frequent Missed Appointments",
        "condition": lambda m, s: _missed_appts_high(m),
        "priority_fn": _elevated_priority,
        "reason": (
            "Missed appointment count is 3 or more (above the 75th percentile). "
            "This is an observable care access barrier signal."
        ),
        "evidence_fn": lambda m, s: [
            f"Missed appointments: {m.get('Missed_Appointments', 'N/A')} "
            f"(population 75th percentile: 3; max: 5)",
            f"Distance to facility: {m.get('Distance_To_Facility_Miles', 'N/A')} miles"
            + (" -- access barrier may be a contributing factor" if _distance_moderate(m) else ""),
        ],
        "next_step": (
            "Consider whether telehealth, transportation support information, "
            "or appointment reminder tools are applicable. "
            "Human review required before contact."
        ),
    },

    {
        "rule_id":   "CO-004",
        "enabled":   True,
        "category":  "Care Outreach",
        "action":    "Care Outreach -- Geographic Access Barrier",
        "condition": lambda m, s: (
            _distance_high(m) and
            _shap_feature_increases_risk(s, "Distance_To_Facility_Miles")
        ),
        "priority_fn": _priority_by_risk,
        "reason": (
            "Distance to nearest facility is above 40 miles (above the 75th percentile) "
            "and is identified as an increasing-risk driver in the SHAP analysis. "
            "This is a measurable geographic access signal."
        ),
        "evidence_fn": lambda m, s: [
            f"Distance to facility: {m.get('Distance_To_Facility_Miles', 'N/A')} miles "
            f"(population 75th percentile: 37.2 miles)",
        ] + ([
            f"Missed appointments: {m.get('Missed_Appointments')} "
            "(co-occurring access signal)"
        ] if _missed_appts_high(m) else []),
        "next_step": (
            "Consider whether telehealth options or nearby network facility information "
            "are available to share. Human review required before contact."
        ),
    },

    {
        "rule_id":   "CO-005",
        "enabled":   True,
        "category":  "Care Outreach",
        "action":    "Care Outreach -- Senior Member with Access Signals",
        "condition": lambda m, s: (
            _is_senior(m) and
            (_days_since_visit_moderate(m) or _missed_appts_high(m) or _distance_moderate(m))
        ),
        "priority_fn": _elevated_priority,
        "reason": (
            "Member age is 65 or above and at least one care access signal is present "
            "(visit gap, missed appointments, or distance). "
            "Age 65+ combined with access barriers is an observable multi-signal pattern."
        ),
        "evidence_fn": lambda m, s: [
            f"Age: {m.get('Age', 'N/A')} years (65+ group)",
        ] + ([
            f"Days since last visit: {m.get('Days_Since_Last_Visit')} days"
        ] if _days_since_visit_moderate(m) else []) + ([
            f"Missed appointments: {m.get('Missed_Appointments')}"
        ] if _missed_appts_high(m) else []) + ([
            f"Distance to facility: {m.get('Distance_To_Facility_Miles')} miles"
        ] if _distance_moderate(m) else []),
        "next_step": (
            "Consider whether senior-specific care coordination or transport resources "
            "are relevant. Human review required before contact."
        ),
    },

    # --------------------------------------------------------------------------
    # CATEGORY 4: Pharmacy Support
    #   The dataset does NOT contain a direct pharmacy column.
    #   Rules are grounded in observable proxies: insurance type, out-of-pocket
    #   cost, and specialty -- which are legitimate observational indicators
    #   in a health-plan context.
    # --------------------------------------------------------------------------
    {
        "rule_id":   "PS-001",
        "enabled":   True,
        "category":  "Pharmacy Support",
        "action":    "Pharmacy Support -- High Out-of-Pocket Cost Signal",
        "condition": lambda m, s: (
            _cost_high(m) and
            not _is_self_pay(m)
        ),
        "priority_fn": _priority_by_risk,
        "reason": (
            "Average out-of-pocket cost is above $1,368 (75th percentile). "
            "In a health-plan context, high out-of-pocket cost is an observable signal "
            "that may relate to pharmacy or specialty medication spend. "
            "No specific medication data is available."
        ),
        "evidence_fn": lambda m, s: [
            f"Average out-of-pocket cost: ${m.get('Avg_Out_Of_Pocket_Cost', 'N/A')} "
            f"(population 75th percentile: $1,368; maximum: $1,999)",
            f"Insurance type: {m.get('Insurance_Type', 'N/A')}",
        ],
        "next_step": (
            "Consider whether medication cost-assistance programs, "
            "pharmacy benefit information, or generic substitution resources "
            "are available to share. This is an informational action only. "
            "Human review required before contact."
        ),
    },

    {
        "rule_id":   "PS-002",
        "enabled":   True,
        "category":  "Pharmacy Support",
        "action":    "Pharmacy Support -- Specialty Care Member",
        "condition": lambda m, s: (
            str(m.get("Specialty", "")).strip() in
            ["Cardiology", "Neurology", "Orthopedics"] and
            _cost_moderate(m)
        ),
        "priority_fn": _priority_by_risk,
        "reason": (
            "Member is assigned to a specialty associated with higher medication complexity "
            "(Cardiology, Neurology, or Orthopedics) and out-of-pocket cost is above the "
            "population median. No medication data is present in this dataset."
        ),
        "evidence_fn": lambda m, s: [
            f"Specialty: {m.get('Specialty', 'N/A')} (specialty care designation)",
            f"Average out-of-pocket cost: ${m.get('Avg_Out_Of_Pocket_Cost', 'N/A')} "
            f"(above population median of $716)",
        ],
        "next_step": (
            "Consider whether specialty pharmacy benefit information or "
            "medication adherence support resources are applicable. "
            "Human review required before contact."
        ),
    },

    # --------------------------------------------------------------------------
    # CATEGORY 5: General Member Engagement
    # --------------------------------------------------------------------------
    {
        "rule_id":   "GE-001",
        "enabled":   True,
        "category":  "General Member Engagement",
        "action":    "General Member Engagement -- Low Multi-Dimensional Engagement",
        "condition": lambda m, s: (
            _no_portal_usage(m) and
            _visits_low(m) and
            _referrals_zero(m)
        ),
        "priority_fn": _priority_by_risk,
        "reason": (
            "Portal usage is absent, annual visits are low (<= 3), and no referrals "
            "have been made. Three co-occurring low-engagement signals are present."
        ),
        "evidence_fn": lambda m, s: [
            f"Portal usage: 0 (no recorded portal use)",
            f"Visits last year: {m.get('Visits_Last_Year', 'N/A')} (at or below 3)",
            f"Referrals made: 0 (no referrals recorded)",
        ],
        "next_step": (
            "Consider general member engagement outreach covering portal enrollment, "
            "available services summary, and preventive care reminders. "
            "Human review required before contact."
        ),
    },

    {
        "rule_id":   "GE-002",
        "enabled":   True,
        "category":  "General Member Engagement",
        "action":    "General Member Engagement -- Long-Tenure Low Engagement",
        "condition": lambda m, s: (
            int(m.get("Tenure_Months", 0)) >= 60 and
            _no_portal_usage(m) and
            _visits_low(m)
        ),
        "priority_fn": _priority_by_risk,
        "reason": (
            "Member tenure is 60 months or more (at or above the population median) "
            "but portal usage is absent and annual visits are low. "
            "Long-tenure low-engagement is an observable dissonance signal."
        ),
        "evidence_fn": lambda m, s: [
            f"Tenure: {m.get('Tenure_Months', 'N/A')} months (60+ months)",
            f"Portal usage: 0",
            f"Visits last year: {m.get('Visits_Last_Year', 'N/A')}",
        ],
        "next_step": (
            "Consider loyalty-oriented member communication highlighting available "
            "plan services. Human review required before contact."
        ),
    },

    {
        "rule_id":   "GE-003",
        "enabled":   True,
        "category":  "General Member Engagement",
        "action":    "General Member Engagement -- SHAP Engagement Driver Identified",
        "condition": lambda m, s: (
            _shap_any_feature_increases(s, [
                "engagement_score", "Portal_Usage",
                "Visits_Last_Year", "Referrals_Made",
            ])
        ),
        "priority_fn": _priority_by_risk,
        "reason": (
            "One or more engagement-related features are identified as increasing-risk "
            "drivers in the model's SHAP analysis. "
            "This reflects the model's assessment of engagement signal strength."
        ),
        "evidence_fn": lambda m, s: [
            d["impact"] for d in s
            if d.get("feature") in [
                "engagement_score", "Portal_Usage",
                "Visits_Last_Year", "Referrals_Made"
            ] and d.get("direction") == "increases_risk"
        ] or [
            "Engagement-related feature identified as a SHAP risk driver"
        ],
        "next_step": (
            "Consider including this member in general engagement campaigns. "
            "Human review required before contact."
        ),
    },
    {
        "rule_id":   "GE-004",
        "enabled":   True,
        "category":  "General Member Engagement",
        "action":    "General Member Engagement -- Composite Satisfaction Risk Driver",
        "condition": lambda m, s: (
            _shap_feature_increases_risk(s, "composite_satisfaction") and
            float(m.get("Overall_Satisfaction", 5)) < 3.2
        ),
        "priority_fn": _priority_by_risk,
        "reason": (
            "Composite satisfaction score is identified as an increasing-risk driver "
            "and overall satisfaction is below the population median (3.2). "
            "This is a multi-satisfaction-dimension signal."
        ),
        "evidence_fn": lambda m, s: [
            f"Overall satisfaction: {m.get('Overall_Satisfaction', 'N/A')} "
            f"(population median: 3.2)",
            f"Staff satisfaction: {m.get('Staff_Satisfaction', 'N/A')}",
            f"Wait-time satisfaction: {m.get('Wait_Time_Satisfaction', 'N/A')}",
            f"Provider rating: {m.get('Provider_Rating', 'N/A')}",
        ],
        "next_step": (
            "Consider a general member satisfaction survey or check-in communication. "
            "Human review required before contact."
        ),
    },

    {
        "rule_id":   "GE-005",
        "enabled":   True,
        "category":  "General Member Engagement",
        "action":    "General Member Engagement -- Routine Member Monitoring",
        "condition": lambda m, s: True,   # Always fires (catch-all)
        "priority_fn": lambda m, rl: "Low",
        "reason": (
            "Standard routine monitoring applies to all members regardless of current risk "
            "signal. This is a baseline engagement recommendation."
        ),
        "evidence_fn": lambda m, s: [
            f"Risk level at time of assessment: {m.get('risk_level', 'not computed')}",
            f"Tenure: {m.get('Tenure_Months', 'N/A')} months",
            f"Visits last year: {m.get('Visits_Last_Year', 'N/A')}",
        ],
        "next_step": (
            "Include member in standard periodic engagement communications "
            "and routine care reminder workflows. "
            "Human review required before contact."
        ),
    },
]


# ==============================================================================
# 2.  PRIORITY ORDERING
#     Used to sort recommendations before returning.
# ==============================================================================
PRIORITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}

CATEGORY_ORDER = {
    "Service Recovery":          0,
    "Care Outreach":             1,
    "Benefit Education":         2,
    "Pharmacy Support":          3,
    "General Member Engagement": 4,
}


# ==============================================================================
# 3.  CORE FUNCTION -- get_retention_recommendations
# ==============================================================================
def get_retention_recommendations(
    member_data: dict,
    shap_drivers: list,
    risk_level: str = None,
    deduplicate_categories: bool = True,
    max_recommendations: int = 3,
) -> list[dict]:
    """
    Generate a prioritised list of 1-3 consolidated retention recommendations for a single member.

    Rules belonging to the same action category (e.g. Service Recovery, Care Outreach)
    are merged into a single recommendation card with consolidated observable evidence.
    """
    if risk_level is None:
        risk_level = member_data.get("risk_level", "Medium")

    category_groups: dict[str, list[dict]] = {}

    for rule in RULE_CATALOG:
        if not rule.get("enabled", True):
            continue

        try:
            triggered = rule["condition"](member_data, shap_drivers)
        except Exception:
            triggered = False

        if not triggered:
            continue

        try:
            priority = rule["priority_fn"](member_data, risk_level)
        except Exception:
            priority = "Medium"

        try:
            evidence = rule["evidence_fn"](member_data, shap_drivers)
        except Exception:
            evidence = []

        fired = {
            "rule_id":            rule["rule_id"],
            "action":             rule["action"],
            "category":           rule["category"],
            "priority":           priority,
            "reason":             rule["reason"],
            "evidence":           [e for e in evidence if e],
            "suggested_next_step": rule["next_step"],
        }

        cat = rule["category"]
        if cat not in category_groups:
            category_groups[cat] = []
        category_groups[cat].append(fired)

    consolidated = []

    # Filter out catch-all baseline GE-005 if other specific recommendations exist
    if len(category_groups) > 1 and "General Member Engagement" in category_groups:
        category_groups["General Member Engagement"] = [
            r for r in category_groups["General Member Engagement"] if r["rule_id"] != "GE-005"
        ]
        if not category_groups["General Member Engagement"]:
            del category_groups["General Member Engagement"]

    CATEGORY_SUGGESTIONS = {
        "Service Recovery": "Review unresolved billing/service issues, assess provider continuity, and coordinate priority service recovery.",
        "Care Outreach": "Initiate care coordination outreach to address clinical gaps, verify transportation access support, and facilitate scheduling.",
        "Benefit Education": "Share information on preventive care coverage, digital member portal access, and financial/benefit assistance programs.",
        "Pharmacy Support": "Review specialty pharmacy benefits, medication cost-assistance resources, and adherence support options.",
        "General Member Engagement": "Include member in routine engagement communications, preventive care check-ins, and onboarding support."
    }

    CATEGORY_REASONS = {
        "Service Recovery": "Multiple service-related factors are contributing to the member's predicted churn risk.",
        "Care Outreach": "Observable clinical care gap and appointment access barriers detected for this member.",
        "Benefit Education": "Member engagement and plan utilization signals indicate key benefit education opportunities.",
        "Pharmacy Support": "Cost and specialty care factors suggest potential pharmacy benefit support opportunities.",
        "General Member Engagement": "Baseline engagement and satisfaction indicators suggest routine proactive outreach."
    }

    for cat, rules in category_groups.items():
        if not rules:
            continue
        
        # Pick highest priority in this category
        best_priority = min(rules, key=lambda r: PRIORITY_ORDER.get(r["priority"], 99))["priority"]
        
        # Merge evidence items cleanly (remove duplicate strings)
        merged_evidence = []
        for r in rules:
            for item in r["evidence"]:
                if item not in merged_evidence:
                    merged_evidence.append(item)

        # Formulate consolidated card
        reason_text = CATEGORY_REASONS.get(cat, rules[0]["reason"]) if len(rules) > 1 else rules[0]["reason"]
        next_step = CATEGORY_SUGGESTIONS.get(cat, rules[0]["suggested_next_step"])

        consolidated.append({
            "rule_id": rules[0]["rule_id"],
            "action": cat,
            "category": cat,
            "priority": best_priority,
            "reason": reason_text,
            "evidence": merged_evidence,
            "suggested_next_step": next_step
        })

    # Sort by priority first (Critical > High > Medium > Low), then category order
    consolidated.sort(key=lambda r: (
        PRIORITY_ORDER.get(r["priority"], 99),
        CATEGORY_ORDER.get(r["category"], 99),
    ))

    # Cap at max_recommendations (1 to 3 cards max)
    return consolidated[:max_recommendations]


# ==============================================================================
# 4.  END-TO-END HELPER -- advise_member(member_id)
# ==============================================================================
def advise_member(member_id: str, top_shap: int = 5) -> dict:
    """
    Full pipeline: load member data, get SHAP explanation, generate recommendations.

    Parameters
    ----------
    member_id : str
        PatientID (e.g., "C20983").
    top_shap : int
        Number of SHAP drivers to request (default 5).

    Returns
    -------
    dict with keys:
        member_id       : str
        risk_score      : float
        risk_level      : str
        shap_drivers    : list
        recommendations : list[dict]
        total_recommendations : int
        disclaimer      : str
    """
    # Lazy import to avoid circular dependency when shap_explainer is not needed
    from shap_explainer import get_member_explanation
    import pandas as pd

    # Load raw dataset for member_data
    raw_df = pd.read_csv("patient_churn_dataset.csv.xls")
    match  = raw_df[raw_df["PatientID"] == member_id]

    if match.empty:
        return {
            "member_id":            member_id,
            "error":                f"Member ID '{member_id}' not found.",
            "recommendations":      [],
            "total_recommendations": 0,
        }

    raw_row     = match.iloc[0].to_dict()
    explanation = get_member_explanation(member_id, top_n=top_shap, save_waterfall=False)

    if explanation.get("error"):
        return {
            "member_id":             member_id,
            "error":                 explanation["error"],
            "recommendations":       [],
            "total_recommendations": 0,
        }

    risk_score   = explanation["risk_score"]
    risk_level   = explanation["risk_level"]
    shap_drivers = explanation["drivers"]

    # Merge risk info into member_data for priority_fn use
    member_data_enriched = copy.copy(raw_row)
    member_data_enriched["risk_score"] = risk_score
    member_data_enriched["risk_level"] = risk_level

    recommendations = get_retention_recommendations(
        member_data  = member_data_enriched,
        shap_drivers = shap_drivers,
        risk_level   = risk_level,
    )

    return {
        "member_id":            member_id,
        "risk_score":           risk_score,
        "risk_level":           risk_level,
        "shap_drivers":         shap_drivers,
        "recommendations":      recommendations,
        "total_recommendations": len(recommendations),
        "disclaimer": (
            "This output is produced by a rule-based decision-support tool. "
            "No claims about member intent, health status, or financial situation "
            "are made. Human review is required before any action is taken."
        ),
    }


# ==============================================================================
# 5.  MAIN -- demo run when called as a script
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  RETENTION ADVISOR -- RULE-BASED RECOMMENDATION ENGINE")
    print("=" * 70)
    print()

    # Demonstrate with 4 representative members (one per risk zone)
    DEMO_MEMBERS = ["C21047", "C21002", "C20983", "C21031"]

    import os
    import json

    all_results = []

    for mid in DEMO_MEMBERS:
        print(f"\n{'-' * 70}")
        print(f"  Member: {mid}")
        print(f"{'-' * 70}")

        result = advise_member(mid)
        all_results.append(result)

        if result.get("error"):
            print(f"  ERROR: {result['error']}")
            continue

        print(f"  Risk Score      : {result['risk_score']:.4f} ({result['risk_score']*100:.1f}%)")
        print(f"  Risk Level      : {result['risk_level']}")
        print(f"  Recommendations : {result['total_recommendations']}")
        print()

        for i, rec in enumerate(result["recommendations"], 1):
            print(f"  [{i}] [{rec['priority']:8s}] {rec['action']}")
            print(f"       Category : {rec['category']}")
            print(f"       Reason   : {rec['reason']}")
            print(f"       Evidence :")
            for ev in rec["evidence"]:
                print(f"         - {ev}")
            print(f"       Next Step: {rec['suggested_next_step']}")
            print()

    # Save results to JSON
    out_path = os.path.join("outputs", "retention_recommendations.json")
    os.makedirs("outputs", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"  RETENTION ADVISOR COMPLETE")
    print(f"  Full results saved to: {out_path}")
    print(f"  Rules defined in RULE_CATALOG (shap_explainer.py imports required).")
    print(f"  All actions require human review before execution.")
    print(f"{'=' * 70}")
