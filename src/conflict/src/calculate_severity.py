import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# =========================================================
# CONFIGURATION
# =========================================================

FEATURE_FILE = "data/processed/conflict_features_real.csv"
MODEL_FILE = "models/conflict_xgboost_stratified.pkl"
OUTPUT_FILE = "data/processed/conflict_severity_real.csv"

MODEL_FEATURES = [
    "ROS",
    "TOS",
    "ACS",
    "OCS",
    "KIS"
]


# =========================================================
# LOAD DATA
# =========================================================

print("=" * 60)
print("O-RAN CONFLICT SEVERITY CALCULATION")
print("=" * 60)

features = pd.read_csv(FEATURE_FILE)

print("\nFeature dataset:")
print("Shape:", features.shape)


# =========================================================
# LOAD TRAINED XGBOOST MODEL
# =========================================================

print("\nLoading XGBoost model...")

model = joblib.load(MODEL_FILE)

print("Model loaded:", MODEL_FILE)


# =========================================================
# PREPARE MODEL INPUT
# =========================================================

X = features[MODEL_FEATURES].copy()

X = (
    X
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
)


# =========================================================
# CONFLICT PROBABILITY
# =========================================================

print("\nCalculating conflict probability...")

conflict_probability = model.predict_proba(X)[:, 1]

features["conflict_probability"] = conflict_probability


# =========================================================
# POST-EVENT EVIDENCE
# =========================================================

# affected_PVS:
# Performance degradation experienced by the slice
# that lost allocation.

affected_pvs = np.clip(
    features["affected_PVS"].values,
    0.0,
    1.0
)

# COS:
# Cross-slice conflict evidence.
#
# Existing definition:
# COS = ACS * affected_PVS

cos = np.clip(
    features["COS"].values,
    0.0,
    1.0
)


# =========================================================
# SEVERITY SCORE
# =========================================================

features["severity_score"] = 100.0 * (
    0.50 * features["conflict_probability"]
    + 0.30 * affected_pvs
    + 0.20 * cos
)

# Numerical safety
features["severity_score"] = np.clip(
    features["severity_score"],
    0.0,
    100.0
)


# =========================================================
# SEVERITY BAND
# =========================================================

def severity_band(score):

    if score < 25:
        return "Low"

    elif score < 50:
        return "Moderate"

    elif score < 75:
        return "High"

    else:
        return "Critical"


features["severity_band"] = (
    features["severity_score"]
    .apply(severity_band)
)


# =========================================================
# SAVE RESULTS
# =========================================================

Path(OUTPUT_FILE).parent.mkdir(
    parents=True,
    exist_ok=True
)

features.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# SUMMARY
# =========================================================

print("\n" + "=" * 60)
print("SEVERITY CALCULATION COMPLETE")
print("=" * 60)

print("\nSeverity statistics:")

print(
    features[
        [
            "conflict_probability",
            "affected_PVS",
            "COS",
            "severity_score"
        ]
    ]
    .describe()
    .round(4)
    .to_string()
)


print("\nSeverity band distribution:")

print(
    features["severity_band"]
    .value_counts()
    .reindex(
        ["Low", "Moderate", "High", "Critical"],
        fill_value=0
    )
    .to_string()
)


# =========================================================
# TOP SEVERITY EVENTS
# =========================================================

print("\nTop 15 severity events:")

print(
    features[
        [
            "decision_id",
            "action_time",
            "action_a",
            "action_b",
            "ACS",
            "OCS",
            "KIS",
            "affected_PVS",
            "COS",
            "conflict_probability",
            "severity_score",
            "severity_band"
        ]
    ]
    .sort_values(
        "severity_score",
        ascending=False
    )
    .head(15)
    .round(5)
    .to_string(index=False)
)


print("\nSaved:")
print(OUTPUT_FILE)