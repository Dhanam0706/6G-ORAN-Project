import pandas as pd
import numpy as np

INPUT_FILE = "data/processed/aligned_decisions.csv"
OUTPUT_FILE = "data/processed/conflict_features_real.csv"

EPS = 1e-9

df = pd.read_csv(INPUT_FILE)

print("Input shape:", df.shape)

# ---------------------------------------------------------
# Separate slices
# ---------------------------------------------------------

a = df[df["slice_name"] == "slice-a"].copy()
b = df[df["slice_name"] == "slice-b"].copy()

a = a.sort_values("decision_id").reset_index(drop=True)
b = b.sort_values("decision_id").reset_index(drop=True)

# ---------------------------------------------------------
# Create decision-level dataframe
# ---------------------------------------------------------

features = a[
    [
        "decision_id",
        "run_id",
        "action_time",
    ]
].copy()

features["action_a"] = a["action_value"].values
features["action_b"] = b["action_value"].values

# ---------------------------------------------------------
# ACTION CHANGE SCORE
# ---------------------------------------------------------

features["allocation_change"] = (
    features["action_a"].diff().abs()
)

features["ACS"] = (
    features["allocation_change"] / 100.0
)

# First decision has no previous action
features.loc[features.index[0], "ACS"] = 0.0
features.loc[features.index[0], "allocation_change"] = 0.0

# ---------------------------------------------------------
# OPPOSING / COMPETITION SCORE
# ---------------------------------------------------------

features["OCS"] = (
    (features["action_a"] - features["action_b"]).abs() / 100.0
)

# ---------------------------------------------------------
# RESOURCE OCCUPANCY SCORE
# ---------------------------------------------------------

a_ros = (
    a["before_dl_prb_util"] +
    a["before_ul_prb_util"]
) / 2.0

b_ros = (
    b["before_dl_prb_util"] +
    b["before_ul_prb_util"]
) / 2.0

features["ROS_a"] = a_ros.values
features["ROS_b"] = b_ros.values

features["ROS"] = (
    features["ROS_a"] +
    features["ROS_b"]
) / 2.0

# ---------------------------------------------------------
# THROUGHPUT / OUTPUT SCORE
# ---------------------------------------------------------

features["TOS_a_before"] = (
    a["before_dl_mbps"] +
    a["before_ul_mbps"]
).values

features["TOS_b_before"] = (
    b["before_dl_mbps"] +
    b["before_ul_mbps"]
).values

features["TOS_a_after"] = (
    a["after_dl_mbps"] +
    a["after_ul_mbps"]
).values

features["TOS_b_after"] = (
    b["after_dl_mbps"] +
    b["after_ul_mbps"]
).values

features["delta_TOS_a"] = (
    features["TOS_a_after"] -
    features["TOS_a_before"]
)

features["delta_TOS_b"] = (
    features["TOS_b_after"] -
    features["TOS_b_before"]
)

features["TOS"] = (
    features["TOS_a_before"] +
    features["TOS_b_before"]
) / 2.0

# ---------------------------------------------------------
# LINK / RADIO HEALTH SCORE
# ---------------------------------------------------------
# Higher KIS = healthier radio/link condition.
#
# Components:
#   BLER       -> lower is better
#   Retransmit -> lower is better
#   SNR        -> higher is better
#
# We use bounded reference transformations instead of
# dataset-wide min-max scaling.

def health_bler(bler):
    return 1.0 - np.clip(bler / 0.5, 0.0, 1.0)


def health_retx(retx):
    return 1.0 - np.clip(retx / 0.5, 0.0, 1.0)


def health_snr(snr):
    return np.clip(snr / 50.0, 0.0, 1.0)


def compute_kis(frame):
    bler = (
        health_bler(frame["before_dl_bler"]) +
        health_bler(frame["before_ul_bler"])
    ) / 2.0

    retx = health_retx(
        frame["before_dl_retx_frac"]
    )

    snr = (
        health_snr(frame["before_pusch_snr"]) +
        health_snr(frame["before_pucch_snr"])
    ) / 2.0

    return (
        bler +
        retx +
        snr
    ) / 3.0


features["KIS_a"] = compute_kis(a).values
features["KIS_b"] = compute_kis(b).values

features["KIS"] = (
    features["KIS_a"] +
    features["KIS_b"]
) / 2.0

# ---------------------------------------------------------
# PERFORMANCE VARIATION SCORE
# ---------------------------------------------------------
# Measures adverse performance degradation.
#
# Components:
#   1. Relative throughput loss
#   2. DL BLER deterioration
#   3. UL BLER deterioration
#   4. DL retransmission deterioration
#
# Each component is bounded between 0 and 1.
# Higher PVS = greater performance degradation.

def relative_loss(before, after):
    return np.clip(
        (before - after) /
        (np.abs(before) + EPS),
        0.0,
        1.0
    )


# ---------------------------------------------------------
# Throughput degradation
# ---------------------------------------------------------

tos_loss_a = relative_loss(
    features["TOS_a_before"],
    features["TOS_a_after"]
)

tos_loss_b = relative_loss(
    features["TOS_b_before"],
    features["TOS_b_after"]
)


# ---------------------------------------------------------
# BLER deterioration
# ---------------------------------------------------------

dl_bler_a = np.clip(
    a["after_dl_bler"].values -
    a["before_dl_bler"].values,
    0.0,
    0.5
) / 0.5

ul_bler_a = np.clip(
    a["after_ul_bler"].values -
    a["before_ul_bler"].values,
    0.0,
    0.5
) / 0.5

dl_bler_b = np.clip(
    b["after_dl_bler"].values -
    b["before_dl_bler"].values,
    0.0,
    0.5
) / 0.5

ul_bler_b = np.clip(
    b["after_ul_bler"].values -
    b["before_ul_bler"].values,
    0.0,
    0.5
) / 0.5


# ---------------------------------------------------------
# Retransmission deterioration
# ---------------------------------------------------------

retx_a = np.clip(
    a["after_dl_retx_frac"].values -
    a["before_dl_retx_frac"].values,
    0.0,
    0.5
) / 0.5

retx_b = np.clip(
    b["after_dl_retx_frac"].values -
    b["before_dl_retx_frac"].values,
    0.0,
    0.5
) / 0.5


# ---------------------------------------------------------
# Slice-level PVS
# ---------------------------------------------------------

features["PVS_a"] = (
    tos_loss_a +
    dl_bler_a +
    ul_bler_a +
    retx_a
) / 4.0

features["PVS_b"] = (
    tos_loss_b +
    dl_bler_b +
    ul_bler_b +
    retx_b
) / 4.0

features["PVS"] = (
    features["PVS_a"] +
    features["PVS_b"]
) / 2.0

# ---------------------------------------------------------
# CROSS-SLICE CONFLICT SCORE
# ---------------------------------------------------------
# Determine which slice lost allocation.
#
# A gains -> B affected
# A loses -> A affected
#
# COS combines allocation change with degradation
# experienced by the affected slice.

features["affected_PVS"] = np.where(
    features["allocation_change"] > 0,
    features["PVS_b"],
    np.where(
        features["allocation_change"] < 0,
        features["PVS_a"],
        0.0
    )
)

features["COS"] = (
    features["ACS"] *
    features["affected_PVS"]
)

# ---------------------------------------------------------
# Clean numerical values
# ---------------------------------------------------------

numeric_cols = [
    "ROS",
    "ROS_a",
    "ROS_b",
    "TOS",
    "TOS_a_before",
    "TOS_b_before",
    "TOS_a_after",
    "TOS_b_after",
    "delta_TOS_a",
    "delta_TOS_b",
    "ACS",
    "OCS",
    "KIS",
    "KIS_a",
    "KIS_b",
    "PVS",
    "PVS_a",
    "PVS_b",
    "COS",
    "affected_PVS",
]

features[numeric_cols] = (
    features[numeric_cols]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

features.to_csv(
    OUTPUT_FILE,
    index=False
)

print("Feature shape:", features.shape)
print("Saved:", OUTPUT_FILE)

print("\nFeature statistics:")
print(
    features[
        [
            "ROS",
            "TOS",
            "ACS",
            "OCS",
            "KIS",
            "PVS",
            "COS",
        ]
    ]
    .describe()
    .round(6)
    .to_string()
)

print("\nTop candidate conflict cases:")

print(
    features[
        [
            "decision_id",
            "action_a",
            "action_b",
            "ACS",
            "OCS",
            "KIS",
            "PVS",
            "PVS_a",
            "PVS_b",
            "COS",
            "delta_TOS_a",
            "delta_TOS_b",
        ]
    ]
    .sort_values("COS", ascending=False)
    .head(15)
    .round(5)
    .to_string(index=False)
)