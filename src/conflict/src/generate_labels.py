import pandas as pd
import numpy as np

INPUT = "data/processed/aligned_decisions.csv"
OUTPUT = "data/processed/conflict_labeled_real.csv"

# =========================================================
# Configuration
# =========================================================

ACTION_THRESHOLD_PP = 10.0
THROUGHPUT_LOSS_THRESHOLD = 0.10

# =========================================================
# Load aligned data
# =========================================================

df = pd.read_csv(INPUT)

print("Input shape:", df.shape)

# =========================================================
# Separate slices
# =========================================================

a = (
    df[df["slice_name"] == "slice-a"]
    .sort_values("action_time")
    .reset_index(drop=True)
)

b = (
    df[df["slice_name"] == "slice-b"]
    .sort_values("action_time")
    .reset_index(drop=True)
)

# =========================================================
# Create one decision-level dataframe
# =========================================================

decision = pd.DataFrame()

decision["decision_id"] = a["decision_id"]
decision["run_id"] = a["run_id"]
decision["action_time"] = a["action_time"]

decision["action_a"] = a["action_value"].values
decision["action_b"] = b["action_value"].values

# =========================================================
# Previous allocation
# =========================================================

decision["prev_action_a"] = (
    decision["action_a"].shift(1)
)

decision["action_change_pp"] = (
    decision["action_a"]
    - decision["prev_action_a"]
).abs()

# =========================================================
# Determine direction of reallocation
# =========================================================

decision["direction"] = np.where(
    decision["action_a"] > decision["prev_action_a"],
    "A_gains",
    np.where(
        decision["action_a"] < decision["prev_action_a"],
        "A_loses",
        "no_change"
    )
)

# =========================================================
# Determine affected slice
# =========================================================

decision["affected_slice"] = np.where(
    decision["direction"] == "A_gains",
    "B",
    np.where(
        decision["direction"] == "A_loses",
        "A",
        "none"
    )
)

# =========================================================
# TOS before/after
# =========================================================

decision["before_TOS_a"] = (
    a["before_dl_mbps"].values
    + a["before_ul_mbps"].values
)

decision["before_TOS_b"] = (
    b["before_dl_mbps"].values
    + b["before_ul_mbps"].values
)

decision["after_TOS_a"] = (
    a["after_dl_mbps"].values
    + a["after_ul_mbps"].values
)

decision["after_TOS_b"] = (
    b["after_dl_mbps"].values
    + b["after_ul_mbps"].values
)

decision["delta_TOS_a"] = (
    decision["after_TOS_a"]
    - decision["before_TOS_a"]
)

decision["delta_TOS_b"] = (
    decision["after_TOS_b"]
    - decision["before_TOS_b"]
)

# =========================================================
# Select affected slice
# =========================================================

decision["affected_before_TOS"] = np.where(
    decision["affected_slice"] == "A",
    decision["before_TOS_a"],
    np.where(
        decision["affected_slice"] == "B",
        decision["before_TOS_b"],
        0
    )
)

decision["affected_delta_TOS"] = np.where(
    decision["affected_slice"] == "A",
    decision["delta_TOS_a"],
    np.where(
        decision["affected_slice"] == "B",
        decision["delta_TOS_b"],
        0
    )
)

# =========================================================
# Relative throughput loss
# =========================================================

decision["relative_TOS_loss"] = np.where(
    decision["affected_before_TOS"] > 0,
    -decision["affected_delta_TOS"]
    / decision["affected_before_TOS"],
    0
)

decision["relative_TOS_loss"] = (
    decision["relative_TOS_loss"]
    .clip(0, 1)
)

# =========================================================
# Reliability deterioration
# =========================================================

decision["affected_delta_dl_bler"] = np.where(
    decision["affected_slice"] == "A",
    a["delta_dl_bler"].values,
    np.where(
        decision["affected_slice"] == "B",
        b["delta_dl_bler"].values,
        0
    )
)

decision["affected_delta_ul_bler"] = np.where(
    decision["affected_slice"] == "A",
    a["delta_ul_bler"].values,
    np.where(
        decision["affected_slice"] == "B",
        b["delta_ul_bler"].values,
        0
    )
)

decision["affected_delta_retx"] = np.where(
    decision["affected_slice"] == "A",
    a["delta_dl_retx_frac"].values,
    np.where(
        decision["affected_slice"] == "B",
        b["delta_dl_retx_frac"].values,
        0
    )
)

# Positive values indicate deterioration.

decision["bler_deterioration"] = (
    decision["affected_delta_dl_bler"].clip(lower=0)
    +
    decision["affected_delta_ul_bler"].clip(lower=0)
) / 2

decision["retx_deterioration"] = (
    decision["affected_delta_retx"]
    .clip(lower=0)
)

# =========================================================
# Reliability evidence
# =========================================================

decision["reliability_deterioration"] = (
    (decision["bler_deterioration"] > 0)
    |
    (decision["retx_deterioration"] > 0)
)

# =========================================================
# Generate binary conflict label
# =========================================================

decision["conflict"] = (
    (decision["action_change_pp"] >= ACTION_THRESHOLD_PP)
    &
    (decision["relative_TOS_loss"] >= THROUGHPUT_LOSS_THRESHOLD)
    &
    decision["reliability_deterioration"]
).astype(int)

# =========================================================
# Remove first decision because it has no previous action
# =========================================================

decision = decision.iloc[1:].copy()

# =========================================================
# Save
# =========================================================

decision.to_csv(OUTPUT, index=False)

# =========================================================
# Results
# =========================================================

print("\n" + "=" * 60)
print("CONFLICT LABEL RESULTS")
print("=" * 60)

print("Output shape:", decision.shape)

print("\nLabel distribution:")

counts = decision["conflict"].value_counts().sort_index()

print(counts)

print("\nLabel percentages:")

print(
    decision["conflict"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)

print("\nPositive conflicts:", int(decision["conflict"].sum()))

print(
    "Conflict rate:",
    round(
        decision["conflict"].mean() * 100,
        2
    ),
    "%"
)

# =========================================================
# Condition counts
# =========================================================

print("\nCondition counts:")

condition_action = (
    decision["action_change_pp"]
    >= ACTION_THRESHOLD_PP
)

condition_throughput = (
    decision["relative_TOS_loss"]
    >= THROUGHPUT_LOSS_THRESHOLD
)

condition_reliability = (
    decision["reliability_deterioration"]
)

print(
    "Action threshold:",
    int(condition_action.sum())
)

print(
    "Throughput threshold:",
    int(condition_throughput.sum())
)

print(
    "Reliability deterioration:",
    int(condition_reliability.sum())
)

print(
    "Action + throughput:",
    int(
        (
            condition_action
            & condition_throughput
        ).sum()
    )
)

print(
    "Final conflict:",
    int(decision["conflict"].sum())
)

# =========================================================
# Show examples
# =========================================================

print("\nExample conflict events:")

examples = (
    decision[decision["conflict"] == 1]
    .sort_values(
        ["relative_TOS_loss", "action_change_pp"],
        ascending=False
    )
    .head(15)
)

print(
    examples[
        [
            "decision_id",
            "action_a",
            "action_b",
            "action_change_pp",
            "affected_slice",
            "affected_before_TOS",
            "affected_delta_TOS",
            "relative_TOS_loss",
            "bler_deterioration",
            "retx_deterioration",
            "conflict"
        ]
    ].to_string(index=False)
)

print("\nSaved:", OUTPUT)