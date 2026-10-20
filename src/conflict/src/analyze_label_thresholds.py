import pandas as pd
import numpy as np

INPUT = "data/processed/aligned_decisions.csv"

df = pd.read_csv(INPUT)

print("Input shape:", df.shape)

# =========================================================
# 1. Convert aligned data into one row per decision
# =========================================================

a = df[df["slice_name"] == "slice-a"].copy()
b = df[df["slice_name"] == "slice-b"].copy()

a = a.sort_values("action_time").reset_index(drop=True)
b = b.sort_values("action_time").reset_index(drop=True)

# Use slice-a allocation as the reference because:
# action_a + action_b = 100%

decision = pd.DataFrame()

decision["decision_id"] = a["decision_id"]
decision["action_a"] = a["action_value"]
decision["action_b"] = b["action_value"]

# =========================================================
# 2. Previous allocation and allocation change
# =========================================================

decision["prev_action_a"] = decision["action_a"].shift(1)

decision["action_change_pp"] = (
    decision["action_a"] - decision["prev_action_a"]
).abs()

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
# 3. Identify affected slice
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
# 4. Calculate TOS for both slices
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
# 5. Select affected-slice throughput
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
# 6. Relative throughput loss
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
# 7. Reliability deterioration
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

# Positive values represent deterioration.

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
# 8. Remove first decision
# =========================================================

decision = decision.iloc[1:].copy()

# =========================================================
# 9. Basic information
# =========================================================

print("\nDecision-level dataset:", decision.shape)

print("\nDirection counts:")
print(decision["direction"].value_counts())

# =========================================================
# 10. Distribution analysis
# =========================================================

print("\n" + "=" * 65)
print("ACTION CHANGE (percentage points)")
print("=" * 65)

print(
    decision["action_change_pp"].describe(
        percentiles=[
            0.50,
            0.75,
            0.90,
            0.95,
            0.99
        ]
    )
)

print("\n" + "=" * 65)
print("RELATIVE THROUGHPUT LOSS")
print("=" * 65)

print(
    decision["relative_TOS_loss"].describe(
        percentiles=[
            0.50,
            0.75,
            0.90,
            0.95,
            0.99
        ]
    )
)

print("\n" + "=" * 65)
print("BLER DETERIORATION")
print("=" * 65)

print(
    decision["bler_deterioration"].describe(
        percentiles=[
            0.50,
            0.75,
            0.90,
            0.95,
            0.99
        ]
    )
)

print("\n" + "=" * 65)
print("RETRANSMISSION DETERIORATION")
print("=" * 65)

print(
    decision["retx_deterioration"].describe(
        percentiles=[
            0.50,
            0.75,
            0.90,
            0.95,
            0.99
        ]
    )
)

# =========================================================
# 11. Candidate event counts
# =========================================================

print("\n" + "=" * 65)
print("ACTION + THROUGHPUT CONDITIONS")
print("=" * 65)

for action_threshold in [2, 5, 10, 15]:

    for loss_threshold in [0.05, 0.10, 0.15, 0.20]:

        mask = (
            (decision["action_change_pp"] >= action_threshold)
            &
            (decision["relative_TOS_loss"] >= loss_threshold)
        )

        print(
            f"action >= {action_threshold:>2} pp | "
            f"loss >= {loss_threshold:.2f} | "
            f"count = {mask.sum():4d} | "
            f"{mask.mean() * 100:5.2f}%"
        )

# =========================================================
# 12. Add reliability condition
# =========================================================

print("\n" + "=" * 65)
print("ACTION + THROUGHPUT + RELIABILITY")
print("=" * 65)

for action_threshold in [5, 10, 15]:

    for loss_threshold in [0.05, 0.10, 0.15]:

        mask = (
            (decision["action_change_pp"] >= action_threshold)
            &
            (decision["relative_TOS_loss"] >= loss_threshold)
            &
            (
                (decision["bler_deterioration"] > 0)
                |
                (decision["retx_deterioration"] > 0)
            )
        )

        print(
            f"action >= {action_threshold:>2} pp | "
            f"loss >= {loss_threshold:.2f} | "
            f"reliability > 0 | "
            f"count = {mask.sum():4d} | "
            f"{mask.mean() * 100:5.2f}%"
        )

# =========================================================
# 13. Strong-event inspection
# =========================================================

print("\n" + "=" * 65)
print("STRONGEST CANDIDATE EVENTS")
print("=" * 65)

candidates = decision[
    (decision["action_change_pp"] >= 10)
    &
    (decision["relative_TOS_loss"] >= 0.10)
].copy()

candidates["event_strength"] = (
    candidates["action_change_pp"] / 100
    *
    candidates["relative_TOS_loss"]
)

print(
    candidates[
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
            "event_strength"
        ]
    ]
    .sort_values("event_strength", ascending=False)
    .head(20)
    .to_string(index=False)
)

print("\nAnalysis complete.")