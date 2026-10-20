import pandas as pd
import numpy as np

INPUT_FILE = "data/processed/conflict_features_real.csv"

df = pd.read_csv(INPUT_FILE)

print("\n" + "=" * 70)
print("REAL FEATURE VALIDATION")
print("=" * 70)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

# ---------------------------------------------------------
# 1. Basic statistics
# ---------------------------------------------------------

features = ["ROS", "TOS", "ACS", "OCS", "KIS", "PVS", "COS"]

print("\n" + "-" * 70)
print("1. FEATURE STATISTICS")
print("-" * 70)

print(
    df[features]
    .describe()
    .T
    .round(6)
    .to_string()
)

# ---------------------------------------------------------
# 2. Missing / infinite values
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("2. MISSING / INFINITE VALUES")
print("-" * 70)

print("\nMissing values:")
print(df[features].isna().sum())

print("\nInfinite values:")
print(np.isinf(df[features]).sum())

# ---------------------------------------------------------
# 3. Percentage of zero values
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("3. ZERO-VALUE ANALYSIS")
print("-" * 70)

zero_stats = {}

for f in features:
    zero_stats[f] = {
        "zero_count": int((df[f] == 0).sum()),
        "zero_percent": float((df[f] == 0).mean() * 100)
    }

zero_df = pd.DataFrame(zero_stats).T

print(zero_df.round(3).to_string())

# ---------------------------------------------------------
# 4. Correlation analysis
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("4. FEATURE CORRELATION")
print("-" * 70)

corr = df[features].corr()

print(corr.round(3).to_string())

# ---------------------------------------------------------
# 5. Correlation of COS with other features
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("5. COS CORRELATION")
print("-" * 70)

print(
    corr["COS"]
    .sort_values(ascending=False)
    .round(4)
    .to_string()
)

# ---------------------------------------------------------
# 6. Relationship between ACS and OCS
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("6. ACS vs OCS")
print("-" * 70)

print("Correlation:", round(df["ACS"].corr(df["OCS"]), 4))

print("\nACS statistics:")
print(df["ACS"].describe().round(4).to_string())

print("\nOCS statistics:")
print(df["OCS"].describe().round(4).to_string())

# ---------------------------------------------------------
# 7. COS sparsity
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("7. COS DISTRIBUTION")
print("-" * 70)

print("COS > 0:", int((df["COS"] > 0).sum()))
print("COS = 0:", int((df["COS"] == 0).sum()))
print(
    "COS > 0 percentage:",
    round((df["COS"] > 0).mean() * 100, 3)
)

print("\nCOS quantiles:")
print(
    df["COS"]
    .quantile([0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1])
    .round(6)
    .to_string()
)

# ---------------------------------------------------------
# 8. PVS distribution
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("8. PVS DISTRIBUTION")
print("-" * 70)

print(
    df["PVS"]
    .quantile([0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1])
    .round(6)
    .to_string()
)

# ---------------------------------------------------------
# 9. KIS distribution
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("9. KIS DISTRIBUTION")
print("-" * 70)

print(
    df["KIS"]
    .quantile([0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1])
    .round(6)
    .to_string()
)

# ---------------------------------------------------------
# 10. Large allocation changes
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("10. ALLOCATION CHANGE")
print("-" * 70)

print("Total decisions:", len(df))

print(
    "ACS > 0:",
    int((df["ACS"] > 0).sum())
)

print(
    "ACS > 0.05:",
    int((df["ACS"] > 0.05).sum())
)

print(
    "ACS > 0.10:",
    int((df["ACS"] > 0.10).sum())
)

print(
    "ACS > 0.20:",
    int((df["ACS"] > 0.20).sum())
)

# ---------------------------------------------------------
# 11. Inspect strongest candidate conflict cases
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("11. TOP CANDIDATE CONFLICT CASES")
print("-" * 70)

candidate = df.sort_values(
    ["COS", "PVS", "ACS"],
    ascending=False
)

cols = [
    "decision_id",
    "action_a",
    "action_b",
    "ACS",
    "OCS",
    "ROS",
    "TOS",
    "KIS",
    "PVS",
    "COS",
    "delta_TOS_a",
    "delta_TOS_b"
]

print(
    candidate[cols]
    .head(20)
    .round(5)
    .to_string(index=False)
)

# ---------------------------------------------------------
# 12. Strong reallocation + adverse performance
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("12. STRONG REALLOCATION + PERFORMANCE DEGRADATION")
print("-" * 70)

strong = df[
    (df["ACS"] > 0.10) &
    (df["PVS"] > df["PVS"].median())
]

print("Number of cases:", len(strong))

if len(strong) > 0:
    print(
        strong[cols]
        .sort_values("COS", ascending=False)
        .head(20)
        .round(5)
        .to_string(index=False)
    )

# ---------------------------------------------------------
# 13. Save correlation matrix
# ---------------------------------------------------------

corr.to_csv("data/processed/feature_correlation.csv")

print("\nCorrelation matrix saved to:")
print("data/processed/feature_correlation.csv")

print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)