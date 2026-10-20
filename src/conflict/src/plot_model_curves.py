import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score
)

from xgboost import XGBClassifier


FEATURES = ["ROS", "TOS", "ACS", "OCS", "KIS"]

FEATURES_PATH = "data/processed/conflict_features_real.csv"
LABELS_PATH = "data/processed/conflict_labeled_real.csv"

OUTPUT_DIR = "data/plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# 1. Load real-data features and labels
# ---------------------------------------------------------

features = pd.read_csv(FEATURES_PATH)
labels = pd.read_csv(LABELS_PATH)

df = features.merge(
    labels[["decision_id", "conflict"]],
    on="decision_id",
    how="inner"
)

print("Merged dataset")
print("Shape:", df.shape)


# ---------------------------------------------------------
# 2. Select predictive features
# ---------------------------------------------------------

X = df[FEATURES]
y = df["conflict"]


# ---------------------------------------------------------
# 3. Reproduce the same stratified 70:30 split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\n==============================")
print("STRATIFIED SPLIT")
print("==============================")

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTesting distribution:")
print(y_test.value_counts().sort_index())


# ---------------------------------------------------------
# 4. Class imbalance handling
# ---------------------------------------------------------

scale_pos_weight = (
    (y_train == 0).sum() /
    (y_train == 1).sum()
)

print(
    "\nscale_pos_weight:",
    round(scale_pos_weight, 4)
)


# ---------------------------------------------------------
# 5. Train same XGBoost configuration
# ---------------------------------------------------------

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

model.fit(X_train, y_train)


# ---------------------------------------------------------
# 6. Generate conflict probabilities
# ---------------------------------------------------------

probabilities = model.predict_proba(X_test)[:, 1]


# ---------------------------------------------------------
# 7. Calculate ROC-AUC and PR-AUC
# ---------------------------------------------------------

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

pr_auc = average_precision_score(
    y_test,
    probabilities
)

print("\n==============================")
print("MODEL CURVES")
print("==============================")

print("ROC-AUC:", round(roc_auc, 4))
print("PR-AUC:", round(pr_auc, 4))


# ---------------------------------------------------------
# 8. ROC curve
# ---------------------------------------------------------

fpr, tpr, roc_thresholds = roc_curve(
    y_test,
    probabilities
)

plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    label=f"XGBoost (ROC-AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve - Stratified Real-Data Evaluation"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

roc_path = os.path.join(
    OUTPUT_DIR,
    "roc_curve_real_stratified.png"
)

plt.savefig(
    roc_path,
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# 9. Precision-Recall curve
# ---------------------------------------------------------

precision, recall, pr_thresholds = precision_recall_curve(
    y_test,
    probabilities
)

prevalence = y_test.mean()

plt.figure(figsize=(7, 6))

plt.plot(
    recall,
    precision,
    label=f"XGBoost (PR-AUC = {pr_auc:.4f})"
)

plt.axhline(
    prevalence,
    linestyle="--",
    label=f"Positive prevalence = {prevalence:.4f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title(
    "Precision-Recall Curve - Stratified Real-Data Evaluation"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

pr_path = os.path.join(
    OUTPUT_DIR,
    "precision_recall_curve_real_stratified.png"
)

plt.savefig(
    pr_path,
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# 10. Final output
# ---------------------------------------------------------

print("\n==============================")
print("SAVED PLOTS")
print("==============================")

print(roc_path)
print(pr_path)