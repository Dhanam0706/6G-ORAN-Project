import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier


FEATURES = ["ROS", "TOS", "ACS", "OCS", "KIS"]

FEATURES_PATH = "data/processed/conflict_features_real.csv"
LABELS_PATH = "data/processed/conflict_labeled_real.csv"

OUTPUT_DIR = "data/plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# 1. Load data
# =========================================================

features = pd.read_csv(FEATURES_PATH)
labels = pd.read_csv(LABELS_PATH)

df = features.merge(
    labels[["decision_id", "conflict"]],
    on="decision_id",
    how="inner"
)

print("Merged dataset")
print("Shape:", df.shape)


# =========================================================
# 2. Prepare X and y
# =========================================================

X = df[FEATURES]
y = df["conflict"]


# =========================================================
# 3. Same stratified split as previous experiments
# =========================================================

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


# =========================================================
# 4. Class imbalance
# =========================================================

scale_pos_weight = (
    (y_train == 0).sum() /
    (y_train == 1).sum()
)

print(
    "scale_pos_weight:",
    round(scale_pos_weight, 4)
)


# =========================================================
# 5. Train identical XGBoost model
# =========================================================

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

print("\nModel trained.")


# =========================================================
# 6. SHAP explainer
# =========================================================

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)


# =========================================================
# 7. Global feature importance
# =========================================================

mean_abs_shap = np.abs(shap_values).mean(axis=0)

importance = pd.DataFrame({
    "feature": FEATURES,
    "mean_abs_shap": mean_abs_shap
})

importance = importance.sort_values(
    "mean_abs_shap",
    ascending=False
)

print("\n==============================")
print("GLOBAL SHAP IMPORTANCE")
print("==============================")

print(
    importance.to_string(index=False)
)


# =========================================================
# 8. Save importance table
# =========================================================

importance.to_csv(
    "data/processed/shap_feature_importance_real.csv",
    index=False
)


# =========================================================
# 9. SHAP summary plot
# =========================================================

plt.figure(figsize=(8, 6))

shap.summary_plot(
    shap_values,
    X_test,
    show=False
)

plt.tight_layout()

summary_path = os.path.join(
    OUTPUT_DIR,
    "shap_summary_real.png"
)

plt.savefig(
    summary_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 10. SHAP bar plot
# =========================================================

plt.figure(figsize=(8, 6))

shap.summary_plot(
    shap_values,
    X_test,
    plot_type="bar",
    show=False
)

plt.tight_layout()

bar_path = os.path.join(
    OUTPUT_DIR,
    "shap_bar_real.png"
)

plt.savefig(
    bar_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 11. Test-set probabilities
# =========================================================

probabilities = model.predict_proba(
    X_test
)[:, 1]

predictions = (
    probabilities >= 0.50
).astype(int)


# =========================================================
# 12. Select TP / FP / FN examples
# =========================================================

test_results = X_test.copy()

test_results["actual"] = y_test.values
test_results["probability"] = probabilities
test_results["prediction"] = predictions

tp = test_results[
    (test_results["actual"] == 1) &
    (test_results["prediction"] == 1)
]

fp = test_results[
    (test_results["actual"] == 0) &
    (test_results["prediction"] == 1)
]

fn = test_results[
    (test_results["actual"] == 1) &
    (test_results["prediction"] == 0)
]

print("\n==============================")
print("PREDICTION COUNTS")
print("==============================")

print("True positives:", len(tp))
print("False positives:", len(fp))
print("False negatives:", len(fn))


# =========================================================
# 13. Explain highest-probability cases
# =========================================================

top_indices = np.argsort(probabilities)[::-1][:10]

top_rows = X_test.iloc[top_indices].copy()

top_rows["actual"] = y_test.iloc[top_indices].values
top_rows["predicted_probability"] = probabilities[top_indices]

top_rows.to_csv(
    "data/processed/top_conflict_predictions_real.csv",
    index=False
)

print("\n==============================")
print("TOP 10 CONFLICT PREDICTIONS")
print("==============================")

print(
    top_rows.to_string(index=False)
)


# =========================================================
# 14. Final paths
# =========================================================

print("\n==============================")
print("SAVED SHAP OUTPUTS")
print("==============================")

print("data/processed/shap_feature_importance_real.csv")
print(summary_path)
print(bar_path)
print("data/processed/top_conflict_predictions_real.csv")