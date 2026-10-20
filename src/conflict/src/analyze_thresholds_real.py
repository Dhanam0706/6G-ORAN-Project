import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)
from xgboost import XGBClassifier


# ============================================================
# 1. LOAD DATA
# ============================================================

features_path = "data/processed/conflict_features_real.csv"
labels_path = "data/processed/conflict_labeled_real.csv"

features = pd.read_csv(features_path)
labels = pd.read_csv(labels_path)

df = pd.merge(
    features,
    labels[["decision_id", "conflict"]],
    on="decision_id",
    how="inner"
)

print("Merged dataset")
print("Shape:", df.shape)


# ============================================================
# 2. SELECT PREDICTIVE FEATURES
# ============================================================

feature_cols = [
    "ROS",
    "TOS",
    "ACS",
    "OCS",
    "KIS"
]

X = df[feature_cols]
y = df["conflict"]


# ============================================================
# 3. SAME STRATIFIED SPLIT AS PREVIOUS EXPERIMENT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

print("\n==============================")
print("STRATIFIED SPLIT")
print("==============================")

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting distribution:")
print(y_test.value_counts().sort_index())


# ============================================================
# 4. CLASS WEIGHT
# ============================================================

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count

print("\nscale_pos_weight:", round(scale_pos_weight, 4))


# ============================================================
# 5. TRAIN SAME XGBOOST MODEL
# ============================================================

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

print("\nTraining XGBoost...")
model.fit(X_train, y_train)
print("Training completed.")


# ============================================================
# 6. PREDICT CONFLICT PROBABILITIES
# ============================================================

y_prob = model.predict_proba(X_test)[:, 1]

print("\n==============================")
print("PROBABILITY SUMMARY")
print("==============================")

print("Minimum probability:", round(y_prob.min(), 4))
print("Maximum probability:", round(y_prob.max(), 4))
print("Mean probability:", round(y_prob.mean(), 4))
print("Median probability:", round(np.median(y_prob), 4))

print("\nROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))
print("PR-AUC:", round(average_precision_score(y_test, y_prob), 4))


# ============================================================
# 7. THRESHOLD ANALYSIS
# ============================================================

thresholds = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]

results = []

for threshold in thresholds:

    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    results.append({
        "threshold": threshold,
        "alerts": int(y_pred.sum()),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity
    })


results_df = pd.DataFrame(results)


# ============================================================
# 8. PRINT RESULTS
# ============================================================

print("\n==============================")
print("THRESHOLD ANALYSIS")
print("==============================")

print(
    results_df.to_string(
        index=False,
        formatters={
            "precision": "{:.4f}".format,
            "recall": "{:.4f}".format,
            "f1": "{:.4f}".format,
            "specificity": "{:.4f}".format
        }
    )
)


# ============================================================
# 9. SAVE RESULTS
# ============================================================

output_path = "data/processed/threshold_analysis_real.csv"

results_df.to_csv(
    output_path,
    index=False
)

print("\nSaved threshold analysis:")
print(output_path)