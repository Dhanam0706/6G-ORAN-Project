import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)
import joblib


# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

FEATURE_FILE = "data/processed/conflict_features_real.csv"
LABEL_FILE = "data/processed/conflict_labeled_real.csv"
MODEL_FILE = "models/conflict_xgboost_real.pkl"

features = pd.read_csv(FEATURE_FILE)
labels = pd.read_csv(LABEL_FILE)


# --------------------------------------------------
# 2. Merge features and labels
# --------------------------------------------------

df = features.merge(
    labels[["decision_id", "conflict"]],
    on="decision_id",
    how="inner"
)

# Make sure decisions are chronological
df["action_time"] = pd.to_datetime(
    df["action_time"],
    format="mixed",
    utc=True
)

df = df.sort_values("action_time").reset_index(drop=True)


print("\nMerged dataset")
print("Shape:", df.shape)

print("\nLabel distribution:")
print(df["conflict"].value_counts())

print("\nConflict rate:")
print(
    round(df["conflict"].mean() * 100, 2),
    "%"
)


# --------------------------------------------------
# 3. Select predictive features
# --------------------------------------------------

MODEL_FEATURES = [
    "ROS",
    "TOS",
    "ACS",
    "OCS",
    "KIS"
]

X = df[MODEL_FEATURES]
y = df["conflict"]


print("\nModel features:")
print(MODEL_FEATURES)

print("\nMissing values:")
print(X.isna().sum())


# --------------------------------------------------
# 4. Chronological train/test split
# --------------------------------------------------

split_index = int(len(df) * 0.70)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print("\nChronological split")
print("-------------------")

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining label distribution:")
print(y_train.value_counts())

print("\nTesting label distribution:")
print(y_test.value_counts())


# --------------------------------------------------
# 5. Handle class imbalance
# --------------------------------------------------

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count

print("\nClass imbalance")
print("----------------")
print("Negative samples:", negative_count)
print("Positive samples:", positive_count)
print("scale_pos_weight:", round(scale_pos_weight, 4))


# --------------------------------------------------
# 6. Train XGBoost
# --------------------------------------------------

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

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# --------------------------------------------------
# 7. Predictions
# --------------------------------------------------

y_probability = model.predict_proba(X_test)[:, 1]

y_pred = (y_probability >= 0.50).astype(int)


# --------------------------------------------------
# 8. Evaluation
# --------------------------------------------------

print("\n==============================")
print("REAL-DATA XGBOOST RESULTS")
print("==============================")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        digits=4,
        zero_division=0
    )
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

pr_auc = average_precision_score(
    y_test,
    y_probability
)

print("\nROC-AUC:", round(roc_auc, 4))
print("PR-AUC:", round(pr_auc, 4))


# --------------------------------------------------
# 9. Feature importance
# --------------------------------------------------

importance = pd.DataFrame({
    "feature": MODEL_FEATURES,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\nFeature importance:")
print(
    importance.to_string(index=False)
)


# --------------------------------------------------
# 10. Save model
# --------------------------------------------------

joblib.dump(
    model,
    MODEL_FILE
)

print("\nModel saved:")
print(MODEL_FILE)