import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier


INPUT_FILE = "data/processed/conflict_features.csv"
MODEL_FILE = "models/conflict_xgboost.pkl"


def main():

    print("Loading feature dataset...")
    df = pd.read_csv(INPUT_FILE)

    print("Dataset shape:", df.shape)

    # Features used by the ML model
    feature_columns = [
    "ROS",
    "TOS",
    "ACS",
    "OCS",
    "KIS",
    "PVS",
    "COS"
]

    X = df[feature_columns]
    y = df["conflict"]

    print("\nFeatures:")
    print(feature_columns)

    print("\nTarget distribution:")
    print(y.value_counts())

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # Create XGBoost classifier
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )

    print("\nTraining XGBoost model...")

    model.fit(X_train, y_train)

    print("Training completed.")

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)

    print("\n==============================")
    print("MODEL RESULTS")
    print("==============================")

    print("\nAccuracy:", round(accuracy, 4))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, MODEL_FILE)

    print("\nModel saved to:")
    print(MODEL_FILE)


if __name__ == "__main__":
    main()
