import joblib
import pandas as pd
import shap


MODEL_FILE = "models/conflict_xgboost.pkl"

FEATURE_COLUMNS = [
    "ROS",
    "TOS",
    "ACS",
    "OCS",
    "KIS",
    "PVS",
    "COS"
]


def main():

    print("==============================")
    print("O-RAN CONFLICT EXPLANATION")
    print("==============================")

    # Load trained model
    print("\nLoading XGBoost model...")
    model = joblib.load(MODEL_FILE)
    print("Model loaded successfully.")

    # Example conflict scenario
    values = [
        0.9,  # ROS
        0.8,  # TOS
        0.9,  # ACS
        0.7,  # OCS
        0.8,  # KIS
        0.6,  # PVS
        0.9   # COS
    ]

    input_data = pd.DataFrame(
        [values],
        columns=FEATURE_COLUMNS
    )

    # Prediction
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    print("\n==============================")
    print("PREDICTION")
    print("==============================")

    if prediction == 1:
        print("Prediction: CONFLICT")
    else:
        print("Prediction: NO CONFLICT")

    print(f"Conflict probability: {probabilities[1]:.4f}")

    # Create SHAP explainer
    print("\nCalculating SHAP explanations...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    # For binary classification, take the contribution for the
    # predicted conflict class
    if len(shap_values.shape) == 2:
        contributions = shap_values[0]
    else:
        contributions = shap_values

    print("\n==============================")
    print("SHAP FEATURE CONTRIBUTIONS")
    print("==============================")

    explanation = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Value": values,
        "SHAP Contribution": contributions
    })

    explanation["Absolute Contribution"] = (
        explanation["SHAP Contribution"].abs()
    )

    explanation = explanation.sort_values(
        "Absolute Contribution",
        ascending=False
    )

    print(
        explanation[
            ["Feature", "Value", "SHAP Contribution"]
        ].to_string(index=False)
    )

    print("\n==============================")
    print("INTERPRETATION")
    print("==============================")

    positive_features = explanation[
        explanation["SHAP Contribution"] > 0
    ]

    negative_features = explanation[
        explanation["SHAP Contribution"] < 0
    ]

    if len(positive_features) > 0:
        print("\nFeatures contributing toward CONFLICT:")

        for _, row in positive_features.iterrows():
            print(
                f"  {row['Feature']}: "
                f"{row['SHAP Contribution']:.4f}"
            )

    if len(negative_features) > 0:
        print("\nFeatures contributing away from CONFLICT:")

        for _, row in negative_features.iterrows():
            print(
                f"  {row['Feature']}: "
                f"{row['SHAP Contribution']:.4f}"
            )


if __name__ == "__main__":
    main()
