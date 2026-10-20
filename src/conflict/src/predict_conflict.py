import joblib
import pandas as pd


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
    print("O-RAN CONFLICT PREDICTION")
    print("==============================")

    # Load trained model
    print("\nLoading XGBoost model...")
    model = joblib.load(MODEL_FILE)

    print("Model loaded successfully.")

    # Get feature values from user
    print("\nEnter the conflict feature values.")
    print("Use values between 0 and 1.")

    values = []

    for feature in FEATURE_COLUMNS:
        value = float(input(f"{feature}: "))
        values.append(value)

    # Create input dataframe
    input_data = pd.DataFrame(
        [values],
        columns=FEATURE_COLUMNS
    )

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Get model-estimated probability
    probabilities = model.predict_proba(input_data)[0]

    no_conflict_probability = probabilities[0]
    conflict_probability = probabilities[1]

    print("\n==============================")
    print("PREDICTION RESULT")
    print("==============================")

    if prediction == 1:
        print("Prediction: CONFLICT")
    else:
        print("Prediction: NO CONFLICT")

    print(
        f"Model-estimated conflict probability: "
        f"{conflict_probability:.4f}"
    )

    print(
        f"Model-estimated no-conflict probability: "
        f"{no_conflict_probability:.4f}"
    )


if __name__ == "__main__":
    main()
