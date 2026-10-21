

import os
import sys
from pathlib import Path
from getpass import getpass
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import psycopg2

# Allow import of existing detector
sys.path.insert(0, str(Path(__file__).resolve().parent))

from drift_detection import detect_data_drift


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DB_HOST = "192.168.109.154"
DB_PORT = 5433
DB_NAME = "rankpi"
DB_USER = "rankpi"

MODEL_ID = 9
VERSION_ID = 1
RUN_ID = 3

FEATURE_NAME = "slice5.dl_bler"
SOURCE_COLUMN = "dl_bler"
SLICE_ID = 5

WINDOW_SIZE = 500

# Number of normal and drift experiments
NORMAL_CASES = 20
DRIFT_CASES_PER_LEVEL = 10

# Smaller controlled changes
DRIFT_LEVELS = [0.02, 0.05, 0.10, 0.15, 0.20]

# Thresholds to evaluate
THRESHOLDS = np.arange(0.10, 0.81, 0.05)

MAX_ACCEPTABLE_FPR = 0.05

RANDOM_SEED = 42


# ============================================================
# DATABASE CONNECTION
# ============================================================

def connect_to_database():

    password = os.getenv("DB_PASSWORD")

    if not password:
        password = getpass("Enter PostgreSQL password: ")

    print("\nConnecting to PostgreSQL...")

    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=password
    )

    print("Database connection successful!")

    return conn


# ============================================================
# FETCH CHRONOLOGICAL DATA
# ============================================================

def fetch_real_data(conn):

    query = """
        SELECT ts, dl_bler
        FROM v_ue_rate
        WHERE run_id = %s
          AND slice_id = %s
          AND dl_bler IS NOT NULL
        ORDER BY ts ASC;
    """

    with conn.cursor() as cur:

        cur.execute(
            query,
            (RUN_ID, SLICE_ID)
        )

        rows = cur.fetchall()

    if len(rows) < WINDOW_SIZE * 2:

        raise RuntimeError(
            f"Not enough data. Found {len(rows)} samples."
        )

    timestamps = np.array(
        [row[0] for row in rows]
    )

    values = np.array(
        [float(row[1]) for row in rows],
        dtype=float
    )

    return timestamps, values


# ============================================================
# CREATE NORMAL CHRONOLOGICAL CASES
# ============================================================

def create_normal_case(
    values,
    case_number
):

    # Use consecutive windows.
    #
    # Example:
    #
    # reference = samples 0:500
    # current   = samples 500:1000
    #
    # Next case:
    #
    # reference = samples 1000:1500
    # current   = samples 1500:2000

    start = (
        case_number * WINDOW_SIZE * 2
    )

    end = start + WINDOW_SIZE * 2

    if end > len(values):

        return None

    reference = values[
        start:start + WINDOW_SIZE
    ].copy()

    current = values[
        start + WINDOW_SIZE:end
    ].copy()

    return reference, current


# ============================================================
# CREATE CONTROLLED DRIFT CASE
# ============================================================

def create_drift_case(
    values,
    case_number,
    drift_level
):

    # Use chronological adjacent windows.
    normal_case = create_normal_case(
        values,
        case_number
    )

    if normal_case is None:
        return None

    reference, current = normal_case

    # Apply a controlled multiplicative change.
    #
    # Example:
    # drift_level = 0.10
    # current = current * 1.10
    #
    # This is only a synthetic calibration scenario.

    current = current * (
        1.0 + drift_level
    )

    return reference, current


# ============================================================
# RUN EXISTING DETECTOR QUIETLY
# ============================================================

def calculate_score(
    reference,
    current
):

    # The detector prints its own table.
    # During calibration we suppress that output.

    silent_output = StringIO()

    with redirect_stdout(silent_output):

        result = detect_data_drift(
            kpi_column=SOURCE_COLUMN,
            reference_data=reference,
            current_data=current,
            make_plots=False
        )

    return float(
        result["drift_score"]
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    scores,
    labels,
    threshold
):

    predictions = [
        1 if score >= threshold else 0
        for score in scores
    ]

    tp = sum(
        1
        for actual, predicted
        in zip(labels, predictions)
        if actual == 1 and predicted == 1
    )

    tn = sum(
        1
        for actual, predicted
        in zip(labels, predictions)
        if actual == 0 and predicted == 0
    )

    fp = sum(
        1
        for actual, predicted
        in zip(labels, predictions)
        if actual == 0 and predicted == 1
    )

    fn = sum(
        1
        for actual, predicted
        in zip(labels, predictions)
        if actual == 1 and predicted == 0
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    fpr = (
        fp / (fp + tn)
        if fp + tn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "threshold": threshold,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "f1": f1
    }


# ============================================================
# MAIN
# ============================================================

def main():

    conn = None

    try:

        # ----------------------------------------------------
        # 1. Connect
        # ----------------------------------------------------

        conn = connect_to_database()

        # ----------------------------------------------------
        # 2. Get chronological real data
        # ----------------------------------------------------

        timestamps, values = fetch_real_data(
            conn
        )

        print("\n========================================")
        print("CALIBRATION DATA")
        print("========================================")

        print(f"Model ID       : {MODEL_ID}")
        print(f"Version ID     : {VERSION_ID}")
        print(f"Run ID         : {RUN_ID}")
        print(f"Feature        : {FEATURE_NAME}")
        print(f"Database column: {SOURCE_COLUMN}")
        print(f"Slice ID       : {SLICE_ID}")
        print(f"Real samples   : {len(values)}")

        print(
            f"Time range     : "
            f"{timestamps[0]} -> {timestamps[-1]}"
        )

        # ----------------------------------------------------
        # 3. Generate normal cases
        # ----------------------------------------------------

        scores = []
        labels = []

        print("\nGenerating chronological normal cases...")

        normal_created = 0

        for case_number in range(
            NORMAL_CASES
        ):

            case = create_normal_case(
                values,
                case_number
            )

            if case is None:
                break

            reference, current = case

            score = calculate_score(
                reference,
                current
            )

            scores.append(score)
            labels.append(0)

            normal_created += 1

        # ----------------------------------------------------
        # 4. Generate controlled drift cases
        # ----------------------------------------------------

        print(
            "Generating controlled drift cases..."
        )

        drift_created = 0

        # Start after the normal cases so the
        # calibration datasets don't overlap.

        drift_start_case = NORMAL_CASES

        for level_index, drift_level in enumerate(
            DRIFT_LEVELS
        ):

            for repetition in range(
                DRIFT_CASES_PER_LEVEL
            ):

                case_number = (
                    drift_start_case
                    + level_index
                    * DRIFT_CASES_PER_LEVEL
                    + repetition
                )

                case = create_drift_case(
                    values,
                    case_number,
                    drift_level
                )

                if case is None:
                    break

                reference, current = case

                score = calculate_score(
                    reference,
                    current
                )

                scores.append(score)
                labels.append(1)

                drift_created += 1

        # ----------------------------------------------------
        # 5. Score distributions
        # ----------------------------------------------------

        normal_scores = [
            score
            for score, label
            in zip(scores, labels)
            if label == 0
        ]

        drift_scores = [
            score
            for score, label
            in zip(scores, labels)
            if label == 1
        ]

        print("\n========================================")
        print("SCORE DISTRIBUTION")
        print("========================================")

        print(
            f"Normal cases : "
            f"{len(normal_scores)}"
        )

        print(
            f"Drift cases  : "
            f"{len(drift_scores)}"
        )

        if normal_scores:

            print(
                f"Normal min   : "
                f"{min(normal_scores):.4f}"
            )

            print(
                f"Normal mean  : "
                f"{np.mean(normal_scores):.4f}"
            )

            print(
                f"Normal max   : "
                f"{max(normal_scores):.4f}"
            )

        if drift_scores:

            print(
                f"Drift min    : "
                f"{min(drift_scores):.4f}"
            )

            print(
                f"Drift mean   : "
                f"{np.mean(drift_scores):.4f}"
            )

            print(
                f"Drift max    : "
                f"{max(drift_scores):.4f}"
            )

        # ----------------------------------------------------
        # 6. Threshold evaluation
        # ----------------------------------------------------

        results = []

        for threshold in THRESHOLDS:

            result = calculate_metrics(
                scores,
                labels,
                float(threshold)
            )

            results.append(result)

        print("\n========================================")
        print("THRESHOLD CALIBRATION")
        print("========================================")

        print(
            "\nThreshold | Precision | Recall | "
            "F1     | FPR"
        )

        print(
            "----------+-----------+--------+"
            "--------+--------"
        )

        for result in results:

            print(
                f"{result['threshold']:9.2f} | "
                f"{result['precision']:9.3f} | "
                f"{result['recall']:6.3f} | "
                f"{result['f1']:6.3f} | "
                f"{result['fpr']:6.3f}"
            )

        # ----------------------------------------------------
        # 7. Select candidate threshold
        # ----------------------------------------------------

        valid_results = [
            result
            for result in results
            if result["fpr"] <= MAX_ACCEPTABLE_FPR
        ]

        print("\n========================================")
        print("CALIBRATION RESULT")
        print("========================================")

        print(
            "Current threshold : 0.60"
        )

        if valid_results:

            best = max(
                valid_results,
                key=lambda result: result["f1"]
            )

            print(
                f"Calibrated threshold: "
                f"{best['threshold']:.2f}"
            )

            print(
                f"Precision         : "
                f"{best['precision']:.3f}"
            )

            print(
                f"Recall            : "
                f"{best['recall']:.3f}"
            )

            print(
                f"F1                : "
                f"{best['f1']:.3f}"
            )

            print(
                f"FPR               : "
                f"{best['fpr']:.3f}"
            )

            print(
                f"TP={best['tp']} "
                f"TN={best['tn']} "
                f"FP={best['fp']} "
                f"FN={best['fn']}"
            )

        else:

            print(
                "No tested threshold achieved "
                "FPR <= 0.05."
            )

            print(
                "Do NOT change the current "
                "0.60 threshold based on this "
                "calibration experiment."
            )

        # ----------------------------------------------------
        # 8. Interpretation
        # ----------------------------------------------------

        print("\n========================================")
        print("INTERPRETATION")
        print("========================================")

        print(
            "Normal cases use consecutive "
            "chronological windows."
        )

        print(
            "Drift cases use the same "
            "chronological structure with "
            "controlled synthetic shifts."
        )

        print(
            "These controlled shifts are "
            "calibration scenarios, not claims "
            "about actual RAN behaviour."
        )

        print(
            "The threshold must be validated "
            "with additional features and "
            "independent test data before being "
            "used as a final project threshold."
        )

    except Exception as e:

        print("\nERROR:")
        print(e)

    finally:

        if conn is not None:

            conn.close()

            print(
                "\nDatabase connection closed."
            )


if __name__ == "__main__":
    main()