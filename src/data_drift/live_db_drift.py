import os
import sys
import json
from pathlib import Path
from getpass import getpass

import psycopg2

# Allow import of the existing drift detector
sys.path.insert(0, str(Path(__file__).resolve().parent))

from drift_detection import detect_data_drift


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DB_HOST = "192.168.109.154"
DB_PORT = 5433
DB_NAME = "rankpi"
DB_USER = "rankpi"

# Number of observations used for each window
WINDOW_SIZE = 500

# Database slice mapping
SLICE_MAPPING = {
    "slice1": 1,
    "slice5": 5
}


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
# GET CURRENTLY DEPLOYED MODEL
# ============================================================

def get_deployed_model(conn):

    query = """
        SELECT
            rm.model_id,
            rm.name AS model_name,
            rmv.version_id,
            rmv.version,
            md.deployment_id,
            md.run_id
        FROM model_deployment md
        JOIN ran_model_version rmv
            ON md.version_id = rmv.version_id
        JOIN ran_model rm
            ON rmv.model_id = rm.model_id
        WHERE md.mode = 'deploy'
        ORDER BY md.started_at DESC
        LIMIT 1;
    """

    with conn.cursor() as cur:

        cur.execute(query)

        row = cur.fetchone()

    if row is None:
        raise RuntimeError("No deployed model found.")

    return {
        "model_id": row[0],
        "model_name": row[1],
        "version_id": row[2],
        "version": row[3],
        "deployment_id": row[4],
        "run_id": row[5]
    }


# ============================================================
# GET MODEL INPUT FEATURES
# ============================================================

def get_model_inputs(conn, version_id):

    query = """
        SELECT
            ordinal,
            name,
            source_expr
        FROM model_io
        WHERE version_id = %s
          AND direction = 'input'
        ORDER BY ordinal;
    """

    with conn.cursor() as cur:

        cur.execute(query, (version_id,))

        rows = cur.fetchall()

    if not rows:
        raise RuntimeError(
            f"No input features found for version_id={version_id}"
        )

    features = []

    for ordinal, name, source_expr in rows:

        # Example:
        # v_ue_rate.pusch_snr
        column_name = source_expr.split(".")[-1]

        # Example:
        # slice1.pusch_snr
        slice_name = name.split(".")[0]

        features.append({
            "ordinal": ordinal,
            "model_feature": name,
            "source_column": column_name,
            "slice_name": slice_name
        })

    return features


# ============================================================
# FETCH FEATURE DATA
# ============================================================

def fetch_feature_data(
    conn,
    run_id,
    slice_id,
    source_column
):

    # Only columns known to exist in v_ue_rate
    allowed_columns = {
        "dl_mbps",
        "d_dl_prb",
        "pusch_snr",
        "pucch_snr",
        "wb_cqi",
        "dl_mcs1",
        "ul_mcs1",
        "phr",
        "dl_bler",
        "ul_bler"
    }

    if source_column not in allowed_columns:

        raise ValueError(
            f"Unsupported database column: {source_column}"
        )

    query = f"""
        SELECT ts, {source_column}
        FROM v_ue_rate
        WHERE run_id = %s
          AND slice_id = %s
          AND {source_column} IS NOT NULL
        ORDER BY ts DESC
        LIMIT %s;
    """

    with conn.cursor() as cur:

        cur.execute(
            query,
            (
                run_id,
                slice_id,
                WINDOW_SIZE * 2
            )
        )

        rows = cur.fetchall()

    if len(rows) < WINDOW_SIZE * 2:

        raise RuntimeError(
            f"Not enough data for {source_column} "
            f"(slice_id={slice_id}). "
            f"Found {len(rows)}, need {WINDOW_SIZE * 2}."
        )

    # Database returns newest → oldest.
    # Reverse so windows are chronological.
    rows.reverse()

    values = [
        float(row[1])
        for row in rows
    ]

    reference_window = values[:WINDOW_SIZE]

    current_window = values[
        WINDOW_SIZE:WINDOW_SIZE * 2
    ]

    return reference_window, current_window


# ============================================================
# RUN DRIFT DETECTION
# ============================================================

def run_feature_drift_detection(
    conn,
    run_id,
    feature
):

    slice_name = feature["slice_name"]

    source_column = feature["source_column"]

    slice_id = SLICE_MAPPING.get(slice_name)

    if slice_id is None:

        raise RuntimeError(
            f"No slice ID mapping found for {slice_name}"
        )

    reference_window, current_window = fetch_feature_data(
        conn,
        run_id,
        slice_id,
        source_column
    )

    result = detect_data_drift(
        kpi_column=source_column,
        reference_data=reference_window,
        current_data=current_window,
        make_plots=False
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    conn = None

    try:

        # ----------------------------------------------------
        # 1. Connect to database
        # ----------------------------------------------------

        conn = connect_to_database()

        # ----------------------------------------------------
        # 2. Find currently deployed model
        # ----------------------------------------------------

        model = get_deployed_model(conn)

        print("\n========================================")
        print("DEPLOYED MODEL")
        print("========================================")

        print(
            f"Model ID      : "
            f"{model['model_id']}"
        )

        print(
            f"Model Name    : "
            f"{model['model_name']}"
        )

        print(
            f"Version ID    : "
            f"{model['version_id']}"
        )

        print(
            f"Version       : "
            f"{model['version']}"
        )

        print(
            f"Deployment ID : "
            f"{model['deployment_id']}"
        )

        print(
            f"Run ID        : "
            f"{model['run_id']}"
        )

        # ----------------------------------------------------
        # 3. Get model input features
        # ----------------------------------------------------

        features = get_model_inputs(
            conn,
            model["version_id"]
        )

        print("\n========================================")
        print("MODEL INPUT FEATURES")
        print("========================================")

        for feature in features:

            print(
                f"{feature['ordinal']:2d}. "
                f"{feature['model_feature']}"
                f"  ->  "
                f"{feature['source_column']}"
            )

        # ----------------------------------------------------
        # 4. Run drift detection
        # ----------------------------------------------------

        print("\n========================================")
        print("DATA DRIFT ANALYSIS")
        print("========================================")

        results = []

        for feature in features:

            print(
                f"\nChecking: "
                f"{feature['model_feature']}"
            )

            try:

                result = run_feature_drift_detection(
                    conn,
                    model["run_id"],
                    feature
                )

                # Store complete detector result
                # for JSON generation.
                results.append({
                    "feature": feature["model_feature"],
                    "kpi": feature["source_column"],
                    "status": result.get("status"),
                    "drift_score": result.get("drift_score"),
                    "reason": result.get("reason"),

                    "ks_statistic": result.get(
                        "ks_statistic"
                    ),

                    "ks_p_value": result.get(
                        "ks_p_value"
                    ),

                    "wasserstein": result.get(
                        "wasserstein"
                    ),

                    "psi": result.get(
                        "psi"
                    )
                })

                print(
                    f"  Status      : "
                    f"{result.get('status')}"
                )

                print(
                    f"  Drift Score : "
                    f"{result.get('drift_score')}"
                )

            except Exception as e:

                print(
                    f"  ERROR: {e}"
                )

        # ----------------------------------------------------
        # 5. Count model-level results
        # ----------------------------------------------------

        no_drift_count = sum(
            1
            for r in results
            if r["status"] == "NO_DRIFT"
        )

        warning_count = sum(
            1
            for r in results
            if r["status"] == "WARNING"
        )

        drift_count = sum(
            1
            for r in results
            if r["status"] == "DRIFT_DETECTED"
        )

        # ----------------------------------------------------
        # 6. Determine overall model status
        # ----------------------------------------------------

        if drift_count > 0:

            overall_status = "DRIFT_DETECTED"

        elif warning_count > 0:

            overall_status = "WARNING"

        else:

            overall_status = "NO_DRIFT"

        # ----------------------------------------------------
        # 7. JSON report
        # ----------------------------------------------------

        json_report = {

            "model": {

                "model_id": model["model_id"],

                "model_name": model["model_name"],

                "version_id": model["version_id"],

                "version": model["version"],

                "deployment_id": model["deployment_id"],

                "run_id": model["run_id"]
            },

            "data_drift": {

                "overall_status": overall_status,

                "features_analyzed": len(results),

                "summary": {

                    "no_drift": no_drift_count,

                    "warning": warning_count,

                    "drift_detected": drift_count
                },

                "features": results
            }
        }

        # ----------------------------------------------------
        # 8. Print model-level report
        # ----------------------------------------------------

        print("\n========================================")
        print("MODEL-LEVEL DRIFT REPORT")
        print("========================================")

        print(
            f"Model: "
            f"{model['model_id']} "
            f"({model['model_name']})"
        )

        print(
            f"Version: "
            f"{model['version']}"
        )

        print("\nFeature Results:")

        for item in results:

            print(
                f"  {item['feature']:<28} "
                f"{item['status']:<16} "
                f"score={item['drift_score']}"
            )

        print("\n----------------------------------------")

        print(
            f"Features analysed : "
            f"{len(results)}"
        )

        print(
            f"NO_DRIFT          : "
            f"{no_drift_count}"
        )

        print(
            f"WARNING           : "
            f"{warning_count}"
        )

        print(
            f"DRIFT_DETECTED    : "
            f"{drift_count}"
        )

        print(
            f"Overall status    : "
            f"{overall_status}"
        )

        # ----------------------------------------------------
        # 9. Print JSON
        # ----------------------------------------------------

        print("\n========================================")
        print("JSON DRIFT REPORT")
        print("========================================")

        print(
            json.dumps(
                json_report,
                indent=2
            )
        )

        # ----------------------------------------------------
        # 10. Save JSON file
        # ----------------------------------------------------

        output_file = "data_drift_report.json"

        with open(
            output_file,
            "w"
        ) as f:

            json.dump(
                json_report,
                f,
                indent=2
            )

        print(
            f"\nJSON report saved to: "
            f"{output_file}"
        )

        print("========================================")

    except Exception as e:

        print("\nERROR:")
        print(e)

    finally:

        if conn is not None:

            conn.close()

            print(
                "\nDatabase connection closed."
            )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()