from typing import TypedDict

import psycopg2
import numpy as np

from river.drift import ADWIN
from langgraph.graph import StateGraph, START, END


# ============================================================
# STATE
# ============================================================

class DriftState(TypedDict):
    model_id: int
    degraded_kpis: list[str]

    version_id: int | None
    version: str | None
    deployment_id: int | None
    run_id: int | None
    cell_id: int | None

    model_inputs: list[dict]

    telemetry: list[dict]

    model_vector: list[dict]

    reference_mean: list[float]
    reference_covariance: list[list[float]]
    active_dimensions: list[int]

    mahalanobis_distance: float | None

    drift_scores: list[dict]
    drift_detections: list[dict]


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db_connection():

    return psycopg2.connect(
        host="192.168.109.154",
        port=5433,
        dbname="rankpi",
        user="rankpi"
    )


# ============================================================
# NODE 1
# RECEIVE REQUEST
# ============================================================

def receive_request(state: DriftState):

    print("Received model_id:", state["model_id"])
    print("Degraded KPI(s):", state["degraded_kpis"])

    return state


# ============================================================
# NODE 2
# RESOLVE MODEL
# ============================================================

def resolve_model(state: DriftState):

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            md.deployment_id,
            rmv.version_id,
            rmv.version,
            md.run_id,
            md.cell_id
        FROM model_deployment md
        JOIN ran_model_version rmv
            ON md.version_id = rmv.version_id
        WHERE rmv.model_id = %s
        ORDER BY md.started_at DESC
        LIMIT 1
    """, (state["model_id"],))

    row = cur.fetchone()

    cur.close()
    conn.close()

    if row is None:

        raise ValueError(
            f"No deployment found for model "
            f"{state['model_id']}"
        )

    (
        deployment_id,
        version_id,
        version,
        run_id,
        cell_id
    ) = row

    print("Resolved deployment:", deployment_id)
    print("Resolved version_id:", version_id)
    print("Resolved version:", version)
    print("Resolved run_id:", run_id)
    print("Resolved cell_id:", cell_id)

    return {
        "deployment_id": deployment_id,
        "version_id": version_id,
        "version": version,
        "run_id": run_id,
        "cell_id": cell_id
    }


# ============================================================
# NODE 3
# RESOLVE MODEL INPUTS
# ============================================================

def resolve_inputs(state: DriftState):

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            ordinal,
            name,
            source_expr,
            scope_type,
            transform,
            scale,
            offset_
        FROM model_io
        WHERE version_id = %s
          AND direction = 'input'
        ORDER BY ordinal
    """, (state["version_id"],))

    rows = cur.fetchall()

    cur.close()
    conn.close()

    if not rows:

        raise ValueError(
            f"No model inputs found for version "
            f"{state['version_id']}"
        )

    inputs = []

    for row in rows:

        (
            ordinal,
            name,
            source_expr,
            scope_type,
            transform,
            scale,
            offset
        ) = row

        inputs.append({
            "ordinal": ordinal,
            "name": name,
            "source_expr": source_expr,
            "scope_type": str(scope_type),
            "transform": transform,
            "scale": scale,
            "offset": offset
        })

    print("\nNumber of model inputs:", len(inputs))

    for item in inputs:

        print(
            f"{item['ordinal']}: "
            f"{item['name']} <- {item['source_expr']} "
            f"| transform={item['transform']} "
            f"| scale={item['scale']}"
        )

    return {
        "model_inputs": inputs
    }


# ============================================================
# NODE 4
# FETCH COMPLETE TELEMETRY
# ============================================================

def fetch_live_data(state: DriftState):

    conn = get_db_connection()
    cur = conn.cursor()

    # --------------------------------------------------------
    # Discover required telemetry columns dynamically
    # --------------------------------------------------------

    source_columns = []

    for item in state["model_inputs"]:

        column = (
            item["source_expr"]
            .split(".")[-1]
        )

        if column not in source_columns:

            source_columns.append(column)

    if not source_columns:

        raise ValueError(
            "No telemetry columns found"
        )

    # --------------------------------------------------------
    # Build SELECT list
    # --------------------------------------------------------

    select_columns = ", ".join(
        f'"{column}"'
        for column in source_columns
    )

    # --------------------------------------------------------
    # Fetch all telemetry for the resolved run
    #
    # We need the complete run now because:
    #
    # 80% = reference period
    # 20% = streaming period
    #
    # --------------------------------------------------------

    query = f"""
        SELECT
            ts,
            ue_id,
            slice_id,
            {select_columns}
        FROM v_ue_rate
        WHERE run_id = %s
          AND cell_id = %s
        ORDER BY
            ts,
            slice_id,
            ue_id
    """

    cur.execute(
        query,
        (
            state["run_id"],
            state["cell_id"]
        )
    )

    rows = cur.fetchall()

    column_names = [
        description[0]
        for description in cur.description
    ]

    cur.close()
    conn.close()

    if not rows:

        raise ValueError(
            f"No telemetry found for run "
            f"{state['run_id']}"
        )

    # --------------------------------------------------------
    # Convert rows into dictionaries
    # --------------------------------------------------------

    telemetry = [
        dict(zip(column_names, row))
        for row in rows
    ]

    print(
        "\nFetched telemetry rows:",
        len(telemetry)
    )

    print(
        "Telemetry columns:",
        column_names
    )

    return {
        "telemetry": telemetry
    }


# ============================================================
# HELPER
# BUILD ONE MODEL VECTOR FROM ONE SECOND
# ============================================================

def build_vector_from_rows(
    rows_at_time: list[dict],
    model_inputs: list[dict]
):

    vector = []

    for item in model_inputs:

        name = item["name"]

        column = (
            item["source_expr"]
            .split(".")[-1]
        )

        # ----------------------------------------------------
        # Determine slice
        # ----------------------------------------------------

        slice_part = name.split(".")[0]

        if not slice_part.startswith("slice"):

            raise ValueError(
                f"Cannot determine slice "
                f"from {name}"
            )

        slice_id = int(
            slice_part.replace(
                "slice",
                ""
            )
        )

        # ----------------------------------------------------
        # Collect values for this slice
        # ----------------------------------------------------

        values = []

        for row in rows_at_time:

            if row["slice_id"] == slice_id:

                value = row[column]

                if value is not None:

                    values.append(
                        float(value)
                    )

        if not values:

            return None

        # ----------------------------------------------------
        # Apply transform
        # ----------------------------------------------------

        transform = item["transform"]

        if transform == "mean-over-1s":

            value = (
                sum(values)
                / len(values)
            )

        else:

            raise ValueError(
                f"Unsupported transform: "
                f"{transform}"
            )

        # ----------------------------------------------------
        # Apply scale
        # ----------------------------------------------------

        scale = item["scale"]

        if scale is not None:

            value = (
                value / float(scale)
            )

        # ----------------------------------------------------
        # Apply offset
        # ----------------------------------------------------

        offset = item["offset"]

        if offset is not None:

            value = (
                value + float(offset)
            )

        vector.append(value)

    return vector


# ============================================================
# NODE 5
# BUILD CURRENT MODEL VECTOR
# ============================================================

def build_model_vector(state: DriftState):

    telemetry = state["telemetry"]
    model_inputs = state["model_inputs"]

    # --------------------------------------------------------
    # Find latest timestamp
    # --------------------------------------------------------

    latest_ts = max(
        row["ts"]
        for row in telemetry
    )

    latest_second = latest_ts.replace(
        microsecond=0
    )

    # --------------------------------------------------------
    # Select latest 1-second window
    # --------------------------------------------------------

    latest_rows = [

        row

        for row in telemetry

        if row["ts"].replace(
            microsecond=0
        ) == latest_second
    ]

    vector = build_vector_from_rows(
        latest_rows,
        model_inputs
    )

    if vector is None:

        raise ValueError(
            "Unable to build current "
            "model vector"
        )

    model_vector = []

    for item, value in zip(
        model_inputs,
        vector
    ):

        model_vector.append({
            "ordinal": item["ordinal"],
            "name": item["name"],
            "value": value
        })

    print("\nModel input vector:")

    for item in model_vector:

        print(
            f"{item['ordinal']:2d} "
            f"{item['name']:20s} "
            f"= {item['value']}"
        )

    print(
        "\nModel vector dimension:",
        len(model_vector)
    )

    return {
        "model_vector": model_vector
    }


# ============================================================
# NODE 6
# BUILD REFERENCE DISTRIBUTION
# ============================================================

def build_reference_distribution(
    state: DriftState
):

    telemetry = state["telemetry"]
    model_inputs = state["model_inputs"]

    # --------------------------------------------------------
    # Group all telemetry into 1-second windows
    # --------------------------------------------------------

    timestamps = {}

    for row in telemetry:

        ts = row["ts"].replace(
            microsecond=0
        )

        if ts not in timestamps:

            timestamps[ts] = []

        timestamps[ts].append(row)

    # --------------------------------------------------------
    # Sort timestamps
    # --------------------------------------------------------

    sorted_timestamps = sorted(
        timestamps.keys()
    )

    print(
        "\nHistorical second-level windows:",
        len(sorted_timestamps)
    )

    # --------------------------------------------------------
    # First 80% = reference
    # Remaining 20% = stream
    # --------------------------------------------------------

    split_index = int(
        len(sorted_timestamps) * 0.80
    )

    reference_timestamps = (
        sorted_timestamps[:split_index]
    )

    print(
        "Reference windows:",
        len(reference_timestamps)
    )

    print(
        "Streaming windows:",
        len(sorted_timestamps)
        - len(reference_timestamps)
    )

    # --------------------------------------------------------
    # Build reference vectors
    # --------------------------------------------------------

    reference_vectors = []

    for ts in reference_timestamps:

        vector = build_vector_from_rows(
            timestamps[ts],
            model_inputs
        )

        if vector is not None:

            reference_vectors.append(
                vector
            )

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    X = np.asarray(
        reference_vectors,
        dtype=float
    )

    print(
        "Reference vectors:",
        X.shape[0]
    )

    print(
        "Original dimensions:",
        X.shape[1]
    )

    # --------------------------------------------------------
    # Remove constant dimensions
    # --------------------------------------------------------

    variances = np.var(
        X,
        axis=0
    )

    active_dimensions = [

        i

        for i, variance in enumerate(
            variances
        )

        if variance > 1e-12
    ]

    if not active_dimensions:

        raise ValueError(
            "All reference dimensions "
            "are constant"
        )

    X_active = X[
        :,
        active_dimensions
    ]

    # --------------------------------------------------------
    # Reference mean
    # --------------------------------------------------------

    reference_mean = np.mean(
        X_active,
        axis=0
    )

    # --------------------------------------------------------
    # Reference covariance
    # --------------------------------------------------------

    reference_covariance = np.cov(
        X_active,
        rowvar=False
    )

    # --------------------------------------------------------
    # Handle one-dimensional covariance
    # --------------------------------------------------------

    if reference_covariance.ndim == 0:

        reference_covariance = np.array([
            [
                float(
                    reference_covariance
                )
            ]
        ])

    # --------------------------------------------------------
    # Regularization
    # --------------------------------------------------------

    regularization = 1e-6

    reference_covariance = (

        reference_covariance

        + regularization
        * np.eye(
            reference_covariance.shape[0]
        )
    )

    # --------------------------------------------------------
    # Store streaming windows
    # --------------------------------------------------------

    streaming_timestamps = (
        sorted_timestamps[split_index:]
    )

    # --------------------------------------------------------
    # Print reference information
    # --------------------------------------------------------

    print(
        "Active dimensions:",
        len(active_dimensions)
    )

    print(
        "Removed constant dimensions:",
        len(model_inputs)
        - len(active_dimensions)
    )

    print(
        "Reference mean dimension:",
        len(reference_mean)
    )

    print(
        "Covariance shape:",
        reference_covariance.shape
    )

    return {

        "reference_mean":
            reference_mean.tolist(),

        "reference_covariance":
            reference_covariance.tolist(),

        "active_dimensions":
            active_dimensions,

        # Store stream information temporarily
        "stream_timestamps":
            [
                {
                    "ts": ts,
                    "rows": timestamps[ts]
                }

                for ts in streaming_timestamps
            ]
    }


# ============================================================
# NODE 7
# CALCULATE MAHALANOBIS + ADWIN
# ============================================================

def calculate_mahalanobis_and_adwin(
    state: DriftState
):

    telemetry = state["telemetry"]
    model_inputs = state["model_inputs"]

    reference_mean = np.asarray(
        state["reference_mean"],
        dtype=float
    )

    reference_covariance = np.asarray(
        state["reference_covariance"],
        dtype=float
    )

    active_dimensions = np.asarray(
        state["active_dimensions"],
        dtype=int
    )

    # --------------------------------------------------------
    # Group telemetry into seconds
    # --------------------------------------------------------

    timestamps = {}

    for row in telemetry:

        ts = row["ts"].replace(
            microsecond=0
        )

        if ts not in timestamps:

            timestamps[ts] = []

        timestamps[ts].append(row)

    sorted_timestamps = sorted(
        timestamps.keys()
    )

    # --------------------------------------------------------
    # 80% reference / 20% stream
    # --------------------------------------------------------

    split_index = int(
        len(sorted_timestamps) * 0.80
    )

    streaming_timestamps = (
        sorted_timestamps[split_index:]
    )

    # --------------------------------------------------------
    # Covariance inverse
    # --------------------------------------------------------

    covariance_inverse = np.linalg.pinv(
        reference_covariance
    )

    # --------------------------------------------------------
    # Create ADWIN
    # --------------------------------------------------------

    adwin = ADWIN(
        delta=0.002
    )

    drift_scores = []
    drift_detections = []

    print("\n========================================")
    print("Mahalanobis + ADWIN")
    print("========================================")

    print(
        "ADWIN delta:",
        0.002
    )

    print(
        "Streaming windows:",
        len(streaming_timestamps)
    )

    # --------------------------------------------------------
    # Process every streaming second
    # --------------------------------------------------------

    for index, ts in enumerate(
        streaming_timestamps
    ):

        rows_at_time = timestamps[ts]

        vector = build_vector_from_rows(
            rows_at_time,
            model_inputs
        )

        if vector is None:

            continue

        current_vector = np.asarray(
            vector,
            dtype=float
        )

        # ----------------------------------------------------
        # Select active dimensions
        # ----------------------------------------------------

        current_active = (
            current_vector[
                active_dimensions
            ]
        )

        # ----------------------------------------------------
        # Difference from reference mean
        # ----------------------------------------------------

        difference = (
            current_active
            - reference_mean
        )

        # ----------------------------------------------------
        # Mahalanobis D²
        # ----------------------------------------------------

        distance_squared = (

            difference.T
            @ covariance_inverse
            @ difference
        )

        distance_squared = max(
            0.0,
            float(distance_squared)
        )

        distance = np.sqrt(
            distance_squared
        )

        # ----------------------------------------------------
        # Feed D² to ADWIN
        # ----------------------------------------------------

        adwin.update(
            distance_squared
        )

        # ----------------------------------------------------
        # Store score
        # ----------------------------------------------------

        score_record = {

            "index": index,

            "timestamp": str(ts),

            "mahalanobis_d2":
                distance_squared,

            "mahalanobis_d":
                distance,

            "adwin_width":
                adwin.width
        }

        drift_scores.append(
            score_record
        )

        # ----------------------------------------------------
        # Check ADWIN
        # ----------------------------------------------------

        if adwin.drift_detected:

            detection = {

                "index": index,

                "timestamp": str(ts),

                "mahalanobis_d2":
                    distance_squared,

                "mahalanobis_d":
                    distance,

                "adwin_width":
                    adwin.width
            }

            drift_detections.append(
                detection
            )

            print(
                f"DRIFT DETECTED | "
                f"time={ts} | "
                f"D²={distance_squared:.6f} | "
                f"D={distance:.6f} | "
                f"ADWIN width={adwin.width}"
            )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n========================================")
    print("Detection Summary")
    print("========================================")

    print(
        "Mahalanobis scores:",
        len(drift_scores)
    )

    print(
        "Drift detections:",
        len(drift_detections)
    )

    if drift_detections:

        print("\nDetected drift points:")

        for detection in drift_detections:

            print(
                f"Index={detection['index']} | "
                f"Time={detection['timestamp']} | "
                f"D²={detection['mahalanobis_d2']:.6f} | "
                f"D={detection['mahalanobis_d']:.6f} | "
                f"Width={detection['adwin_width']}"
            )

    else:

        print(
            "No drift detected in the "
            "streaming period."
        )

    print("========================================")

    # --------------------------------------------------------
    # Latest score
    # --------------------------------------------------------

    latest_distance = None

    if drift_scores:

        latest_distance = (
            drift_scores[-1]["mahalanobis_d"]
        )

    return {

        "mahalanobis_distance":
            latest_distance,

        "drift_scores":
            drift_scores,

        "drift_detections":
            drift_detections
    }


# ============================================================
# LANGGRAPH WORKFLOW
# ============================================================

builder = StateGraph(DriftState)


# ------------------------------------------------------------
# Add nodes
# ------------------------------------------------------------

builder.add_node(
    "receive_request",
    receive_request
)

builder.add_node(
    "resolve_model",
    resolve_model
)

builder.add_node(
    "resolve_inputs",
    resolve_inputs
)

builder.add_node(
    "fetch_live_data",
    fetch_live_data
)

builder.add_node(
    "build_model_vector",
    build_model_vector
)

builder.add_node(
    "build_reference_distribution",
    build_reference_distribution
)

builder.add_node(
    "calculate_mahalanobis_and_adwin",
    calculate_mahalanobis_and_adwin
)


# ------------------------------------------------------------
# Define graph
# ------------------------------------------------------------

builder.add_edge(
    START,
    "receive_request"
)

builder.add_edge(
    "receive_request",
    "resolve_model"
)

builder.add_edge(
    "resolve_model",
    "resolve_inputs"
)

builder.add_edge(
    "resolve_inputs",
    "fetch_live_data"
)

builder.add_edge(
    "fetch_live_data",
    "build_model_vector"
)

builder.add_edge(
    "build_model_vector",
    "build_reference_distribution"
)

builder.add_edge(
    "build_reference_distribution",
    "calculate_mahalanobis_and_adwin"
)

builder.add_edge(
    "calculate_mahalanobis_and_adwin",
    END
)


# ------------------------------------------------------------
# Compile
# ------------------------------------------------------------

graph = builder.compile()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    result = graph.invoke({

        # Test request
        "model_id": 9,

        # KPI reported by Monitoring Agent
        "degraded_kpis": [
            "throughput"
        ]
    })

    print("\n========================================")
    print("Agent finished.")
    print("========================================")

    print(
        "Final Mahalanobis distance:",
        result["mahalanobis_distance"]
    )

    print(
        "Total drift detections:",
        len(result["drift_detections"])
    )
