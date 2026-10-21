#!/usr/bin/env python3
"""6G-O-RAN concept drift agent.

Required environment variables for a degraded investigation:
    DB_HOST, DB_NAME, DB_USER, and DB_PASSWORD (or PGPASSWORD).
Optional variables:
    DB_PORT (default 5432), ADWIN_DELTA (default 0.002),
    GEMINI_MODEL, GEMINI_API_KEY.

The score field is assumed to be a per-decision reward and value_est
is assumed to be a compatible value estimate; verify these semantics
against the system that writes the database records.
"""

import os
import sys
import json
import argparse
import threading
from typing import TypedDict, Annotated

import pandas as pd
import psycopg2

from river.drift import ADWIN

from langchain_core.tools import tool
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
)

from langgraph.graph import (
    StateGraph,
    END,
)

from langgraph.graph.message import (
    add_messages,
)


# ============================================================
# CONFIGURATION
# ============================================================

# Set these values in the shell/environment. Do not commit real
# credentials or private infrastructure addresses to source control.
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD") or os.getenv("PGPASSWORD")

ADWIN_DELTA = float(os.getenv("ADWIN_DELTA", "0.002"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")


# ============================================================
# RUNTIME
# ============================================================

# Keep investigation state isolated per thread so simultaneous
# model investigations cannot overwrite each other's intermediate data.
_runtime_local = threading.local()


def get_runtime():
    """Return the runtime dictionary for the current thread."""
    runtime = getattr(_runtime_local, "data", None)
    if runtime is None:
        reset_runtime()
        runtime = _runtime_local.data
    return runtime


def reset_runtime():
    """Create a fresh, isolated runtime state for this investigation."""
    _runtime_local.data = {
        "model_metadata": None,
        "current_deployment": None,
        "reference_deployment": None,
        "current_decisions": None,
        "reference_decisions": None,
        "score_validation": None,
        "td_signal": None,
        "drift_result": None,
    }
    return _runtime_local.data


reset_runtime()


# ============================================================
# JSON HELPERS
# ============================================================

def json_default(obj):

    if hasattr(obj, "item"):
        return obj.item()

    if hasattr(obj, "isoformat"):
        return obj.isoformat()

    return str(obj)


def normalize_json(value):

    if value is None:
        return None

    if isinstance(value, dict):
        return value

    if isinstance(value, str):

        try:
            return json.loads(value)

        except Exception:
            return value

    return value


def clean_record(record):

    cleaned = {}

    for key, value in record.items():

        if pd.isna(value):
            cleaned[key] = None

        else:
            cleaned[key] = value

    return cleaned


# ============================================================
# DATABASE
# ============================================================

def db_connect():
    """Connect to PostgreSQL using environment-provided settings."""
    required = {
        "DB_HOST": DB_HOST,
        "DB_NAME": DB_NAME,
        "DB_USER": DB_USER,
        "DB_PASSWORD or PGPASSWORD": DB_PASSWORD,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(
            "Missing database environment setting(s): "
            + ", ".join(missing)
            + ". Set them before running a degraded-model investigation."
        )

    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def execute_query(
    query,
    params=None,
):
    """
    Execute a PostgreSQL query using a psycopg2
    cursor and return the result as a pandas DataFrame.

    This avoids the pandas warning that occurs when
    passing a raw psycopg2 connection to read_sql_query().
    """

    conn = db_connect()

    try:

        with conn.cursor() as cursor:

            cursor.execute(
                query,
                params,
            )

            rows = cursor.fetchall()

            columns = [
                description[0]
                for description in cursor.description
            ]

        return pd.DataFrame(
            rows,
            columns=columns,
        )

    finally:

        conn.close()


# ============================================================
# TOOL 1
# GET MODEL METADATA
# ============================================================

@tool
def get_model_metadata(
    model_id: int,
):
    """
    Get model information from ran_model.
    """

    query = """
        SELECT
            model_id,
            name AS model_name,
            kind

        FROM ran_model

        WHERE model_id = %s

        LIMIT 1
    """

    df = execute_query(
        query,
        (model_id,),
    )

    if df.empty:

        raise ValueError(
            f"Model {model_id} does not exist "
            f"in ran_model."
        )

    record = clean_record(
        df.iloc[0].to_dict()
    )

    get_runtime()[
        "model_metadata"
    ] = record

    return record


# ============================================================
# TOOL 2
# GET CURRENT DEPLOYMENT
# ============================================================

@tool
def get_current_deployment(
    model_id: int,
):
    """
    Get an active deployment (mode='deploy' and ended_at IS NULL).
    If no active deployment exists, fail instead of silently selecting
    an old deployment that merely started most recently.

    Relationship:

        model_deployment.version_id
                    |
                    v
        ran_model_version.version_id
                    |
                    v
        ran_model_version.model_id
                    |
                    v
        ran_model.model_id
    """

    query = """
        SELECT

            md.deployment_id,
            md.version_id,

            rmv.model_id,
            rmv.version,
            rmv.status,
            rmv.algorithm,
            rmv.hyperparams,

            rm.name AS model_name,
            rm.kind,

            md.mode,
            md.started_at,
            md.ended_at,
            md.run_id,
            md.cell_id,
            md.control_period_ms,
            md.seed,
            md.notes

        FROM model_deployment md

        JOIN ran_model_version rmv
            ON rmv.version_id = md.version_id

        JOIN ran_model rm
            ON rm.model_id = rmv.model_id

        WHERE rmv.model_id = %s
          AND md.mode = 'deploy'
          AND md.ended_at IS NULL

        ORDER BY
            md.started_at DESC,
            md.deployment_id DESC

        LIMIT 1
    """

    df = execute_query(
        query,
        (model_id,),
    )

    if df.empty:

        raise ValueError(
            f"No active deployment found for model_id {model_id}. "
            "Expected mode='deploy' and ended_at IS NULL."
        )

    record = clean_record(
        df.iloc[0].to_dict()
    )

    record["hyperparams"] = normalize_json(
        record.get("hyperparams")
    )

    get_runtime()[
        "current_deployment"
    ] = record

    return record


# ============================================================
# TOOL 3
# GET REFERENCE DEPLOYMENT
# ============================================================

@tool
def get_reference_deployment(
    model_id: int,
    current_deployment_id: int,
):
    """
    Find the reference training deployment.

    First preference:
        same model
        same version
        mode = train
        completed deployment

    Fallback:
        latest completed training deployment
        for the same model.
    """

    current = get_runtime().get(
        "current_deployment"
    )

    current_version_id = None

    if current:

        current_version_id = current.get(
            "version_id"
        )

    # --------------------------------------------------------
    # SAME VERSION REFERENCE
    # --------------------------------------------------------

    if current_version_id is not None:

        query_same_version = """
            SELECT

                md.deployment_id,
                md.version_id,

                rmv.model_id,
                rmv.version,
                rmv.status,
                rmv.algorithm,
                rmv.hyperparams,

                rm.name AS model_name,
                rm.kind,

                md.mode,
                md.started_at,
                md.ended_at,
                md.run_id,
                md.cell_id,
                md.control_period_ms,
                md.seed,
                md.notes

            FROM model_deployment md

            JOIN ran_model_version rmv
                ON rmv.version_id = md.version_id

            JOIN ran_model rm
                ON rm.model_id = rmv.model_id

            WHERE rmv.model_id = %s
              AND md.version_id = %s
              AND md.mode = 'train'
              AND md.ended_at IS NOT NULL
              AND md.deployment_id <> %s

            ORDER BY
                md.ended_at DESC,
                md.deployment_id DESC

            LIMIT 1
        """

        df = execute_query(
            query_same_version,
            (
                model_id,
                current_version_id,
                current_deployment_id,
            ),
        )

        if not df.empty:

            record = clean_record(
                df.iloc[0].to_dict()
            )

            record["hyperparams"] = (
                normalize_json(
                    record.get(
                        "hyperparams"
                    )
                )
            )

            get_runtime()[
                "reference_deployment"
            ] = record

            return record

    # --------------------------------------------------------
    # FALLBACK REFERENCE
    # --------------------------------------------------------

    query_fallback = """
        SELECT

            md.deployment_id,
            md.version_id,

            rmv.model_id,
            rmv.version,
            rmv.status,
            rmv.algorithm,
            rmv.hyperparams,

            rm.name AS model_name,
            rm.kind,

            md.mode,
            md.started_at,
            md.ended_at,
            md.run_id,
            md.cell_id,
            md.control_period_ms,
            md.seed,
            md.notes

        FROM model_deployment md

        JOIN ran_model_version rmv
            ON rmv.version_id = md.version_id

        JOIN ran_model rm
            ON rm.model_id = rmv.model_id

        WHERE rmv.model_id = %s
          AND md.mode = 'train'
          AND md.ended_at IS NOT NULL
          AND md.deployment_id <> %s

        ORDER BY
            md.ended_at DESC,
            md.deployment_id DESC

        LIMIT 1
    """

    df = execute_query(
        query_fallback,
        (
            model_id,
            current_deployment_id,
        ),
    )

    if df.empty:

        raise ValueError(
            f"No completed training deployment "
            f"found for model_id {model_id}."
        )

    record = clean_record(
        df.iloc[0].to_dict()
    )

    record["hyperparams"] = normalize_json(
        record.get(
            "hyperparams"
        )
    )

    get_runtime()[
        "reference_deployment"
    ] = record

    return record


# ============================================================
# TOOL 4
# FETCH DECISION STREAMS
# ============================================================

@tool
def fetch_decision_streams(
    current_deployment_id: int,
    reference_deployment_id: int,
):
    """
    Fetch the decision streams required for
    TD-error based concept drift detection.

    model_decision actual columns used:

        decision_id
        deployment_id
        step
        value_est
        score
    """

    query = """
        SELECT

            decision_id,
            deployment_id,
            step,
            value_est,
            score

        FROM model_decision

        WHERE deployment_id IN (%s, %s)

        ORDER BY
            deployment_id,
            step,
            decision_id
    """

    df = execute_query(
        query,
        (
            reference_deployment_id,
            current_deployment_id,
        ),
    )

    if df.empty:

        raise ValueError(
            "No model decision records found "
            "for the selected deployments."
        )

    df["deployment_id"] = pd.to_numeric(
        df["deployment_id"],
        errors="coerce",
    )

    reference_df = df[
        df["deployment_id"]
        == reference_deployment_id
    ].copy()

    current_df = df[
        df["deployment_id"]
        == current_deployment_id
    ].copy()

    if reference_df.empty:

        raise ValueError(
            "Reference deployment has no "
            "decision records."
        )

    if current_df.empty:

        raise ValueError(
            "Current deployment has no "
            "decision records."
        )

    get_runtime()[
        "reference_decisions"
    ] = reference_df

    get_runtime()[
        "current_decisions"
    ] = current_df

    return {
        "reference_rows": len(
            reference_df
        ),
        "current_rows": len(
            current_df
        ),
    }


# ============================================================
# TOOL 5
# VALIDATE SCORE SEMANTICS
# ============================================================

@tool
def validate_score_semantics(
    deployment_id: int,
):
    """
    Basic consistency validation of the
    model decision stream.

    This is not a drift detector.
    """

    query = """
        SELECT
            COUNT(*) AS total_rows,
            COUNT(score) AS score_rows,
            COUNT(value_est) AS value_rows

        FROM model_decision

        WHERE deployment_id = %s
    """

    df = execute_query(
        query,
        (deployment_id,),
    )

    if df.empty:
        raise ValueError(
            f"No model decisions found for deployment_id={deployment_id}."
        )

    row = df.iloc[0]
    total_rows = int(row["total_rows"])
    score_rows = int(row["score_rows"])
    value_rows = int(row["value_rows"])

    # At least two complete decision records are needed to construct
    # a signal using a current and next-available value estimate.
    validated = (
        total_rows >= 2
        and score_rows >= 2
        and value_rows >= 2
    )

    result = {
        "deployment_id": deployment_id,
        "validated": validated,
        "total_rows": total_rows,
        "score_rows": score_rows,
        "value_rows": value_rows,
        "minimum_required_rows": 2,
        "semantic_assumption": (
            "score must represent the per-decision reward and value_est "
            "must represent a compatible value estimate; verify these "
            "meanings against the database producer/model implementation."
        ),
    }

    get_runtime()["score_validation"] = result

    if not validated:
        raise ValueError(
            f"Decision data validation failed for deployment_id={deployment_id}: "
            f"total_rows={total_rows}, score_rows={score_rows}, "
            f"value_rows={value_rows}; at least two usable records are required."
        )

    return result


# ============================================================
# TOOL 6
# BUILD TD-ERROR SIGNAL
# ============================================================

@tool
def build_concept_drift_signal():
    """
    Construct the absolute TD-error stream.

    TD-style signal:

        delta_t =
            score_t
            + gamma * V_(next available decision)
            - V_t

    Important real-time behavior:

        The detector uses the NEXT AVAILABLE
        decision record.

        It does NOT require:

            next_step == current_step + 1

        Therefore, if the stream contains:

            step 100
            step 101
            step 103
            step 104

        the transitions are:

            100 -> 101
            101 -> 103
            103 -> 104

        The detector does not wait for a missing
        step and does not block real-time monitoring.

    Drift signal:

        abs(delta_t)
    """

    reference_df = get_runtime().get(
        "reference_decisions"
    )

    current_df = get_runtime().get(
        "current_decisions"
    )

    current_deployment = get_runtime().get(
        "current_deployment"
    )

    if reference_df is None:

        raise ValueError(
            "Reference decision stream is missing."
        )

    if current_df is None:

        raise ValueError(
            "Current decision stream is missing."
        )

    if current_deployment is None:

        raise ValueError(
            "Current deployment metadata is missing."
        )

    hyperparams = normalize_json(
        current_deployment.get(
            "hyperparams"
        )
    )

    gamma = 0.95

    if isinstance(
        hyperparams,
        dict,
    ):

        try:

            gamma = float(
                hyperparams.get(
                    "gamma",
                    0.95,
                )
            )

        except Exception:

            gamma = 0.95

    def calculate_td_error(
        df,
    ):
        """
        Calculate the TD-style signal using
        consecutive AVAILABLE records in the
        ordered stream.

        This intentionally does NOT require
        numerical step continuity.

        Example:

            current row:
                step = 102
                value = V102
                score = R102

            next available row:
                step = 104
                value = V104

            signal:

                R102 + gamma * V104 - V102

        This avoids blocking when step 103 is
        missing from the runtime stream.
        """

        data = df.copy()

        data["value_est"] = pd.to_numeric(
            data["value_est"],
            errors="coerce",
        )

        data["score"] = pd.to_numeric(
            data["score"],
            errors="coerce",
        )

        data["step"] = pd.to_numeric(
            data["step"],
            errors="coerce",
        )

        data["decision_id"] = pd.to_numeric(
            data["decision_id"],
            errors="coerce",
        )

        data = data.dropna(
            subset=[
                "value_est",
                "score",
                "step",
                "decision_id",
            ]
        )

        # Ignore infinities as well as null/non-numeric values.
        numeric_columns = ["value_est", "score", "step", "decision_id"]
        data = data[
            data[numeric_columns].apply(
                lambda column: column.map(
                    lambda value: pd.notna(value)
                    and abs(float(value)) != float("inf")
                )
            ).all(axis=1)
        ]

        if len(data) < 2:
            return pd.DataFrame(
                columns=["step", "next_step", "td_error", "abs_td_error"]
            )

        # ----------------------------------------------------
        # Sort chronologically.
        #
        # We DO NOT check:
        #
        #     next_step == current_step + 1
        #
        # The next row after sorting is simply
        # the next available decision record.
        # ----------------------------------------------------

        data = data.sort_values(
            [
                "step",
                "decision_id",
            ]
        ).reset_index(
            drop=True
        )

        td_errors = []

        # ----------------------------------------------------
        # NEXT AVAILABLE RECORD LOGIC
        # ----------------------------------------------------

        for i in range(
            len(data) - 1
        ):

            current_row = data.iloc[i]

            next_row = data.iloc[
                i + 1
            ]

            current_value = float(
                current_row["value_est"]
            )

            current_score = float(
                current_row["score"]
            )

            next_value = float(
                next_row["value_est"]
            )

            current_step = int(
                current_row["step"]
            )

            next_step = int(
                next_row["step"]
            )

            td_error = (
                current_score
                + gamma * next_value
                - current_value
            )

            td_errors.append(
                {
                    "step": current_step,

                    "next_step": next_step,

                    "td_error": float(
                        td_error
                    ),

                    "abs_td_error": float(
                        abs(td_error)
                    ),
                }
            )

        return pd.DataFrame(
            td_errors
        )

    # --------------------------------------------------------
    # BUILD REFERENCE SIGNAL
    # --------------------------------------------------------

    reference_signal = (
        calculate_td_error(
            reference_df
        )
    )

    # --------------------------------------------------------
    # BUILD CURRENT / LIVE SIGNAL
    # --------------------------------------------------------

    current_signal = (
        calculate_td_error(
            current_df
        )
    )

    if reference_signal.empty:

        raise ValueError(
            "Reference TD-error signal is empty."
        )

    if current_signal.empty:

        raise ValueError(
            "Current TD-error signal is empty."
        )

    get_runtime()[
        "td_signal"
    ] = {
        "gamma": gamma,
        "reference": reference_signal,
        "current": current_signal,
    }

    return {
        "gamma": gamma,

        "reference_samples": len(
            reference_signal
        ),

        "current_samples": len(
            current_signal
        ),
    }


# ============================================================
# TOOL 7
# RUN ADWIN
# ============================================================

@tool
def run_adwin(
    delta: float = ADWIN_DELTA,
):
    """
    Run ADWIN over the absolute TD-error stream.

    Reference stream:
        used to establish the reference distribution.

    Current stream:
        monitored for distribution changes.

    ADWIN provides the deterministic
    concept-drift decision.
    """

    signal = get_runtime().get(
        "td_signal"
    )

    if signal is None:

        raise ValueError(
            "TD-error signal has not been built."
        )

    reference_signal = signal[
        "reference"
    ]

    current_signal = signal[
        "current"
    ]

    detector = ADWIN(
        delta=delta
    )

    # --------------------------------------------------------
    # Reference / warm-up stream
    # --------------------------------------------------------

    for value in reference_signal[
        "abs_td_error"
    ]:

        detector.update(
            float(value)
        )

    # --------------------------------------------------------
    # Current / live stream
    # --------------------------------------------------------

    detections = []

    for _, row in current_signal.iterrows():

        value = float(
            row["abs_td_error"]
        )

        detector.update(
            value
        )

        if detector.drift_detected:

            detection = {
                "step": int(
                    row["step"]
                ),

                "td_error": float(
                    row["td_error"]
                ),

                "abs_td_error": float(
                    row["abs_td_error"]
                ),

                "width": int(
                    detector.width
                ),
            }

            # Include next_step when available.
            if "next_step" in row:

                detection[
                    "next_step"
                ] = int(
                    row["next_step"]
                )

            detections.append(detection)

            # Start a fresh ADWIN window after recording a detection.
            # This avoids depending on River's private _reset() method.
            detector = ADWIN(delta=delta)

    # --------------------------------------------------------
    # CONCEPT DRIFT SEVERITY SCORE
    # --------------------------------------------------------
    #
    # The score measures the standardized shift between
    # the reference and current absolute TD-error signals.
    # It is a shift-magnitude index, NOT a drift probability.
    #
    #     Z =
    #         |mean_current - mean_reference|
    #         --------------------------------
    #         std_reference + epsilon
    #
    # The standardized shift is mapped to 0-100.
    #
    # A 3-sigma shift or greater gives a score of 100.
    #
    # This score is supplementary.
    # ADWIN remains the deterministic drift detector.
    # --------------------------------------------------------

    reference_values = pd.to_numeric(
        reference_signal[
            "abs_td_error"
        ],
        errors="coerce",
    ).dropna()

    current_values = pd.to_numeric(
        current_signal[
            "abs_td_error"
        ],
        errors="coerce",
    ).dropna()

    if (
        len(reference_values) > 0
        and len(current_values) > 0
    ):

        reference_mean = float(
            reference_values.mean()
        )

        current_mean = float(
            current_values.mean()
        )

        reference_std = float(
            reference_values.std(
                ddof=1
            )
        )

        # Handle the case where the reference signal
        # has zero or undefined standard deviation.
        if (
            pd.isna(reference_std)
            or reference_std <= 0
        ):

            reference_std = 1e-12

        standardized_shift = (
            abs(
                current_mean
                - reference_mean
            )
            / (
                reference_std
                + 1e-12
            )
        )

        severity_score = min(
            100.0,
            (
                standardized_shift
                / 3.0
            ) * 100.0,
        )

    else:

        severity_score = 0.0

    severity_score = round(
        float(severity_score),
        2,
    )

    result = {
        "algorithm": "ADWIN",

        "delta": float(delta),

        "signal": "absolute_td_error",

        "concept_drift_detected": (
            len(detections) > 0
        ),

        "detection_count": len(
            detections
        ),

        "detections": detections,

        "severity_score": severity_score,
    }

    get_runtime()[
        "drift_result"
    ] = result

    return result


# ============================================================
# TOOL 8
# GET DRIFT RESULT
# ============================================================

@tool
def get_drift_result():
    """
    Return the deterministic ADWIN result.
    """

    result = get_runtime().get(
        "drift_result"
    )

    if result is None:

        raise ValueError(
            "Drift result is not available."
        )

    return result


# ============================================================
# LANGGRAPH STATE
# ============================================================

class AgentState(
    TypedDict
):

    messages: Annotated[
        list,
        add_messages,
    ]


# ============================================================
# GEMINI
# ============================================================

def create_llm():

    if not GEMINI_API_KEY:

        raise RuntimeError(
            "GEMINI_API_KEY environment variable "
            "is not set."
        )

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
    )


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are the Concept Drift Agent in a
6G-O-RAN model monitoring architecture.

The Monitoring Agent has already determined
whether a model is degraded.

When degraded=true, perform the concept-drift
investigation.

Follow this exact sequence:

1. Get model metadata.
2. Get the current deployed deployment.
3. Get the reference training deployment.
4. Fetch the decision streams.
5. Validate the decision data.
6. Build the TD-error signal.
7. Run ADWIN on absolute TD-error.
8. Get the final drift result.

Important rules:

- Use the supplied model_id.
- Do not invent database values.
- Do not change the model_id.
- Do not assume a fixed model ID.
- Do not override ADWIN.
- ADWIN is the deterministic authority
  for concept-drift detection.
- The severity score is a normalized shift-magnitude index,
  not a probability of drift.
- Gemini only orchestrates the workflow.
- Do not perform KS.
- Do not perform PSI.
- Do not perform JSD.
- Do not perform Wasserstein distance.
- Do not perform data-drift detection.
- Do not perform additional KPI/state analysis.
- Do not generate an independent degraded flag.
- The degraded status comes from the
  Monitoring Agent.

For the TD-error signal, the system uses
the next AVAILABLE decision record.
Do not require step + 1 to exist.
Do not wait for missing decision steps.
"""


# ============================================================
# BUILD LANGGRAPH
# ============================================================

def build_graph(
    model_id: int,
    model_name: str,
):

    llm = create_llm()

    tools = [
        get_model_metadata,
        get_current_deployment,
        get_reference_deployment,
        fetch_decision_streams,
        validate_score_semantics,
        build_concept_drift_signal,
        run_adwin,
        get_drift_result,
    ]

    llm_with_tools = llm.bind_tools(
        tools
    )

    tool_map = {
        current_tool.name: current_tool
        for current_tool in tools
    }

    def agent_node(
        state: AgentState
    ):

        response = llm_with_tools.invoke(
            [
                SystemMessage(
                    content=SYSTEM_PROMPT
                ),
                *state["messages"],
            ]
        )

        return {
            "messages": [
                response
            ]
        }

    def tool_node(
        state: AgentState
    ):

        last_message = state[
            "messages"
        ][-1]

        results = []

        for tool_call in (
            last_message.tool_calls
        ):

            tool_name = tool_call[
                "name"
            ]

            tool_args = tool_call.get(
                "args",
                {},
            )

            if tool_name not in tool_map:

                raise ValueError(
                    f"Unknown tool: "
                    f"{tool_name}"
                )

            result = tool_map[
                tool_name
            ].invoke(
                tool_args
            )

            results.append(
                ToolMessage(
                    content=json.dumps(
                        result,
                        default=json_default,
                    ),
                    tool_call_id=tool_call[
                        "id"
                    ],
                )
            )

        return {
            "messages": results
        }

    def route_after_agent(
        state: AgentState
    ):

        last_message = state[
            "messages"
        ][-1]

        if getattr(
            last_message,
            "tool_calls",
            None,
        ):

            return "tools"

        return END

    graph = StateGraph(
        AgentState
    )

    graph.add_node(
        "agent",
        agent_node,
    )

    graph.add_node(
        "tools",
        tool_node,
    )

    graph.set_entry_point(
        "agent"
    )

    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            END: END,
        },
    )

    graph.add_edge(
        "tools",
        "agent",
    )

    return graph.compile()


# ============================================================
# SINGLE MODEL INVESTIGATION
# ============================================================

def run_autonomous_investigation(
    model_id: int,
    model_name: str,
    degraded: bool,
):
    """
    Process one model.

    degraded=false:
        do not access DB
        do not run drift detection

    degraded=true:
        perform complete investigation
    """

    # --------------------------------------------------------
    # NOT DEGRADED
    # --------------------------------------------------------

    if not degraded:

        return {
            "model_id": model_id,
            "model_name": model_name,

            "monitoring_degraded": False,

            "concept_drift_detected": False,

            "severity_score": 0.0,

            "status": "not_investigated",

            "reason": (
                "Monitoring Agent reported "
                "degraded=false."
            ),
        }

    # --------------------------------------------------------
    # DEGRADED
    # --------------------------------------------------------

    reset_runtime()

    try:

        graph = build_graph(
            model_id=model_id,
            model_name=model_name,
        )

        initial_message = HumanMessage(
            content=(
                "Start the concept drift "
                "investigation for "
                f"model_id={model_id}, "
                f"model_name={model_name}. "
                "The Monitoring Agent reported "
                "degraded=true. "
                "Follow the exact investigation "
                "sequence."
            )
        )

        graph.invoke(
            {
                "messages": [
                    initial_message
                ]
            }
        )

        # ----------------------------------------------------
        # ADWIN IS THE FINAL AUTHORITY
        # ----------------------------------------------------

        drift_result = get_runtime().get(
            "drift_result"
        )

        if drift_result is None:

            raise RuntimeError(
                "ADWIN did not produce "
                "a drift result."
            )

        concept_drift_detected = bool(
            drift_result.get(
                "concept_drift_detected",
                False,
            )
        )

        severity_score = float(
            drift_result.get(
                "severity_score",
                0.0,
            )
        )

        return {
            "model_id": model_id,
            "model_name": model_name,

            "monitoring_degraded": True,

            "concept_drift_detected":
                concept_drift_detected,

            "severity_score":
                severity_score,

            "status": "completed",
        }

    except Exception as exc:

        return {
            "model_id": model_id,
            "model_name": model_name,

            "monitoring_degraded": True,

            "concept_drift_detected": False,

            "severity_score": 0.0,

            "status": "investigation_error",

            "error": str(exc),
        }


# ============================================================
# NORMALIZE MODEL INPUT
# ============================================================

def normalize_model_entry(
    entry
):

    if not isinstance(
        entry,
        dict,
    ):

        raise ValueError(
            "Each model entry must be "
            "a JSON object."
        )

    if "model_id" not in entry:

        raise ValueError(
            "model_id is required."
        )

    model_id = int(
        entry["model_id"]
    )

    model_name = entry.get(
        "model_name"
    )

    if model_name is None:

        model_name = (
            f"model-{model_id}"
        )

    if "degraded" not in entry:

        raise ValueError(
            f"degraded flag is required "
            f"for model_id={model_id}."
        )

    degraded = entry[
        "degraded"
    ]

    if isinstance(
        degraded,
        str,
    ):

        degraded = (
            degraded.lower()
            in (
                "true",
                "1",
                "yes",
                "y",
            )
        )

    else:

        degraded = bool(
            degraded
        )

    return {
        "model_id": model_id,
        "model_name": str(
            model_name
        ),
        "degraded": degraded,
    }


# ============================================================
# PROCESS MULTIPLE MODELS
# ============================================================

def run_multiple_models(
    models
):
    """
    Process every model independently.
    """

    results = []

    for model in models:

        normalized = (
            normalize_model_entry(
                model
            )
        )

        result = (
            run_autonomous_investigation(
                model_id=normalized[
                    "model_id"
                ],
                model_name=normalized[
                    "model_name"
                ],
                degraded=normalized[
                    "degraded"
                ],
            )
        )

        results.append(
            result
        )

    return {
        "models_processed": len(
            results
        ),
        "results": results,
    }


# ============================================================
# PROCESS MONITORING PAYLOAD
# ============================================================

def process_monitoring_payload(
    payload
):
    """
    Expected input:

    {
        "models": [
            {
                "model_id": 9,
                "model_name": "ppo-gnn",
                "degraded": true
            }
        ]
    }
    """

    if not isinstance(
        payload,
        dict,
    ):

        raise ValueError(
            "Input must be a JSON object."
        )

    models = payload.get(
        "models"
    )

    if not isinstance(
        models,
        list,
    ):

        raise ValueError(
            "'models' must be a list."
        )

    return run_multiple_models(
        models
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "6G-O-RAN Concept Drift Agent"
        )
    )

    parser.add_argument(
        "--models",
        type=str,
        help="JSON list of models.",
    )

    parser.add_argument(
        "--input-json",
        type=str,
        help="JSON monitoring payload.",
    )

    parser.add_argument(
        "--model-id",
        type=int,
        help="Single model ID.",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        help="Single model name.",
    )

    parser.add_argument(
        "--degraded",
        type=str,
        help="Single model degraded flag.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # MULTIPLE MODELS
    # --------------------------------------------------------

    if args.models:

        try:

            models = json.loads(
                args.models
            )

            result = run_multiple_models(
                models
            )

        except Exception as exc:

            result = {
                "error": str(exc)
            }

        print(
            json.dumps(
                result,
                indent=2,
                default=json_default,
            )
        )

        return

    # --------------------------------------------------------
    # JSON PAYLOAD
    # --------------------------------------------------------

    if args.input_json:

        try:

            payload = json.loads(
                args.input_json
            )

            result = (
                process_monitoring_payload(
                    payload
                )
            )

        except Exception as exc:

            result = {
                "error": str(exc)
            }

        print(
            json.dumps(
                result,
                indent=2,
                default=json_default,
            )
        )

        return

    # --------------------------------------------------------
    # SINGLE MODEL
    # --------------------------------------------------------

    if args.model_id is not None:

        if args.degraded is None:

            raise ValueError(
                "--degraded is required "
                "when using --model-id."
            )

        model_name = (
            args.model_name
            or f"model-{args.model_id}"
        )

        degraded = (
            args.degraded.lower()
            in (
                "true",
                "1",
                "yes",
                "y",
            )
        )

        result = (
            run_autonomous_investigation(
                model_id=args.model_id,
                model_name=model_name,
                degraded=degraded,
            )
        )

        print(
            json.dumps(
                result,
                indent=2,
                default=json_default,
            )
        )

        return

    # --------------------------------------------------------
    # STDIN
    # --------------------------------------------------------

    if not sys.stdin.isatty():

        raw = sys.stdin.read().strip()

        if raw:

            try:

                payload = json.loads(
                    raw
                )

                result = (
                    process_monitoring_payload(
                        payload
                    )
                )

            except Exception as exc:

                result = {
                    "error": str(exc)
                }

            print(
                json.dumps(
                    result,
                    indent=2,
                    default=json_default,
                )
            )

            return

    parser.print_help()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
