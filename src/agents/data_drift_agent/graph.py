from langgraph.graph import StateGraph, START, END

from state import DataDriftState

from tools import (
    get_deployed_model,
    get_model_inputs,
    get_feature_data
)

from agent import decide_next_action

import sys
import os

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..")
)

sys.path.insert(0, PROJECT_ROOT)

from src.data_drift.drift_detection import detect_data_drift


def get_model_node(state: DataDriftState):
    print("\n[AGENT ACTION] GET_MODEL")

    model = get_deployed_model()

    if model is None:
        raise ValueError("No deployed model found in database.")

    return {
        "model": model
    }


def get_features_node(state: DataDriftState):
    print("\n[AGENT ACTION] GET_FEATURES")

    model = state["model"]

    features = get_model_inputs(
        model["version_id"]
    )

    if not features:
        raise ValueError(
            "No model input features found."
        )

    return {
        "features": features
    }


def get_data_node(state: DataDriftState):
    print("\n[AGENT ACTION] GET_DATA")

    model = state["model"]
    features = state["features"]

    feature_data = {}

    for feature in features:

        feature_name = feature["name"]

        if feature_name.startswith("slice1"):
            slice_id = 1

        elif feature_name.startswith("slice5"):
            slice_id = 5

        else:
            print(
                f"Skipping unsupported feature: {feature_name}"
            )
            continue

        source_column = (
            feature["source_expr"]
            .split(".")[-1]
        )

        data = get_feature_data(
            run_id=model["run_id"],
            slice_id=slice_id,
            source_column=source_column,
            limit=100
        )

        feature_data[feature_name] = data

    return {
        "feature_data": feature_data
    }


def run_detector_node(state: DataDriftState):

    print("\n[AGENT ACTION] RUN_DETECTOR")

    feature_data = state["feature_data"]

    drift_results = {}

    for feature_name, rows in feature_data.items():

        if len(rows) < 2:
            print(
                f"Skipping {feature_name}: "
                f"not enough data."
            )
            continue

        # Extract KPI values
        values = [
            row["value"]
            for row in rows
            if row["value"] is not None
        ]

        if len(values) < 2:
            continue

        # Split the retrieved observations
        # into reference and current windows.
        midpoint = len(values) // 2

        reference_data = values[:midpoint]
        current_data = values[midpoint:]

        print(
            f"\nRunning drift detection for "
            f"{feature_name}"
        )

        result = detect_data_drift(
            reference_data=reference_data,
            current_data=current_data
        )

        drift_results[feature_name] = result

    return {
        "drift_results": drift_results,
        "completed": True
    }


def agent_decision(state: DataDriftState):

    result = decide_next_action(state)

    action = result["next_action"]

    print(
        f"\n[AGENT DECISION] {action}"
    )

    print(
        f"[AGENT REASON] "
        f"{result['messages'][-1]}"
    )

    return result


def route_agent(state: DataDriftState):

    action = state["next_action"]

    if action == "GET_MODEL":
        return "get_model"

    elif action == "GET_FEATURES":
        return "get_features"

    elif action == "GET_DATA":
        return "get_data"

    elif action == "RUN_DETECTOR":
        return "run_detector"

    else:
        raise ValueError(
            f"Unknown agent action: {action}"
        )


builder = StateGraph(DataDriftState)

# Agent decision node
builder.add_node(
    "agent",
    agent_decision
)

# Data retrieval nodes
builder.add_node(
    "get_model",
    get_model_node
)

builder.add_node(
    "get_features",
    get_features_node
)

builder.add_node(
    "get_data",
    get_data_node
)

# Drift detector node
builder.add_node(
    "run_detector",
    run_detector_node
)


# START → Agent
builder.add_edge(
    START,
    "agent"
)


# Agent decides where to go next
builder.add_conditional_edges(
    "agent",
    route_agent,
    {
        "get_model": "get_model",
        "get_features": "get_features",
        "get_data": "get_data",
        "run_detector": "run_detector"
    }
)


# After retrieving information,
# return to the autonomous agent
builder.add_edge(
    "get_model",
    "agent"
)

builder.add_edge(
    "get_features",
    "agent"
)

builder.add_edge(
    "get_data",
    "agent"
)


# Detector finishes the workflow
builder.add_edge(
    "run_detector",
    END
)


graph = builder.compile()


if __name__ == "__main__":

    initial_state = {
        "goal": (
            "Retrieve the required PostgreSQL data "
            "and perform Data Drift Detection"
        ),
        "messages": [],
        "completed": False
    }

    result = graph.invoke(
        initial_state
    )

    print(
        "\n==================================="
    )
    print(
        "     AUTONOMOUS DATA DRIFT AGENT"
    )
    print(
        "==================================="
    )

    print("\nMODEL:")
    print(
        result.get("model")
    )

    print("\nNUMBER OF FEATURES:")
    print(
        len(result.get("features", []))
    )

    print("\nFEATURE DATA:")

    for feature, data in result.get(
        "feature_data", {}
    ).items():

        print(
            f"{feature} -> "
            f"{len(data)} rows"
        )

    print("\nDRIFT RESULTS:")

    for feature, result_data in result.get(
        "drift_results", {}
    ).items():

        print(
            f"\n{feature}"
        )

        print(
            result_data
        )

    print(
        "\nAGENT DECISION HISTORY:"
    )

    for message in result.get(
        "messages",
        []
    ):

        print(
            "-",
            message
        )

    print(
        "\n==================================="
    )
    print(
        "        AGENT COMPLETED"
    )
    print(
        "==================================="
    )