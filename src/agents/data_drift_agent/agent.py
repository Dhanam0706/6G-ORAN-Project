from state import DataDriftState


def decide_next_action(state: DataDriftState):
    """
    Autonomous decision-making logic for the Data Drift Agent.

    The agent examines the current state and decides
    what information or operation is required next.
    """

    # ----------------------------------------------
    # 1. Do we know the deployed model?
    # ----------------------------------------------

    if not state.get("model"):

        return {
            "next_action": "GET_MODEL",
            "messages": state.get("messages", []) + [
                "Model information is missing. "
                "I need to retrieve the deployed model."
            ]
        }

    # ----------------------------------------------
    # 2. Do we know the model input features?
    # ----------------------------------------------

    if not state.get("features"):

        return {
            "next_action": "GET_FEATURES",
            "messages": state.get("messages", []) + [
                "The deployed model is known, "
                "but its input features are missing. "
                "I need to retrieve the model inputs."
            ]
        }

    # ----------------------------------------------
    # 3. Do we have KPI data?
    # ----------------------------------------------

    if not state.get("feature_data"):

        return {
            "next_action": "GET_DATA",
            "messages": state.get("messages", []) + [
                "The required model inputs are known, "
                "but KPI observations are missing. "
                "I need to retrieve the required KPI data."
            ]
        }

    # ----------------------------------------------
    # 4. We have everything required
    # ----------------------------------------------

    return {
        "next_action": "RUN_DETECTOR",
        "messages": state.get("messages", []) + [
            "The deployed model, model inputs, "
            "and KPI data are available. "
            "The data is ready for drift detection."
        ]
    }