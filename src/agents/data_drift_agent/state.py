from typing import TypedDict, List, Dict, Any


class DataDriftState(TypedDict, total=False):

    # Agent objective
    goal: str

    # Information discovered by the agent
    model: Dict[str, Any]
    features: List[Dict[str, Any]]

    # Retrieved KPI data
    feature_data: Dict[str, Any]

    # Data Drift results
    drift_results: Dict[str, Any]

    # Agent control
    next_action: str
    completed: bool

    # Agent execution history
    messages: List[str]