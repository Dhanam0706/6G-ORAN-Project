import psycopg2
import pandas as pd


DB_CONFIG = {
    "host": "192.168.109.154",
    "port": 5433,
    "database": "rankpi",
    "user": "rankpi",
    "password": input("Enter PostgreSQL password: ")
}


def get_connection():
    """
    Create a connection to the live PostgreSQL database.
    """
    return psycopg2.connect(**DB_CONFIG)


def get_live_actions(limit=50):
    """
    Retrieve the most recent control actions from the live database.

    Returns:
        pandas.DataFrame containing action and model information.
    """

    query = """
    SELECT
        ca.action_id,
        ca.decision_id,

        md.deployment_id,
        md.ts AS decision_time,

        ca.param_id,

        cp.name AS parameter_name,
        cp.scope_type AS parameter_scope,
        cp.mechanism AS parameter_mechanism,
        cp.unit AS parameter_unit,
        cp.min_value AS parameter_min_value,
        cp.max_value AS parameter_max_value,
        cp.caveat AS parameter_caveat,

        ca.scope_type,
        ca.cell_id,
        ca.slice_id,
        ca.ue_id,
        ca.bearer_rbid,

        ca.raw_value,
        ca.requested_value,
        ca.applied_value,
        ca.clipped,

        CASE
            WHEN ca.applied_value IS NOT NULL
                THEN ca.applied_value
            ELSE ca.requested_value
        END AS effective_value,

        CASE
            WHEN ca.applied_value IS NOT NULL
                THEN 'applied'
            WHEN ca.requested_value IS NOT NULL
                THEN 'requested_fallback'
            ELSE 'missing'
        END AS value_source,

        rm.name AS model_name,
        rm.kind AS model_kind,
        rm.objective AS model_objective,

        rmv.version_id,
        rmv.model_id

    FROM public.control_action ca

    JOIN public.model_decision md
        ON ca.decision_id = md.decision_id

    JOIN public.control_param cp
        ON ca.param_id = cp.param_id

    JOIN public.model_deployment mdp
        ON md.deployment_id = mdp.deployment_id

    JOIN public.ran_model_version rmv
        ON mdp.version_id = rmv.version_id

    JOIN public.ran_model rm
        ON rmv.model_id = rm.model_id

    ORDER BY ca.action_id DESC

    LIMIT %s;
    """

    conn = get_connection()

    try:
        df = pd.read_sql_query(
            query,
            conn,
            params=(limit,)
        )
    finally:
        conn.close()

    return df


def get_new_actions(last_action_id=0, limit=100):
    """
    Retrieve actions created after the last processed action_id.

    This function will later be used by the LangGraph monitoring agent.

    Args:
        last_action_id: Highest action_id already processed.
        limit: Maximum number of new actions to retrieve.

    Returns:
        pandas.DataFrame containing only new actions.
    """

    query = """
    SELECT
        ca.action_id,
        ca.decision_id,

        md.deployment_id,
        md.ts AS decision_time,

        ca.param_id,

        cp.name AS parameter_name,
        cp.scope_type AS parameter_scope,
        cp.mechanism AS parameter_mechanism,
        cp.unit AS parameter_unit,
        cp.min_value AS parameter_min_value,
        cp.max_value AS parameter_max_value,
        cp.caveat AS parameter_caveat,

        ca.scope_type,
        ca.cell_id,
        ca.slice_id,
        ca.ue_id,
        ca.bearer_rbid,

        ca.raw_value,
        ca.requested_value,
        ca.applied_value,
        ca.clipped,

        CASE
            WHEN ca.applied_value IS NOT NULL
                THEN ca.applied_value
            ELSE ca.requested_value
        END AS effective_value,

        CASE
            WHEN ca.applied_value IS NOT NULL
                THEN 'applied'
            WHEN ca.requested_value IS NOT NULL
                THEN 'requested_fallback'
            ELSE 'missing'
        END AS value_source,

        rm.name AS model_name,
        rm.kind AS model_kind,
        rm.objective AS model_objective,

        rmv.version_id,
        rmv.model_id

    FROM public.control_action ca

    JOIN public.model_decision md
        ON ca.decision_id = md.decision_id

    JOIN public.control_param cp
        ON ca.param_id = cp.param_id

    JOIN public.model_deployment mdp
        ON md.deployment_id = mdp.deployment_id

    JOIN public.ran_model_version rmv
        ON mdp.version_id = rmv.version_id

    JOIN public.ran_model rm
        ON rmv.model_id = rm.model_id

    WHERE ca.action_id > %s

    ORDER BY ca.action_id ASC

    LIMIT %s;
    """

    conn = get_connection()

    try:
        df = pd.read_sql_query(
            query,
            conn,
            params=(last_action_id, limit)
        )
    finally:
        conn.close()

    return df


if __name__ == "__main__":

    print("\nConnecting to live PostgreSQL database...\n")

    df = get_live_actions()

    print("Successfully retrieved live control actions.")
    print(f"Number of records: {len(df)}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nLatest control actions:")

    if len(df) > 0:
        print(df.head(10).to_string(index=False))
    else:
        print("No control actions found.")

    print("\nTesting new-action retrieval...")

    if len(df) > 0:

        latest_action_id = int(df["action_id"].max())

        print(
            f"Latest action_id currently in database: "
            f"{latest_action_id}"
        )

        new_actions = get_new_actions(
            last_action_id=latest_action_id
        )

        print(
            f"New actions after {latest_action_id}: "
            f"{len(new_actions)}"
        )

    else:
        print("Cannot test new-action retrieval because no actions exist.")