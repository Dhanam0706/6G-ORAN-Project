from dotenv import load_dotenv
import os
import json
import psycopg2
import pandas as pd
load_dotenv()


def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", "5432"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )


def read_control_actions(limit=1000):
    query = """
        SELECT
            action_id,
            decision_id,
            deployment_id,
            run_id,
            model_name,
            ran_model_version,
            mode,
            step,
            episode,
            ts,
            param,
            unit,
            implemented,
            scope_type,
            target,
            slice_id,
            ue_id,
            cell_id,
            raw_value,
            requested_value,
            applied_value,
            clipped,
            score
        FROM public.v_control_action
        ORDER BY ts, action_id
        LIMIT {limit};
    """

    conn = get_connection()

    try:
        return pd.read_sql_query(query.format(limit=int(limit)), conn)
    finally:
        conn.close()


def read_decisions(limit=1000):
    query = """
        SELECT
            decision_id,
            deployment_id,
            step,
            episode,
            ts,
            obs_from,
            obs_to,
            obs_norm,
            obs_raw,
            score,
            inference_ms
        FROM public.model_decision
        ORDER BY ts, decision_id
        LIMIT {limit};
    """

    conn = get_connection()

    try:
        return pd.read_sql_query(
    query.format(limit=int(limit)),
    conn
)
    finally:
        conn.close()


if __name__ == "__main__":
    print("Reading control actions...")

    actions = read_control_actions(10)

    print("\nControl actions:")
    print(actions.to_string(index=False))

    print("\nReading model decisions...")

    decisions = read_decisions(5)

    print("\nModel decisions:")
    print(decisions[
        [
            "decision_id",
            "deployment_id",
            "step",
            "ts",
            "score"
        ]
    ].to_string(index=False))