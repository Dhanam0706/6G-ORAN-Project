import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import joblib
import shap


# ============================================================
# CONFIGURATION
# ============================================================

DB_HOST = os.getenv("DB_HOST", "192.168.109.154")
DB_PORT = int(os.getenv("DB_PORT", "5433"))
DB_NAME = os.getenv("DB_NAME", "rankpi")
DB_USER = os.getenv("DB_USER", "rankpi")
DB_PASSWORD = "rankpi"

DEPLOYMENT_ID = int(os.getenv("DEPLOYMENT_ID", "17"))

POLL_SECONDS = float(os.getenv("POLL_SECONDS", "5"))
POST_WINDOW_SECONDS = 1.2

MODEL_PATH = "models/conflict_xgboost_stratified.pkl"

STATE_FILE = Path("data/monitor/monitor_state.json")
RESULT_FILE = Path("data/monitor/monitor_results.csv")


# ============================================================
# DATABASE
# ============================================================

def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )


# ============================================================
# STATE
# ============================================================

def load_last_decision():
    if not STATE_FILE.exists():
        return 0

    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)

        return int(state.get("last_decision_id", 0))

    except Exception:
        return 0


def save_last_decision(decision_id):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(STATE_FILE, "w") as f:
        json.dump(
            {"last_decision_id": int(decision_id)},
            f,
            indent=2
        )


# ============================================================
# NEW DECISIONS
# ============================================================

def get_new_decisions(last_decision_id):

    query = """
        SELECT
            md.decision_id,
            md.deployment_id,
            md.step,
            md.episode,
            md.ts
        FROM model_decision md
        WHERE md.deployment_id = %s
          AND md.decision_id > %s
        ORDER BY md.decision_id ASC
    """

    conn = get_connection()

    try:
        return pd.read_sql_query(
            query,
            conn,
            params=(DEPLOYMENT_ID, last_decision_id)
        )
    finally:
        conn.close()


# ============================================================
# CONTROL ACTIONS
# ============================================================

def get_actions(decision_id):

    query = """
        SELECT
            action_id,
            decision_id,
            slice_id,
            target,
            raw_value,
            requested_value,
            applied_value,
            ts
        FROM v_control_action
        WHERE decision_id = %s
          AND param = 'dedicated_prb_ratio'
          AND implemented = TRUE
        ORDER BY slice_id
    """

    conn = get_connection()

    try:
        df = pd.read_sql_query(
            query,
            conn,
            params=(decision_id,)
        )
    finally:
        conn.close()

    if df.empty:
        return None

    # Effective action:
    # use applied_value when available,
    # otherwise requested/raw value as the controller action.
    df["action_value"] = df["applied_value"]

    df["action_value"] = df["action_value"].fillna(
        df["requested_value"]
    )

    df["action_value"] = df["action_value"].fillna(
        df["raw_value"]
    )

    return df


# ============================================================
# PREVIOUS ACTION
# ============================================================

def get_previous_action(decision_id):

    query = """
        SELECT
            decision_id,
            MAX(CASE
                WHEN target = 'slice-a'
                THEN COALESCE(applied_value,
                              requested_value,
                              raw_value)
            END) AS action_a
        FROM v_control_action
        WHERE param = 'dedicated_prb_ratio'
          AND implemented = TRUE
          AND decision_id < %s
        GROUP BY decision_id
        ORDER BY decision_id DESC
        LIMIT 1
    """

    conn = get_connection()

    try:
        df = pd.read_sql_query(
            query,
            conn,
            params=(decision_id,)
        )
    finally:
        conn.close()

    if df.empty:
        return None

    return float(df.iloc[0]["action_a"])


# ============================================================
# RAN STATE
# ============================================================

def get_ran_state(action_time, before=True):

    if before:
        query = """
            SELECT
                slice_id,
                slice_name,

                AVG(dl_mbps) AS dl_mbps,
                AVG(ul_mbps) AS ul_mbps,

                AVG(dl_prb_util) AS dl_prb_util,
                AVG(ul_prb_util) AS ul_prb_util,

                AVG(d_dl_harq_fail) AS dl_harq_fail,
                AVG(d_ul_harq_fail) AS ul_harq_fail,

                AVG(dl_retx_frac) AS dl_retx_frac,

                AVG(pusch_snr) AS pusch_snr,
                AVG(pucch_snr) AS pucch_snr,

                AVG(dl_mcs1) AS dl_mcs1,
                AVG(ul_mcs1) AS ul_mcs1,

                AVG(phr) AS phr,
                AVG(bsr) AS bsr

            FROM v_ue_rate
            WHERE ts >= (%s::timestamptz - INTERVAL '1 second')
  AND ts < %s::timestamptz
              AND slice_id IS NOT NULL

            GROUP BY slice_id, slice_name
            ORDER BY slice_id
        """

        params = (action_time, action_time)

    else:
        query = """
            SELECT
                slice_id,
                slice_name,

                AVG(dl_mbps) AS dl_mbps,
                AVG(ul_mbps) AS ul_mbps,

                AVG(dl_prb_util) AS dl_prb_util,
                AVG(ul_prb_util) AS ul_prb_util,

                AVG(d_dl_harq_fail) AS dl_harq_fail,
                AVG(d_ul_harq_fail) AS ul_harq_fail,

                AVG(dl_retx_frac) AS dl_retx_frac,

                AVG(pusch_snr) AS pusch_snr,
                AVG(pucch_snr) AS pucch_snr,

                AVG(dl_mcs1) AS dl_mcs1,
                AVG(ul_mcs1) AS ul_mcs1,

                AVG(phr) AS phr,
                AVG(bsr) AS bsr

            FROM v_ue_rate
            WHERE ts >= %s::timestamptz
  AND ts < (%s::timestamptz + INTERVAL '1 second')
              AND slice_id IS NOT NULL

            GROUP BY slice_id, slice_name
            ORDER BY slice_id
        """

        params = (action_time, action_time)

    conn = get_connection()

    try:
        return pd.read_sql_query(
            query,
            conn,
            params=params
        )
    finally:
        conn.close()


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def clip01(x):
    return np.clip(x, 0, 1)


def calculate_kis(row):

    bler_values = []

    if pd.notna(row["dl_harq_fail"]):
        bler_values.append(
            1 - clip01(row["dl_harq_fail"] / 0.5)
        )

    if pd.notna(row["ul_harq_fail"]):
        bler_values.append(
            1 - clip01(row["ul_harq_fail"] / 0.5)
        )

    if pd.notna(row["dl_retx_frac"]):
        retx_health = (
            1 - clip01(row["dl_retx_frac"] / 0.5)
        )
    else:
        retx_health = 0.5

    snr_values = []

    if pd.notna(row["pusch_snr"]):
        snr_values.append(
            clip01(row["pusch_snr"] / 50)
        )

    if pd.notna(row["pucch_snr"]):
        snr_values.append(
            clip01(row["pucch_snr"] / 50)
        )

    bler_health = (
        np.mean(bler_values)
        if bler_values
        else 0.5
    )

    snr_health = (
        np.mean(snr_values)
        if snr_values
        else 0.5
    )

    return np.mean([
        bler_health,
        retx_health,
        snr_health
    ])


def build_features(actions, before_state):

    if len(actions) < 2:
        return None

    action_a = None
    action_b = None

    for _, row in actions.iterrows():

        target = str(row["target"]).lower()

        if target == "slice-a":
            action_a = float(row["action_value"])

        elif target == "slice-b":
            action_b = float(row["action_value"])

    if action_a is None or action_b is None:
        return None

    previous_a = get_previous_action(
        int(actions.iloc[0]["decision_id"])
    )

    if previous_a is None:
        return None

    allocation_change = abs(action_a - previous_a)

    acs = allocation_change / 100.0

    ocs = abs(action_a - action_b) / 100.0

    if len(before_state) < 2:
        return None

    kis_values = []

    for _, row in before_state.iterrows():
        kis_values.append(calculate_kis(row))

    kis = float(np.mean(kis_values))

    tos_values = []

    for _, row in before_state.iterrows():

        dl = float(row["dl_mbps"] or 0)
        ul = float(row["ul_mbps"] or 0)

        tos_values.append(dl + ul)

    tos = float(np.mean(tos_values))

    ros_values = []

    for _, row in before_state.iterrows():

        dl_prb = float(row["dl_prb_util"] or 0)
        ul_prb = float(row["ul_prb_util"] or 0)

        ros_values.append(
            (dl_prb + ul_prb) / 2
        )

    ros = float(np.mean(ros_values))

    return pd.DataFrame([{
        "ROS": ros,
        "TOS": tos,
        "ACS": acs,
        "OCS": ocs,
        "KIS": kis
    }])

# ============================================================
# POST-EVENT PVS / COS / SEVERITY
# ============================================================

def calculate_severity(
    actions,
    before_state,
    after_state,
    probability
):

    if len(before_state) < 2 or len(after_state) < 2:
        return 0.0, 0.0, 0.0

    action_a = None
    action_b = None

    for _, row in actions.iterrows():

        target = str(row["target"]).lower()

        if target == "slice-a":
            action_a = float(row["action_value"])

        elif target == "slice-b":
            action_b = float(row["action_value"])

    if action_a is None or action_b is None:
        return 0.0, 0.0, 0.0

    def slice_metrics(df):

        result = {}

        for slice_id in ["slice-a", "slice-b"]:

            rows = df[
                df["slice_id"].astype(str).str.lower()
                == slice_id
            ]

            if len(rows) == 0:
                result[slice_id] = {
                    "tos": 0.0,
                    "dl_bler": 0.0,
                    "ul_bler": 0.0,
                    "retx": 0.0
                }
                continue

            dl = pd.to_numeric(
                rows["dl_mbps"],
                errors="coerce"
            ).fillna(0.0)

            ul = pd.to_numeric(
                rows["ul_mbps"],
                errors="coerce"
            ).fillna(0.0)

            dl_bler = pd.to_numeric(
                rows["dl_bler"],
                errors="coerce"
            ).fillna(0.0)

            ul_bler = pd.to_numeric(
                rows["ul_bler"],
                errors="coerce"
            ).fillna(0.0)

            retx = pd.to_numeric(
                rows["dl_retx_frac"],
                errors="coerce"
            ).fillna(0.0)

            result[slice_id] = {
                "tos": float((dl + ul).mean()),
                "dl_bler": float(dl_bler.mean()),
                "ul_bler": float(ul_bler.mean()),
                "retx": float(retx.mean())
            }

        return result

    before = slice_metrics(before_state)
    after = slice_metrics(after_state)

    def relative_loss(before_value, after_value):

        return float(
            np.clip(
                (before_value - after_value)
                / (abs(before_value) + 1e-6),
                0.0,
                1.0
            )
        )

    pvs = {}

    for slice_id in ["slice-a", "slice-b"]:

        tos_loss = relative_loss(
            before[slice_id]["tos"],
            after[slice_id]["tos"]
        )

        dl_bler_deterioration = float(
            np.clip(
                after[slice_id]["dl_bler"]
                - before[slice_id]["dl_bler"],
                0.0,
                0.5
            ) / 0.5
        )

        ul_bler_deterioration = float(
            np.clip(
                after[slice_id]["ul_bler"]
                - before[slice_id]["ul_bler"],
                0.0,
                0.5
            ) / 0.5
        )

        retx_deterioration = float(
            np.clip(
                after[slice_id]["retx"]
                - before[slice_id]["retx"],
                0.0,
                0.5
            ) / 0.5
        )

        pvs[slice_id] = (
            tos_loss
            + dl_bler_deterioration
            + ul_bler_deterioration
            + retx_deterioration
        ) / 4.0

    previous_a = get_previous_action(
        int(actions.iloc[0]["decision_id"])
    )

    if previous_a is None:
        return 0.0, 0.0, 0.0

    allocation_change = action_a - previous_a

    if allocation_change > 0:
        affected_pvs = pvs["slice-b"]

    elif allocation_change < 0:
        affected_pvs = pvs["slice-a"]

    else:
        affected_pvs = 0.0

    acs = abs(allocation_change) / 100.0

    cos = float(
        np.clip(
            acs * affected_pvs,
            0.0,
            1.0
        )
    )

    severity = 100.0 * (
        0.50 * probability
        + 0.30 * affected_pvs
        + 0.20 * cos
    )

    severity = float(
        np.clip(
            severity,
            0.0,
            100.0
        )
    )

    return affected_pvs, cos, severity


def get_severity_band(score):

    if score < 25:
        return "Low"

    elif score < 50:
        return "Moderate"

    elif score < 75:
        return "High"

    else:
        return "Critical"
# ============================================================
# MODEL + SHAP
# ============================================================

print("\nLoading XGBoost model...")

model = joblib.load(MODEL_PATH)

explainer = shap.TreeExplainer(model)

FEATURES = [
    "ROS",
    "TOS",
    "ACS",
    "OCS",
    "KIS"
]

print("Model loaded:", MODEL_PATH)


# ============================================================
# PREDICT
# ============================================================

def predict(features):

    X = features[FEATURES]

    probability = float(
        model.predict_proba(X)[0, 1]
    )

    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    values = np.asarray(shap_values)[0]

    explanation = {
        feature: float(value)
        for feature, value in zip(
            FEATURES,
            values
        )
    }

    top_feature = max(
        explanation,
        key=lambda x: abs(explanation[x])
    )

    return probability, explanation, top_feature


# ============================================================
# SAVE RESULT
# ============================================================

def save_result(result):

    RESULT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.DataFrame([result])

    if RESULT_FILE.exists():

        df.to_csv(
            RESULT_FILE,
            mode="a",
            header=False,
            index=False
        )

    else:

        df.to_csv(
            RESULT_FILE,
            index=False
        )


# ============================================================
# PROCESS ONE DECISION
# ============================================================

def process_decision(decision):

    decision_id = int(
        decision["decision_id"]
    )

    action_time = decision["ts"]

    print("\n" + "=" * 60)
    print("Processing decision:", decision_id)
    print("Action time:", action_time)

    actions = get_actions(decision_id)

    if actions is None:
        print("No implemented dedicated_prb_ratio actions.")
        return False

    print("Actions found:", len(actions))

    if len(actions) < 2:
        print("Waiting for both slice actions.")
        return False

    # Historical records are already complete.
    # For live records, wait for the after-action window.
    now = pd.Timestamp.now(tz="UTC")

    action_timestamp = pd.Timestamp(action_time)

    if action_timestamp > now:
        wait_seconds = (
            action_timestamp - now
        ).total_seconds()

        if wait_seconds > 0:
            time.sleep(wait_seconds)

    time.sleep(POST_WINDOW_SECONDS)

    before_state = get_ran_state(
        action_time,
        before=True
    )

    after_state = get_ran_state(
        action_time,
        before=False
    )

    print(
        "Before-state rows:",
        len(before_state)
    )

    print(
        "After-state rows:",
        len(after_state)
    )

    features = build_features(
        actions,
        before_state
    )

    if features is None:
        print("Could not construct complete features.")
        return False

    probability, explanation, top_feature = predict(
        features
    )
    # ========================================================
    # POST-EVENT SEVERITY
    # ========================================================

    affected_pvs, cos, severity_score = calculate_severity(
        actions,
        before_state,
        after_state,
        probability
    )

    severity_band = get_severity_band(
        severity_score
    )

    prediction = int(
        probability >= 0.50
    )

    action_a = float(
        actions.loc[
            actions["target"].str.lower() == "slice-a",
            "action_value"
        ].iloc[0]
    )

    action_b = float(
        actions.loc[
            actions["target"].str.lower() == "slice-b",
            "action_value"
        ].iloc[0]
    )

    result = {
        "decision_id": decision_id,
        "run_id": None,
        "action_time": action_time,
        "action_a": action_a,
        "action_b": action_b,

        "ROS": float(features.iloc[0]["ROS"]),
        "TOS": float(features.iloc[0]["TOS"]),
        "ACS": float(features.iloc[0]["ACS"]),
        "OCS": float(features.iloc[0]["OCS"]),
        "KIS": float(features.iloc[0]["KIS"]),

        "conflict_probability": probability,
        "prediction": prediction,

        "top_shap_feature": top_feature,
         
                 "shap_ROS": explanation["ROS"],
        "shap_TOS": explanation["TOS"],
        "shap_ACS": explanation["ACS"],
        "shap_OCS": explanation["OCS"],
        "shap_KIS": explanation["KIS"],

        "affected_PVS": affected_pvs,
        "COS": cos,
        "severity_score": severity_score,
        "severity_band": severity_band
}

    save_result(result)

    print("\nPrediction")
    print("-" * 40)
    print("Conflict probability:", round(probability, 4))
    print("Prediction:", prediction)
    print("Top SHAP feature:", top_feature)
    print("\nSeverity")
    print("-" * 40)
    print("Affected PVS:", round(affected_pvs, 4))
    print("COS:", round(cos, 4))
    print("Severity score:", round(severity_score, 2), "/ 100")
    print("Severity level:", severity_band)
    print("\nFeatures")
    print(features.to_string(index=False))

    print("\nSHAP")
    for feature, value in explanation.items():
        print(
            f"{feature:>5}: {value:+.4f}"
        )

    print("\nResult saved to:")
    print(RESULT_FILE)

    return True


# ============================================================
# MAIN MONITOR
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("AUTONOMOUS O-RAN CONFLICT MONITOR")
    print("=" * 60)

    print("Database:", DB_HOST)
    print("Port:", DB_PORT)
    print("Database name:", DB_NAME)
    print("Deployment:", DEPLOYMENT_ID)
    print("Polling:", POLL_SECONDS, "seconds")

    last_decision = load_last_decision()

    print(
        "Starting after decision:",
        last_decision
    )

    while True:

        try:

            decisions = get_new_decisions(
                last_decision
            )

            if decisions.empty:

                print(
                    f"\rMonitoring... "
                    f"last decision = {last_decision}",
                    end="",
                    flush=True
                )

                time.sleep(POLL_SECONDS)
                continue

            print(
                "\n\nNew decisions detected:",
                len(decisions)
            )

            for _, decision in decisions.iterrows():

                decision_id = int(
                    decision["decision_id"]
                )

                try:

                    process_decision(
                        decision
                    )

                except Exception as e:

                    print(
                        "\nERROR processing decision",
                        decision_id,
                        ":",
                        repr(e)
                    )

                # Advance watermark even if one event fails,
                # so the monitor does not repeatedly process
                # the same broken record forever.
                last_decision = decision_id

                save_last_decision(
                    last_decision
                )

        except KeyboardInterrupt:

            print(
                "\n\nMonitor stopped by user."
            )
            break

        except Exception as e:

            print(
                "\nMonitor error:",
                repr(e)
            )

            print(
                "Retrying in",
                POLL_SECONDS,
                "seconds..."
            )

            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
