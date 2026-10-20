import pandas as pd

from src.db_reader import get_connection


def read_ran_state(start_time, end_time, run_id):
    """
    Read RAN measurements from v_ue_rate
    for one time window and one run.
    """

    query = """
        SELECT
            ts,
            run_id,
            cell_id,
            rnti,
            session_id,
            ue_id,
            slice_id,
            slice_name,
            dl_mbps,
            ul_mbps,
            dl_prb_util,
            ul_prb_util,
            dl_bler,
            ul_bler,
            dl_retx_frac,
            pusch_snr,
            pucch_snr,
            dl_mcs1,
            ul_mcs1,
            phr,
            bsr
        FROM public.v_ue_rate
        WHERE run_id = %s
          AND slice_id IS NOT NULL
          AND ts >= %s
          AND ts < %s
        ORDER BY ts, slice_id, ue_id;
    """

    conn = get_connection()

    try:
        df = pd.read_sql_query(
            query,
            conn,
            params=[run_id, start_time, end_time]
        )
    finally:
        conn.close()

    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)

    return df


def aggregate_slice_state(ran):
    """
    Convert UE-level RAN measurements into
    slice-level state.
    """

    if ran.empty:
        return pd.DataFrame()

    # First aggregate measurements for each UE.
    ue_state = (
        ran.groupby(
            ["slice_id", "slice_name", "ue_id"],
            dropna=True
        )
        .agg(
            dl_mbps=("dl_mbps", "mean"),
            ul_mbps=("ul_mbps", "mean"),
            dl_prb_util=("dl_prb_util", "mean"),
            ul_prb_util=("ul_prb_util", "mean"),
            dl_bler=("dl_bler", "mean"),
            ul_bler=("ul_bler", "mean"),
            dl_retx_frac=("dl_retx_frac", "mean"),
            pusch_snr=("pusch_snr", "mean"),
            pucch_snr=("pucch_snr", "mean"),
            dl_mcs1=("dl_mcs1", "mean"),
            ul_mcs1=("ul_mcs1", "mean"),
            phr=("phr", "mean"),
            bsr=("bsr", "mean"),
        )
        .reset_index()
    )

    # Then aggregate UE states into slice state.
    slice_state = (
        ue_state.groupby(
            ["slice_id", "slice_name"],
            dropna=True
        )
        .agg(
            dl_mbps=("dl_mbps", "sum"),
            ul_mbps=("ul_mbps", "sum"),
            dl_prb_util=("dl_prb_util", "mean"),
            ul_prb_util=("ul_prb_util", "mean"),
            dl_bler=("dl_bler", "mean"),
            ul_bler=("ul_bler", "mean"),
            dl_retx_frac=("dl_retx_frac", "mean"),
            pusch_snr=("pusch_snr", "mean"),
            pucch_snr=("pucch_snr", "mean"),
            dl_mcs1=("dl_mcs1", "mean"),
            ul_mcs1=("ul_mcs1", "mean"),
            phr=("phr", "mean"),
            bsr=("bsr", "mean"),
        )
        .reset_index()
    )

    return slice_state


def get_decision(decision_id):
    """
    Read one controller decision and obtain its run_id
    through model_deployment.
    """

    query = """
        SELECT
            md.decision_id,
            md.deployment_id,
            md.step,
            md.episode,
            md.ts,
            dep.run_id
        FROM public.model_decision md
        JOIN public.model_deployment dep
          ON md.deployment_id = dep.deployment_id
        WHERE md.decision_id = %s;
    """

    conn = get_connection()

    try:
        decision = pd.read_sql_query(
            query,
            conn,
            params=[decision_id]
        )
    finally:
        conn.close()

    if decision.empty:
        raise ValueError(
            f"Decision {decision_id} was not found."
        )

    decision["ts"] = pd.to_datetime(
        decision["ts"],
        utc=True
    )

    return decision.iloc[0]


def get_actions(decision_id):
    """
    Read all controller actions belonging to one decision.
    """

    query = """
        SELECT
            action_id,
            decision_id,
            deployment_id,
            run_id,
            model_name,
            ran_model_version,
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
        WHERE decision_id = %s
        ORDER BY action_id;
    """

    conn = get_connection()

    try:
        actions = pd.read_sql_query(
            query,
            conn,
            params=[decision_id]
        )
    finally:
        conn.close()

    if not actions.empty:
        actions["ts"] = pd.to_datetime(
            actions["ts"],
            utc=True
        )

    return actions


def effective_action_value(row):
    """
    Use applied_value when available.
    Otherwise fall back to requested_value.
    """

    if pd.notna(row["applied_value"]):
        return row["applied_value"]

    return row["requested_value"]


def align_decision(decision_id, window_seconds=1):
    """
    Build BEFORE and AFTER slice states around
    one real controller decision.

    BEFORE = [t-1s, t)
    AFTER  = [t, t+1s)
    """

    decision = get_decision(decision_id)
    actions = get_actions(decision_id)

    action_time = decision["ts"]
    run_id = int(decision["run_id"])

    before_start = action_time - pd.Timedelta(
        seconds=window_seconds
    )

    before_end = action_time

    after_start = action_time

    after_end = action_time + pd.Timedelta(
        seconds=window_seconds
    )

    # print("\nDecision alignment")
# print("------------------")
# print(f"Decision ID : {decision_id}")
# print(f"Run ID      : {run_id}")
# print(f"Action time : {action_time}")
# print(f"BEFORE      : {before_start} → {before_end}")
# print(f"AFTER       : {after_start} → {after_end}")

    # Read the two windows.
    before_ran = read_ran_state(
        before_start,
        before_end,
        run_id
    )

    after_ran = read_ran_state(
        after_start,
        after_end,
        run_id
    )

    # Aggregate to slice level.
    before_state = aggregate_slice_state(
        before_ran
    )

    after_state = aggregate_slice_state(
        after_ran
    )

    # Prefix columns so BEFORE and AFTER can be merged.
    before_state = before_state.rename(
        columns={
            col: f"before_{col}"
            for col in before_state.columns
            if col not in ["slice_id", "slice_name"]
        }
    )

    after_state = after_state.rename(
        columns={
            col: f"after_{col}"
            for col in after_state.columns
            if col not in ["slice_id", "slice_name"]
        }
    )

    aligned = pd.merge(
        before_state,
        after_state,
        on=["slice_id", "slice_name"],
        how="outer"
    )

    # Add decision information.
    aligned["decision_id"] = decision_id
    aligned["run_id"] = run_id
    aligned["action_time"] = action_time

    # Add action information.
    if not actions.empty:
        actions["effective_value"] = actions.apply(
            effective_action_value,
            axis=1
        )

        action_values = (
            actions.groupby("slice_id")["effective_value"]
            .first()
            .to_dict()
        )

        aligned["action_value"] = aligned[
            "slice_id"
        ].map(action_values)

    # Calculate state changes.
    state_columns = [
        "dl_mbps",
        "ul_mbps",
        "dl_prb_util",
        "ul_prb_util",
        "dl_bler",
        "ul_bler",
        "dl_retx_frac",
        "pusch_snr",
        "pucch_snr",
        "dl_mcs1",
        "ul_mcs1",
        "phr",
        "bsr",
    ]

    for column in state_columns:
        before_col = f"before_{column}"
        after_col = f"after_{column}"
        delta_col = f"delta_{column}"

        if (
            before_col in aligned.columns
            and after_col in aligned.columns
        ):
            aligned[delta_col] = (
                aligned[after_col]
                - aligned[before_col]
            )

    return aligned, actions
def get_decisions(run_id):
    query = """
        SELECT
            md.decision_id,
            md.deployment_id,
            md.step,
            md.episode,
            md.ts,
            dep.run_id
        FROM public.model_decision md
        JOIN public.model_deployment dep
          ON md.deployment_id = dep.deployment_id
        WHERE dep.run_id = %s
        ORDER BY md.ts;
    """

    conn = get_connection()

    try:
        decisions = pd.read_sql_query(
            query,
            conn,
            params=[run_id]
        )
    finally:
        conn.close()

    if decisions.empty:
        raise ValueError(f"No decisions found for run {run_id}.")

    decisions["ts"] = pd.to_datetime(
        decisions["ts"],
        utc=True
    )

    return decisions
def align_all_decisions(run_id, window_seconds=1):
    """
    Efficient batch alignment for all controller decisions.

    Loads:
        1. All decisions
        2. All control actions
        3. All RAN state data

    once, then performs BEFORE/AFTER temporal alignment
    locally using pandas.
    """

    decisions = get_decisions(run_id)

    if decisions.empty:
        return pd.DataFrame()

    total = len(decisions)

    print(f"\nFound {total} decisions for run {run_id}")
    print("Loading RAN state for batch alignment...")

    # ---------------------------------------------------------
    # Determine the complete time range required.
    # ---------------------------------------------------------

    first_time = decisions["ts"].min()
    last_time = decisions["ts"].max()

    query_start = first_time - pd.Timedelta(seconds=window_seconds)
    query_end = last_time + pd.Timedelta(seconds=window_seconds)

    # ---------------------------------------------------------
    # Load all RAN state in ONE database query.
    # ---------------------------------------------------------

    ran = read_ran_state(
        query_start,
        query_end,
        run_id
    )

    if ran.empty:
        print("No RAN state found for this run.")
        return pd.DataFrame()

    print(f"Loaded {len(ran)} RAN state rows.")

    # ---------------------------------------------------------
    # Aggregate RAN state to slice level.
    # ---------------------------------------------------------

    ran["ts"] = pd.to_datetime(
        ran["ts"],
        utc=True
    )

    state_columns = [
        "dl_mbps",
        "ul_mbps",
        "dl_prb_util",
        "ul_prb_util",
        "dl_bler",
        "ul_bler",
        "dl_retx_frac",
        "pusch_snr",
        "pucch_snr",
        "dl_mcs1",
        "ul_mcs1",
        "phr",
        "bsr",
    ]

    # ---------------------------------------------------------
    # Load all control actions for this run.
    # ---------------------------------------------------------

    conn = get_connection()

    action_query = """
        SELECT
            ca.action_id,
            ca.decision_id,
            ca.deployment_id,
            ca.run_id,
            ca.ts,
            ca.param,
            ca.unit,
            ca.implemented,
            ca.scope_type,
            ca.target,
            ca.slice_id,
            ca.ue_id,
            ca.cell_id,
            ca.raw_value,
            ca.requested_value,
            ca.applied_value,
            ca.clipped,
            ca.score
        FROM public.v_control_action ca
        WHERE ca.run_id = %s
          AND ca.param = 'dedicated_prb_ratio'
        ORDER BY ca.ts;
    """

    try:
        actions = pd.read_sql_query(
            action_query,
            conn,
            params=[run_id]
        )
    finally:
        conn.close()

    if not actions.empty:
        actions["ts"] = pd.to_datetime(
            actions["ts"],
            utc=True
        )

        actions["effective_value"] = actions.apply(
            effective_action_value,
            axis=1
        )

    print(f"Loaded {len(actions)} control actions.")

    # ---------------------------------------------------------
    # Pre-aggregate RAN state by slice.
    # ---------------------------------------------------------

    print("Aggregating RAN state by slice...")

    ue_state = (
        ran.groupby(
            ["slice_id", "slice_name", "ue_id"],
            dropna=True
        )
        .agg(
            dl_mbps=("dl_mbps", "mean"),
            ul_mbps=("ul_mbps", "mean"),
            dl_prb_util=("dl_prb_util", "mean"),
            ul_prb_util=("ul_prb_util", "mean"),
            dl_bler=("dl_bler", "mean"),
            ul_bler=("ul_bler", "mean"),
            dl_retx_frac=("dl_retx_frac", "mean"),
            pusch_snr=("pusch_snr", "mean"),
            pucch_snr=("pucch_snr", "mean"),
            dl_mcs1=("dl_mcs1", "mean"),
            ul_mcs1=("ul_mcs1", "mean"),
            phr=("phr", "mean"),
            bsr=("bsr", "mean"),
        )
        .reset_index()
    )

    # Keep the original timestamp by merging at the UE-state level.
    # Rebuild the aggregation with timestamp included so that
    # BEFORE/AFTER windows remain temporally meaningful.

    ran["time_bucket"] = ran["ts"]

    slice_state = (
        ran.groupby(
            ["ts", "slice_id", "slice_name"],
            dropna=True
        )
        .agg(
            dl_mbps=("dl_mbps", "sum"),
            ul_mbps=("ul_mbps", "sum"),
            dl_prb_util=("dl_prb_util", "mean"),
            ul_prb_util=("ul_prb_util", "mean"),
            dl_bler=("dl_bler", "mean"),
            ul_bler=("ul_bler", "mean"),
            dl_retx_frac=("dl_retx_frac", "mean"),
            pusch_snr=("pusch_snr", "mean"),
            pucch_snr=("pucch_snr", "mean"),
            dl_mcs1=("dl_mcs1", "mean"),
            ul_mcs1=("ul_mcs1", "mean"),
            phr=("phr", "mean"),
            bsr=("bsr", "mean"),
        )
        .reset_index()
        .sort_values("ts")
    )

    # ---------------------------------------------------------
    # Align every decision.
    # ---------------------------------------------------------

    aligned_rows = []

    print("Starting temporal alignment...\n")

    for count, (_, decision) in enumerate(
        decisions.iterrows(),
        start=1
    ):
        decision_id = int(decision["decision_id"])
        action_time = decision["ts"]

        before_start = (
            action_time
            - pd.Timedelta(seconds=window_seconds)
        )

        after_end = (
            action_time
            + pd.Timedelta(seconds=window_seconds)
        )

        # BEFORE = [t-window, t)
        before = slice_state[
            (slice_state["ts"] >= before_start)
            & (slice_state["ts"] < action_time)
        ]

        # AFTER = [t, t+window)
        after = slice_state[
            (slice_state["ts"] >= action_time)
            & (slice_state["ts"] < after_end)
        ]

        if before.empty or after.empty:
            continue

        # Aggregate the selected temporal windows to slice level.
        before = (
            before.groupby(
                ["slice_id", "slice_name"],
                dropna=True
            )[state_columns]
            .mean()
            .reset_index()
        )

        after = (
            after.groupby(
                ["slice_id", "slice_name"],
                dropna=True
            )[state_columns]
            .mean()
            .reset_index()
        )

        # Prefix BEFORE columns.
        before = before.rename(
            columns={
                col: f"before_{col}"
                for col in state_columns
            }
        )

        # Prefix AFTER columns.
        after = after.rename(
            columns={
                col: f"after_{col}"
                for col in state_columns
            }
        )

        aligned = pd.merge(
            before,
            after,
            on=["slice_id", "slice_name"],
            how="inner"
        )

        if aligned.empty:
            continue

        aligned["decision_id"] = decision_id
        aligned["run_id"] = run_id
        aligned["action_time"] = action_time

        # -----------------------------------------------------
        # Attach the control action.
        # -----------------------------------------------------

        decision_actions = actions[
            actions["decision_id"] == decision_id
        ]

        if not decision_actions.empty:

            action_values = (
                decision_actions
                .groupby("slice_id")["effective_value"]
                .first()
                .to_dict()
            )

            aligned["action_value"] = (
                aligned["slice_id"]
                .map(action_values)
            )

        else:
            aligned["action_value"] = pd.NA

        # -----------------------------------------------------
        # Calculate BEFORE → AFTER changes.
        # -----------------------------------------------------

        for column in state_columns:

            before_col = f"before_{column}"
            after_col = f"after_{column}"

            aligned[f"delta_{column}"] = (
                aligned[after_col]
                - aligned[before_col]
            )

        aligned_rows.append(aligned)

        if count % 100 == 0:
            print(
                f"Processed {count}/{total} decisions"
            )

    # ---------------------------------------------------------
    # Combine everything.
    # ---------------------------------------------------------

    if not aligned_rows:
        print("No decisions could be aligned.")
        return pd.DataFrame()

    result = pd.concat(
        aligned_rows,
        ignore_index=True
    )

    # Put decision information first.
    first_columns = [
        "decision_id",
        "run_id",
        "action_time",
        "slice_id",
        "slice_name",
        "action_value",
    ]

    remaining_columns = [
        col
        for col in result.columns
        if col not in first_columns
    ]

    result = result[
        first_columns + remaining_columns
    ]

    print(
        f"\nBatch alignment complete."
    )
    print(
        f"Aligned rows: {len(result)}"
    )
    print(
        f"Unique decisions: "
        f"{result['decision_id'].nunique()}"
    )

    return result


if __name__ == "__main__":
    RUN_ID = 7

    output_path = "data/processed/aligned_decisions.csv"

    aligned = align_all_decisions(
        RUN_ID,
        window_seconds=1
    )

    aligned.to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 60)
    print("BATCH ALIGNMENT COMPLETE")
    print("=" * 60)

    print(f"Rows: {len(aligned)}")
    print(f"Columns: {len(aligned.columns)}")
    print(f"Saved to: {output_path}")

    print("\nFirst rows:")
    print(
        aligned.head(10).to_string(index=False)
    )