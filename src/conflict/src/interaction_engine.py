import pandas as pd


def prepare_actions(actions):
    """
    Clean and prepare control-action data.
    """

    actions = actions.copy()

    actions["ts"] = pd.to_datetime(actions["ts"], utc=True)

    numeric_columns = [
        "raw_value",
        "requested_value",
        "applied_value",
        "score"
    ]

    for column in numeric_columns:
        actions[column] = pd.to_numeric(
            actions[column],
            errors="coerce"
        )

    # The database may contain NULL applied values.
    # Until an applied value exists, requested_value is
    # the best available representation of the requested control.
    actions["effective_value"] = actions["applied_value"].fillna(
        actions["requested_value"]
    )

    return actions


def group_actions_by_decision(actions):
    """
    Group actions belonging to the same model decision.
    """

    return (
        actions
        .sort_values(["decision_id", "ts", "action_id"])
        .groupby("decision_id", sort=False)
    )


def detect_direct_conflicts(actions):
    """
    Initial direct-conflict detector.

    For the currently implemented dedicated_prb_ratio control,
    each decision allocates a shared 100% resource between slices.

    This function identifies:
      1. same decision
      2. same control parameter
      3. multiple targets
      4. competing allocation relationship
    """

    actions = prepare_actions(actions)

    conflicts = []

    for decision_id, group in group_actions_by_decision(actions):

        # Only implemented controls are considered.
        group = group[group["implemented"] == True]

        if group.empty:
            continue

        # Currently active control parameter.
        prb_actions = group[
            group["param"] == "dedicated_prb_ratio"
        ]

        if len(prb_actions) < 2:
            continue

        total_requested = prb_actions[
            "effective_value"
        ].sum()

        targets = list(prb_actions["target"].astype(str))

        # The parameter definition says the allocations
        # sum to 100%.
        allocation_error = abs(total_requested - 100.0)

        conflict = {
            "decision_id": decision_id,
            "conflict_type": "direct",
            "parameter": "dedicated_prb_ratio",
            "targets": ",".join(targets),
            "total_allocation": total_requested,
            "allocation_error": allocation_error,
            "num_actions": len(prb_actions),
        }

        # This is deliberately NOT our final conflict label.
        # It is an interaction candidate that will later be
        # combined with RAN-state evidence.
        conflict["direct_candidate"] = (
            len(set(targets)) > 1
        )

        conflicts.append(conflict)

    return pd.DataFrame(conflicts)


if __name__ == "__main__":
    from db_reader import read_control_actions

    print("Reading live control actions...")

    actions = read_control_actions(1000)

    print(f"Loaded {len(actions)} actions.")

    print("\nDetecting direct interaction candidates...")

    conflicts = detect_direct_conflicts(actions)

    if conflicts.empty:
        print("No direct interaction candidates found.")
    else:
        print("\nDirect interaction candidates:")
        print(conflicts.head(20).to_string(index=False))