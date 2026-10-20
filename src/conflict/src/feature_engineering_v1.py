import pandas as pd
import numpy as np


INPUT_FILE = "data/processed/aligned_decisions.csv"
OUTPUT_FILE = "data/processed/conflict_features_real.csv"


def safe_minmax(series):
    """
    Min-max normalization to [0, 1].

    If the series has no variation, return zeros.
    """
    minimum = series.min()
    maximum = series.max()

    if pd.isna(minimum) or pd.isna(maximum):
        return pd.Series(0.0, index=series.index)

    if maximum == minimum:
        return pd.Series(0.0, index=series.index)

    return (series - minimum) / (maximum - minimum)


def normalize_positive_change(series):
    """
    Normalize the magnitude of a positive quantity.
    """
    return safe_minmax(series.clip(lower=0))


def build_features(df):
    """
    Convert two slice-level rows per decision into
    one decision-level feature row.
    """

    df = df.copy()

    # ---------------------------------------------------------
    # Separate the two slices.
    # ---------------------------------------------------------

    a = (
        df[df["slice_name"] == "slice-a"]
        .set_index("decision_id")
        .sort_index()
    )

    b = (
        df[df["slice_name"] == "slice-b"]
        .set_index("decision_id")
        .sort_index()
    )

    common = a.index.intersection(b.index)

    a = a.loc[common]
    b = b.loc[common]

    features = pd.DataFrame(index=common)

    # ---------------------------------------------------------
    # Basic decision information.
    # ---------------------------------------------------------

    features["decision_id"] = common
    features["run_id"] = a["run_id"].values
    features["action_time"] = a["action_time"].values

    # ---------------------------------------------------------
    # Action values.
    # ---------------------------------------------------------

    features["action_a"] = a["action_value"].values
    features["action_b"] = b["action_value"].values

    # ---------------------------------------------------------
    # ACTION CHANGE
    #
    # Current allocation compared with previous decision.
    # ---------------------------------------------------------

    action_a = features["action_a"]

    features["allocation_change"] = (
        action_a.diff().abs().fillna(0)
    )

    features["ACS"] = (
        features["allocation_change"] / 100.0
    )

    # ---------------------------------------------------------
    # OPPOSING RESOURCE COMPETITION
    #
    # Because action_a + action_b = 100.
    # ---------------------------------------------------------

    features["OCS"] = (
        (features["action_a"] - features["action_b"]).abs()
        / 100.0
    )

    # ---------------------------------------------------------
    # RESOURCE OCCUPANCY / PRESSURE
    # ---------------------------------------------------------

    ros_a = (
        a["before_dl_prb_util"] +
        a["before_ul_prb_util"]
    ) / 2.0

    ros_b = (
        b["before_dl_prb_util"] +
        b["before_ul_prb_util"]
    ) / 2.0

    features["ROS_a"] = ros_a.values
    features["ROS_b"] = ros_b.values

    features["ROS"] = (
        (ros_a + ros_b) / 2.0
    ).values

    # ---------------------------------------------------------
    # THROUGHPUT / SERVICE OUTPUT
    # ---------------------------------------------------------

    tos_a_before = (
        a["before_dl_mbps"] +
        a["before_ul_mbps"]
    )

    tos_b_before = (
        b["before_dl_mbps"] +
        b["before_ul_mbps"]
    )

    tos_a_after = (
        a["after_dl_mbps"] +
        a["after_ul_mbps"]
    )

    tos_b_after = (
        b["after_dl_mbps"] +
        b["after_ul_mbps"]
    )

    features["TOS_a_before"] = tos_a_before.values
    features["TOS_b_before"] = tos_b_before.values

    features["TOS_a_after"] = tos_a_after.values
    features["TOS_b_after"] = tos_b_after.values

    features["delta_TOS_a"] = (
        tos_a_after - tos_a_before
    ).values

    features["delta_TOS_b"] = (
        tos_b_after - tos_b_before
    ).values

    features["TOS"] = (
        (tos_a_before + tos_b_before) / 2.0
    ).values

    # ---------------------------------------------------------
    # RELIABILITY / LINK INTEGRITY
    # ---------------------------------------------------------

    bler_a = (
        a["before_dl_bler"] +
        a["before_ul_bler"]
    ) / 2.0

    bler_b = (
        b["before_dl_bler"] +
        b["before_ul_bler"]
    ) / 2.0

    retx_a = a["before_dl_retx_frac"]
    retx_b = b["before_dl_retx_frac"]

    snr_a = (
        a["before_pusch_snr"] +
        a["before_pucch_snr"]
    ) / 2.0

    snr_b = (
        b["before_pusch_snr"] +
        b["before_pucch_snr"]
    ) / 2.0

    # Higher BLER / retransmission = worse.
    # Higher SNR = better.

    bler_score = (
        1.0 - safe_minmax(
            pd.concat([bler_a, bler_b])
        )
    )

    retx_score = (
        1.0 - safe_minmax(
            pd.concat([retx_a, retx_b])
        )
    )

    snr_score = safe_minmax(
        pd.concat([snr_a, snr_b])
    )

    # Split normalized values back to decisions.
    n = len(features)

    bler_a_score = bler_score.iloc[:n].values
    bler_b_score = bler_score.iloc[n:].values

    retx_a_score = retx_score.iloc[:n].values
    retx_b_score = retx_score.iloc[n:].values

    snr_a_score = snr_score.iloc[:n].values
    snr_b_score = snr_score.iloc[n:].values

    features["KIS_a"] = (
        bler_a_score +
        retx_a_score +
        snr_a_score
    ) / 3.0

    features["KIS_b"] = (
        bler_b_score +
        retx_b_score +
        snr_b_score
    ) / 3.0

    features["KIS"] = (
        features["KIS_a"] +
        features["KIS_b"]
    ) / 2.0

    # ---------------------------------------------------------
    # PERFORMANCE VARIATION
    # ---------------------------------------------------------

    # Throughput deterioration.
    throughput_loss_a = (
        -features["delta_TOS_a"]
    ).clip(lower=0)

    throughput_loss_b = (
        -features["delta_TOS_b"]
    ).clip(lower=0)

    # BLER deterioration.
    bler_delta_a = (
        a["delta_dl_bler"] +
        a["delta_ul_bler"]
    ) / 2.0

    bler_delta_b = (
        b["delta_dl_bler"] +
        b["delta_ul_bler"]
    ) / 2.0

    # Retransmission deterioration.
    retx_delta_a = a["delta_dl_retx_frac"]
    retx_delta_b = b["delta_dl_retx_frac"]

    # Normalize each adverse component globally.
    throughput_loss = safe_minmax(
        pd.concat([
            throughput_loss_a,
            throughput_loss_b
        ])
    )

    bler_loss = safe_minmax(
        pd.concat([
            bler_delta_a.clip(lower=0),
            bler_delta_b.clip(lower=0)
        ])
    )

    retx_loss = safe_minmax(
        pd.concat([
            retx_delta_a.clip(lower=0),
            retx_delta_b.clip(lower=0)
        ])
    )

    throughput_a_score = throughput_loss.iloc[:n].values
    throughput_b_score = throughput_loss.iloc[n:].values

    bler_a_score = bler_loss.iloc[:n].values
    bler_b_score = bler_loss.iloc[n:].values

    retx_a_score = retx_loss.iloc[:n].values
    retx_b_score = retx_loss.iloc[n:].values

    pvs_a = (
        throughput_a_score +
        bler_a_score +
        retx_a_score
    ) / 3.0

    pvs_b = (
        throughput_b_score +
        bler_b_score +
        retx_b_score
    ) / 3.0

    features["PVS_a"] = pvs_a
    features["PVS_b"] = pvs_b

    features["PVS"] = (
        features["PVS_a"] +
        features["PVS_b"]
    ) / 2.0

    # ---------------------------------------------------------
    # CROSS-SLICE CONFLICT EVIDENCE
    # ---------------------------------------------------------

    allocation_direction = np.sign(
        features["action_a"].diff()
        .fillna(0)
    )

    # Slice-a gains resources when allocation_direction > 0.
    # Slice-b loses resources at the same time.
    #
    # We therefore look for:
    #
    # A gains + B performance worsens
    # OR
    # B gains + A performance worsens

    pvs_a_series = pd.Series(
        features["PVS_a"].values,
        index=features.index
    )

    pvs_b_series = pd.Series(
        features["PVS_b"].values,
        index=features.index
    )

    affected_slice_loss = np.where(
        allocation_direction > 0,
        pvs_b_series,
        np.where(
            allocation_direction < 0,
            pvs_a_series,
            0.0
        )
    )

    features["COS"] = (
        features["ACS"].values *
        np.asarray(affected_slice_loss)
    )

    # ---------------------------------------------------------
    # Clean numeric values.
    # ---------------------------------------------------------

    features = features.replace(
        [np.inf, -np.inf],
        np.nan
    )

    features = features.fillna(0)

    # ---------------------------------------------------------
    # Keep useful ordering.
    # ---------------------------------------------------------

    ordered_columns = [
        "decision_id",
        "run_id",
        "action_time",

        "action_a",
        "action_b",
        "allocation_change",

        "ROS",
        "ROS_a",
        "ROS_b",

        "TOS",
        "TOS_a_before",
        "TOS_b_before",
        "TOS_a_after",
        "TOS_b_after",
        "delta_TOS_a",
        "delta_TOS_b",

        "ACS",
        "OCS",

        "KIS",
        "KIS_a",
        "KIS_b",

        "PVS",
        "PVS_a",
        "PVS_b",

        "COS",
    ]

    features = features[ordered_columns]

    return features.reset_index(drop=True)


def main():

    print("Loading aligned live-data decisions...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Input shape: {df.shape}")

    features = build_features(df)

    print(f"Feature shape: {features.shape}")

    features.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nSaved real feature dataset to:\n"
        f"{OUTPUT_FILE}"
    )

    print("\nFeature statistics:")
    print(
        features[
            [
                "ROS",
                "TOS",
                "ACS",
                "OCS",
                "KIS",
                "PVS",
                "COS",
            ]
        ].describe()
    )


if __name__ == "__main__":
    main()