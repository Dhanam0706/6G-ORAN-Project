import os
import random
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

NUM_RECORDS = 2000
OUTPUT_FILE = "data/raw/xapp_network_data.csv"

random.seed(42)
np.random.seed(42)


# ---------------------------------------------------------
# xApp definitions
# ---------------------------------------------------------

XAPPS = {
    "Traffic_Steering": {
        "objective": "throughput",
        "priority": 3
    },
    "Energy_Saving": {
        "objective": "energy",
        "priority": 2
    },
    "Load_Balancing": {
        "objective": "load_balancing",
        "priority": 2
    },
    "Mobility_Optimization": {
        "objective": "mobility",
        "priority": 3
    }
}


CELLS = [
    "Cell-101",
    "Cell-102",
    "Cell-103",
    "Cell-104",
    "Cell-105"
]

SLICES = [
    "eMBB",
    "URLLC",
    "mMTC"
]

RESOURCES = [
    "PRB",
    "POWER",
    "HANDOVER",
    "QOS"
]

ACTION_TYPES = [
    "increase",
    "decrease"
]


# ---------------------------------------------------------
# Generate one record
# ---------------------------------------------------------

def generate_record(record_id):

    # Select two different xApps
    xapp_a, xapp_b = random.sample(
        list(XAPPS.keys()),
        2
    )

    # -----------------------------------------------------
    # Spatial/resource context
    # -----------------------------------------------------

    cell_a = random.choice(CELLS)
    cell_b = random.choice(CELLS)

    slice_a = random.choice(SLICES)
    slice_b = random.choice(SLICES)

    resource_a = random.choice(RESOURCES)
    resource_b = random.choice(RESOURCES)

    same_cell = int(cell_a == cell_b)
    same_slice = int(slice_a == slice_b)
    same_resource = int(resource_a == resource_b)

    # Use the first xApp's context as the main record context
    cell = cell_a
    slice_id = slice_a
    resource = resource_a

    # -----------------------------------------------------
    # Actions
    # -----------------------------------------------------

    magnitude_a = round(
        random.uniform(5, 30),
        2
    )

    magnitude_b = round(
        random.uniform(5, 30),
        2
    )

    action_a = random.choice(ACTION_TYPES)
    action_b = random.choice(ACTION_TYPES)

    delta_a = (
        magnitude_a
        if action_a == "increase"
        else -magnitude_a
    )

    delta_b = (
        magnitude_b
        if action_b == "increase"
        else -magnitude_b
    )

    # Opposite actions
    direct_conflict = int(
        delta_a * delta_b < 0
    )

    # -----------------------------------------------------
    # Temporal information
    # -----------------------------------------------------

    start_a = random.randint(0, 1430)
    duration_a = random.randint(1, 30)
    end_a = min(
        start_a + duration_a,
        1439
    )

    start_b = random.randint(0, 1430)
    duration_b = random.randint(1, 30)
    end_b = min(
        start_b + duration_b,
        1439
    )

    # Actual temporal overlap
    temporal_overlap = int(
        start_a < end_b and start_b < end_a
    )

    # Store a combined interval for compatibility
    start_minute = min(start_a, start_b)
    end_minute = max(end_a, end_b)

    # -----------------------------------------------------
    # Base network KPI values
    # -----------------------------------------------------

    throughput = round(
        np.clip(
            np.random.normal(100, 20),
            40,
            160
        ),
        2
    )

    latency = round(
        np.clip(
            np.random.normal(20, 7),
            5,
            60
        ),
        2
    )

    packet_loss = round(
        np.clip(
            np.random.normal(3, 1.5),
            0,
            10
        ),
        2
    )

    prb_utilization = round(
        np.clip(
            np.random.normal(65, 15),
            10,
            100
        ),
        2
    )

    sinr = round(
        np.clip(
            np.random.normal(15, 5),
            -10,
            35
        ),
        2
    )

    bler = round(
        np.clip(
            np.random.normal(3, 1.5),
            0,
            10
        ),
        2
    )

    availability = round(
        np.clip(
            np.random.normal(99, 0.5),
            95,
            100
        ),
        2
    )

    active_ue_count = random.randint(20, 200)

    # -----------------------------------------------------
    # Objective relationship
    # -----------------------------------------------------

    objective_a = XAPPS[xapp_a]["objective"]
    objective_b = XAPPS[xapp_b]["objective"]

    conflicting_pairs = {
        ("throughput", "energy"),
        ("energy", "throughput"),
        ("throughput", "mobility"),
        ("mobility", "throughput"),
        ("load_balancing", "energy"),
        ("energy", "load_balancing")
    }

    objective_conflict = int(
        (
            objective_a,
            objective_b
        ) in conflicting_pairs
    )

    # -----------------------------------------------------
    # Conflict conditions
    # -----------------------------------------------------
    #
    # Conflict becomes stronger when multiple conditions
    # occur together.
    #
    # This is deliberately NOT simply:
    # conflict = direct_conflict
    #
    # so that the ML model has a meaningful classification
    # problem.
    # -----------------------------------------------------

    conflict_score = 0

    if same_resource:
        conflict_score += 1

    if temporal_overlap:
        conflict_score += 1

    if same_cell:
        conflict_score += 1

    if same_slice:
        conflict_score += 1

    if direct_conflict:
        conflict_score += 2

    if objective_conflict:
        conflict_score += 1

    # Strong conflict requires multiple interacting factors
    conflict = int(
        conflict_score >= 5
    )

    # -----------------------------------------------------
    # Simulate KPI degradation when conflict occurs
    # -----------------------------------------------------

    if conflict:

        throughput = round(
            max(
                20,
                throughput - random.uniform(10, 35)
            ),
            2
        )

        latency = round(
            min(
                100,
                latency + random.uniform(5, 25)
            ),
            2
        )

        packet_loss = round(
            min(
                20,
                packet_loss + random.uniform(1, 6)
            ),
            2
        )

        prb_utilization = round(
            min(
                100,
                prb_utilization + random.uniform(5, 20)
            ),
            2
        )

        bler = round(
            min(
                10,
                bler + random.uniform(1, 4)
            ),
            2
        )

    # -----------------------------------------------------
    # Return record
    # -----------------------------------------------------

    return {
        "record_id": record_id,

        "xapp_a": xapp_a,
        "xapp_b": xapp_b,

        "cell_id": cell,
        "slice_id": slice_id,
        "resource_type": resource,

        "cell_a": cell_a,
        "cell_b": cell_b,

        "slice_a": slice_a,
        "slice_b": slice_b,

        "resource_a": resource_a,
        "resource_b": resource_b,

        "action_a": action_a,
        "action_b": action_b,

        "delta_a": round(delta_a, 2),
        "delta_b": round(delta_b, 2),

        "start_a": start_a,
        "end_a": end_a,

        "start_b": start_b,
        "end_b": end_b,

        "start_minute": start_minute,
        "end_minute": end_minute,

        "throughput": throughput,
        "latency": latency,
        "packet_loss": packet_loss,
        "prb_utilization": prb_utilization,
        "sinr": sinr,
        "bler": bler,
        "availability": availability,

        "active_ue_count": active_ue_count,

        "same_cell": same_cell,
        "same_slice": same_slice,
        "same_resource": same_resource,
        "temporal_overlap": temporal_overlap,

        "direct_conflict": direct_conflict,
        "objective_conflict": objective_conflict,

        "conflict_score": conflict_score,
        "conflict": conflict
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    records = []

    for record_id in range(
        1,
        NUM_RECORDS + 1
    ):
        records.append(
            generate_record(record_id)
        )

    df = pd.DataFrame(records)

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("=" * 60)
    print("Synthetic O-RAN xApp Dataset Generated")
    print("=" * 60)

    print(f"Records: {len(df)}")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nClass distribution:")
    print(df["conflict"].value_counts())

    print("\nInteraction conditions:")
    print("Same cell:")
    print(df["same_cell"].value_counts())

    print("\nSame slice:")
    print(df["same_slice"].value_counts())

    print("\nSame resource:")
    print(df["same_resource"].value_counts())

    print("\nTemporal overlap:")
    print(df["temporal_overlap"].value_counts())

    print("\nFirst 5 records:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)


if __name__ == "__main__":
    main()
