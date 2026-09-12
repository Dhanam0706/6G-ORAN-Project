# ================================================================
# 6G-O-RAN DRIFT DETECTION - RESEARCH-GRADE EVALUATION
# ================================================================

import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    confusion_matrix,
    precision_recall_fscore_support,
)

from river.drift import ADWIN

warnings.filterwarnings("ignore")


# ================================================================
# CONFIGURATION
# ================================================================

INPUT_FILE = (
    r"C:\Users\DHANAM\OneDrive\Desktop\JAVA"
    r"\network_timeseries_drift_patterns.csv"
)

TARGET = "ue1_throughput_Mbps"

BASELINE_END = 100

# Last 30% of baseline = validation
BASELINE_VALIDATION_FRACTION = 0.30

LAGS = [1, 2, 3]

TOP_K_FEATURES = 80

# Random Forest
RF_TREES = 300
RF_RANDOM_STATE = 42
RF_MAX_FEATURES = "sqrt"
RF_MIN_SAMPLES_LEAF = 2

# ADWIN
ADWIN_DELTA = 0.002

# Type-LDD
TYPE_LDD_WINDOW_LENGTH = 51
TYPE_LDD_GAP_LENGTH = 50

TYPE_LDD_TRAIN_SAMPLES_PER_CLASS = 500
TYPE_LDD_TEST_SAMPLES_PER_CLASS = 150

TYPE_LDD_EPISODES = 1000

TYPE_LDD_SUPPORT = 5
TYPE_LDD_QUERY = 5

TYPE_LDD_LEARNING_RATE = 0.001
TYPE_LDD_LOCATION_WEIGHT = 0.20

TYPE_LDD_CONFIDENCE_THRESHOLD = 0.60

TYPE_LDD_GAP_CLIP = 20.0

TYPE_LDD_ERROR_FLOOR_RATIO = 0.01
TYPE_LDD_MIN_ERROR_FLOOR = 1e-5

REAL_EVAL_OFFSETS = [10, 25, 40]

MAX_ADWIN_MATCH_DELAY = 120

RANDOM_SEED = 41

DEVICE = torch.device("cpu")

TYPE_CLASSES = [
    "SUDDEN",
    "GRADUAL",
    "INCREMENTAL",
    "NORMAL",
]


# ================================================================
# DRIFT EVENTS
# ================================================================

DRIFT_EVENTS = [

    {
        "event_id": 1,
        "start": 100,
        "end": 149,
        "condition": "SUDDEN_CONGESTION",
        "drift_type": "SUDDEN",
    },

    {
        "event_id": 2,
        "start": 200,
        "end": 249,
        "condition": "GRADUAL_CONGESTION",
        "drift_type": "GRADUAL",
    },

    {
        "event_id": 3,
        "start": 300,
        "end": 349,
        "condition": "INCREMENTAL_CONGESTION",
        "drift_type": "INCREMENTAL",
    },

    {
        "event_id": 4,
        "start": 400,
        "end": 449,
        "condition": "SUDDEN_SEVERE",
        "drift_type": "SUDDEN",
    },

    {
        "event_id": 5,
        "start": 500,
        "end": 549,
        "condition": "GRADUAL_HEAVY_CONGESTION",
        "drift_type": "GRADUAL",
    },

]


# ================================================================
# OUTPUT FILES
# ================================================================

ADWIN_RESULTS_FILE = "adwin_results.csv"
ADWIN_DETECTIONS_FILE = "adwin_detections.csv"
ADWIN_EVENT_EVALUATION_FILE = "adwin_event_evaluation.csv"

TYPE_LDD_VALIDATION_FILE = "typeldd_validation_results.csv"
TYPE_LDD_RESULTS_FILE = "typeld_real_checkpoint_results.csv"

TYPE_LDD_REAL_EVALUATION_FILE = (
    "typeld_real_evaluation_summary.csv"
)

TYPE_LDD_REAL_CONFUSION_FILE = (
    "typeld_real_confusion_matrix.csv"
)

TYPE_LDD_REAL_PER_CLASS_FILE = (
    "typeld_real_per_class_metrics.csv"
)


# ================================================================
# RANDOM SEEDS
# ================================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ================================================================
# HEADER
# ================================================================

print("=" * 75)
print("6G-O-RAN DRIFT DETECTION")
print("RESEARCH-GRADE EVALUATION")
print("=" * 75)


# ================================================================
# LOAD DATA
# ================================================================

print("\nLoading network time-series data...")

df = pd.read_csv(INPUT_FILE)

print("Original rows:", len(df))
print("Original columns:", len(df.columns))


# ================================================================
# CHECK REQUIRED COLUMNS
# ================================================================

for c in ["time", "condition", TARGET]:

    if c not in df.columns:

        raise ValueError(
            f"Missing required column: {c}"
        )


# ================================================================
# CLEAN IMPORTANT COLUMNS
# ================================================================

df["time"] = pd.to_numeric(
    df["time"],
    errors="coerce"
)

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)


df = df.dropna(
    subset=["time", TARGET]
).reset_index(drop=True)


# ================================================================
# CREATE ORIGINAL FEATURE LIST
# ================================================================

feature_columns = [
    c
    for c in df.columns
    if c not in [
        "time",
        "condition",
        TARGET,
    ]
]


print(
    "\nOriginal feature columns:",
    len(feature_columns)
)


# ================================================================
# CREATE LAG FEATURES
# ================================================================

print("\nCreating lag features...")

for lag in LAGS:

    for c in feature_columns:

        df[f"{c}_lag{lag}"] = (
            df[c].shift(lag)
        )


# ================================================================
# REMOVE ROWS WITH MISSING LAGS
# ================================================================

df = df.dropna().reset_index(
    drop=True
)


print(
    "Rows after lagging:",
    len(df)
)


# ================================================================
# BASELINE / STREAM SPLIT
# ================================================================

baseline_df = df[
    df["time"] <= BASELINE_END
].copy()

stream_df = df[
    df["time"] > BASELINE_END
].copy()


print("\nBASELINE / STREAM")
print("-----------------")

print(
    "Baseline rows:",
    len(baseline_df)
)

print(
    "Stream rows:",
    len(stream_df)
)

print(
    f"Baseline time: "
    f"{baseline_df['time'].min()} -> "
    f"{baseline_df['time'].max()}"
)

print(
    f"Stream time: "
    f"{stream_df['time'].min()} -> "
    f"{stream_df['time'].max()}"
)


# ================================================================
# CHRONOLOGICAL BASELINE TRAIN / VALIDATION SPLIT
# ================================================================

print("\nBASELINE CHRONOLOGICAL SPLIT")
print("----------------------------")

baseline_df = (
    baseline_df
    .sort_values("time")
    .reset_index(drop=True)
)


validation_size = int(
    np.ceil(
        len(baseline_df)
        * BASELINE_VALIDATION_FRACTION
    )
)


train_size = (
    len(baseline_df)
    - validation_size
)


if train_size <= 0:

    raise ValueError(
        "Baseline training set is empty."
    )


baseline_train_df = (
    baseline_df.iloc[:train_size]
    .copy()
)

baseline_val_df = (
    baseline_df.iloc[train_size:]
    .copy()
)


print(
    "Baseline total rows:",
    len(baseline_df)
)

print(
    "Baseline training rows:",
    len(baseline_train_df)
)

print(
    "Baseline validation rows:",
    len(baseline_val_df)
)

print(
    f"Training time: "
    f"{baseline_train_df['time'].min()} -> "
    f"{baseline_train_df['time'].max()}"
)

print(
    f"Validation time: "
    f"{baseline_val_df['time'].min()} -> "
    f"{baseline_val_df['time'].max()}"
)

print(
    "Validation fraction:",
    BASELINE_VALIDATION_FRACTION
)


# ================================================================
# CANDIDATE FEATURES
# ================================================================

candidate_features = [

    c

    for c in df.columns

    if c not in [
        "time",
        "condition",
        TARGET,
    ]

]


# ================================================================
# REMOVE CONSTANT FEATURES
#
# IMPORTANT:
# Variance is calculated ONLY on training data.
# ================================================================

train_variances = (
    baseline_train_df[
        candidate_features
    ].var()
)


candidate_features = [

    c

    for c in candidate_features

    if (
        pd.notna(
            train_variances[c]
        )
        and
        train_variances[c] > 0
    )

]


print(
    "\nNon-constant training features:",
    len(candidate_features)
)


# ================================================================
# SELECT TOP FEATURES
#
# IMPORTANT:
# SelectKBest is FIT ONLY on training data.
# ================================================================

k = min(
    TOP_K_FEATURES,
    len(candidate_features)
)


selector = SelectKBest(
    score_func=f_regression,
    k=k
)


# FIT ON TRAINING DATA ONLY
X_train_selected = (
    selector.fit_transform(
        baseline_train_df[
            candidate_features
        ],
        baseline_train_df[TARGET]
    )
)


selected_features = list(

    np.array(candidate_features)[
        selector.get_support()
    ]

)


print(
    "Selected features:",
    len(selected_features)
)


# ================================================================
# VALIDATION TRANSFORMATION
#
# IMPORTANT FIX:
#
# selector was fitted using ALL candidate_features.
# Therefore transform() must also receive ALL
# candidate_features.
#
# The selector then keeps the selected 50 internally.
# ================================================================

X_val_selected = (
    selector.transform(
        baseline_val_df[
            candidate_features
        ]
    )
)


print(
    "Validation feature matrix:",
    X_val_selected.shape
)


# ================================================================
# TRAIN FROZEN RANDOM FOREST
# ================================================================

print("\nTraining frozen Random Forest...")

rf = RandomForestRegressor(

    n_estimators=RF_TREES,

    random_state=RF_RANDOM_STATE,

    max_features=RF_MAX_FEATURES,

    min_samples_leaf=RF_MIN_SAMPLES_LEAF,

    n_jobs=-1,

)


# IMPORTANT:
# RF sees ONLY training data.
rf.fit(

    X_train_selected,

    baseline_train_df[TARGET]

)


print(
    "Random Forest trees:",
    RF_TREES
)

print(
    "Random Forest remains frozen during streaming."
)


# ================================================================
# TRAINING PERFORMANCE
#
# This is NOT the main research metric.
# It is shown only as a reference.
# ================================================================

train_pred = rf.predict(
    X_train_selected
)


train_mae = mean_absolute_error(

    baseline_train_df[TARGET],

    train_pred

)


train_rmse = np.sqrt(

    mean_squared_error(

        baseline_train_df[TARGET],

        train_pred

    )

)


print(
    "\nBASELINE RF TRAINING PERFORMANCE"
)

print(
    "--------------------------------"
)

print(
    f"MAE : {train_mae:.9f}"
)

print(
    f"RMSE: {train_rmse:.9f}"
)


# ================================================================
# INDEPENDENT BASELINE VALIDATION
# ================================================================

val_pred = rf.predict(
    X_val_selected
)


validation_mae = mean_absolute_error(

    baseline_val_df[TARGET],

    val_pred

)


validation_rmse = np.sqrt(

    mean_squared_error(

        baseline_val_df[TARGET],

        val_pred

    )

)


print(
    "\nBASELINE RF VALIDATION PERFORMANCE"
)

print(
    "----------------------------------"
)

print(
    f"MAE : {validation_mae:.9f}"
)

print(
    f"RMSE: {validation_rmse:.9f}"
)

print(
    "\nValidation data was NOT used "
    "for RF training or feature selection."
)


# ================================================================
# PREDICT COMPLETE TIMELINE WITH FROZEN RF
#
# RF expects exactly the 50 selected features.
# ================================================================

all_selected = df[
    selected_features
]


all_pred = rf.predict(
    all_selected
)


all_actual = (
    df[TARGET].to_numpy()
)


all_errors = np.abs(

    all_actual
    - all_pred

)


all_times = (
    df["time"].to_numpy()
)


# ================================================================
# STREAM ONLY
# ================================================================

stream_mask = (
    df["time"].to_numpy()
    > BASELINE_END
)


prediction_errors = (
    all_errors[stream_mask]
)

time_stream = (
    all_times[stream_mask]
)

condition_stream = (
    df.loc[
        stream_mask,
        "condition"
    ].to_numpy()
)

y_stream = (
    df.loc[
        stream_mask,
        TARGET
    ].to_numpy()
)

stream_predictions = (
    all_pred[stream_mask]
)


# ================================================================
# ERROR FLOOR
#
# ONLY BASELINE VALIDATION ERRORS ARE USED.
# ================================================================

baseline_validation_errors = np.abs(

    baseline_val_df[TARGET].to_numpy()
    - val_pred

)


positive_validation_errors = (

    baseline_validation_errors[
        baseline_validation_errors > 0
    ]

)


if len(
    positive_validation_errors
) > 0:

    real_error_floor = max(

        TYPE_LDD_MIN_ERROR_FLOOR,

        float(
            np.median(
                positive_validation_errors
            )
        )
        * TYPE_LDD_ERROR_FLOOR_RATIO

    )

else:

    real_error_floor = (
        TYPE_LDD_MIN_ERROR_FLOOR
    )


print(
    "\nType-LDD error floor:",
    real_error_floor
)


# ================================================================
# ADWIN
# ================================================================

print(
    "\nRunning ADWIN..."
)


adwin = ADWIN(
    delta=ADWIN_DELTA
)


adwin_rows = []
adwin_detections = []


for i, error in enumerate(
    prediction_errors
):

    error = float(error)


    adwin.update(
        error
    )


    detected = bool(
        adwin.drift_detected
    )


    row = {

        "stream_index": i,

        "time": int(
            time_stream[i]
        ),

        "condition":
            condition_stream[i],

        "actual":
            float(
                y_stream[i]
            ),

        "prediction":
            float(
                stream_predictions[i]
            ),

        "prediction_error":
            error,

        "adwin_detection":
            detected,

        "adwin_width":
            float(
                adwin.width
            ),

    }


    adwin_rows.append(
        row
    )


    if detected:

        adwin_detections.append(
            row.copy()
        )


adwin_results_df = pd.DataFrame(
    adwin_rows
)


detections_df = pd.DataFrame(
    adwin_detections
)


adwin_results_df.to_csv(
    ADWIN_RESULTS_FILE,
    index=False
)


detections_df.to_csv(
    ADWIN_DETECTIONS_FILE,
    index=False
)


print(
    "\nADWIN RESULTS"
)

print(
    "-------------"
)

print(
    "ADWIN delta:",
    ADWIN_DELTA
)

print(
    "Total detections:",
    len(adwin_detections)
)


for d in adwin_detections:

    print(

        f"{d['time']}s -> "
        f"error={d['prediction_error']:.6f}, "
        f"width={d['adwin_width']:.0f}, "
        f"condition={d['condition']}"

    )


# ================================================================
# ERROR STATISTICS
# ================================================================

print(
    "\nERROR STATISTICS BY CONDITION"
)

print(
    "-----------------------------"
)


print(

    adwin_results_df
    .groupby("condition")[
        "prediction_error"
    ]
    .agg(
        [
            "count",
            "mean",
            "std",
            "min",
            "max",
        ]
    )

)


# ================================================================
# ADWIN EVENT EVALUATION
# ================================================================

print(
    "\n"
    + "=" * 75
)

print(
    "ADWIN DRIFT-EVENT EVALUATION"
)

print(
    "=" * 75
)


event_rows = []

used_detection_indices = set()


for j, event in enumerate(
    DRIFT_EVENTS
):

    start = event["start"]

    end = event["end"]


    if (
        j + 1
        < len(DRIFT_EVENTS)
    ):

        next_start = (
            DRIFT_EVENTS[
                j + 1
            ]["start"]
        )

    else:

        next_start = np.inf


    search_end = min(

        start
        + MAX_ADWIN_MATCH_DELAY,

        next_start - 1

    )


    candidates = [

        d

        for d in adwin_detections

        if (

            d["stream_index"]
            not in used_detection_indices

            and

            start
            <= d["time"]
            <= search_end

        )

    ]


    if candidates:

        d = min(

            candidates,

            key=lambda x: x["time"]

        )


        used_detection_indices.add(
            d["stream_index"]
        )


        detected = True

        detection_time = float(
            d["time"]
        )

        delay = (
            detection_time
            - start
        )


        if detection_time <= end:

            phase = (
                "DURING_DRIFT"
            )

        else:

            phase = (
                "POST_DRIFT_DELAYED"
            )


    else:

        detected = False

        detection_time = np.nan

        delay = np.nan

        phase = "MISSED"


    event_rows.append({

        "event_id":
            event["event_id"],

        "drift_type":
            event["drift_type"],

        "condition":
            event["condition"],

        "drift_start":
            start,

        "drift_end":
            end,

        "detected":
            detected,

        "adwin_detection_time":
            detection_time,

        "detection_delay_seconds":
            delay,

        "detection_phase":
            phase,

        "max_allowed_match_delay":
            MAX_ADWIN_MATCH_DELAY,

    })


adwin_event_df = pd.DataFrame(
    event_rows
)


adwin_event_df.to_csv(

    ADWIN_EVENT_EVALUATION_FILE,

    index=False

)


print(
    "\nADWIN EVENT RESULTS"
)

print(
    "-------------------"
)

print(
    adwin_event_df.to_string(
        index=False
    )
)


# ================================================================
# ADWIN METRICS
# ================================================================

detected_event_count = int(

    adwin_event_df[
        "detected"
    ].sum()

)


total_event_count = len(
    DRIFT_EVENTS
)


coverage = (

    100
    * detected_event_count
    / total_event_count

)


delays = (

    adwin_event_df.loc[
        adwin_event_df["detected"],
        "detection_delay_seconds"
    ]

)


if len(delays):

    mean_delay = float(
        delays.mean()
    )

    median_delay = float(
        delays.median()
    )

else:

    mean_delay = np.nan

    median_delay = np.nan


timely = int(

    (

        adwin_event_df["detected"]

        &

        (

            adwin_event_df[
                "adwin_detection_time"
            ]

            <=

            adwin_event_df[
                "drift_end"
            ]

        )

    ).sum()

)


print(
    "\nADWIN COVERAGE"
)

print(
    "--------------"
)

print(

    f"Detected events : "
    f"{detected_event_count}/"
    f"{total_event_count}"

)

print(

    f"Detection coverage: "
    f"{coverage:.2f}%"

)

print(

    f"Mean detection delay: "
    f"{mean_delay:.2f}s"

)

print(

    f"Median detection delay: "
    f"{median_delay:.2f}s"

)

print(

    f"Timely detection: "
    f"{timely}/"
    f"{total_event_count} "
    f"("
    f"{100 * timely / total_event_count:.2f}%"
    ")"

)


# ================================================================
# TYPE-LDD PREPROCESSING
# ================================================================

def convert_to_type_ldd_vector(
    raw_values,
    floor_value=None
):

    x = np.asarray(
        raw_values,
        dtype=np.float64
    )


    if len(x) != TYPE_LDD_WINDOW_LENGTH:

        raise ValueError(

            "Type-LDD window must contain "
            f"{TYPE_LDD_WINDOW_LENGTH} samples."

        )


    x = np.maximum(
        x,
        0.0
    )


    if floor_value is None:

        positive_values = (
            x[x > 0]
        )


        if len(
            positive_values
        ):

            floor_value = max(

                TYPE_LDD_MIN_ERROR_FLOOR,

                float(
                    np.median(
                        positive_values
                    )
                )
                * TYPE_LDD_ERROR_FLOOR_RATIO

            )

        else:

            floor_value = (
                TYPE_LDD_MIN_ERROR_FLOOR
            )


    denominator = np.maximum(

        x[:-1],

        floor_value

    )


    gaps = (

        x[1:]
        - x[:-1]

    ) / denominator


    return np.clip(

        gaps,

        -TYPE_LDD_GAP_CLIP,

        TYPE_LDD_GAP_CLIP

    ).astype(
        np.float32
    )


# ================================================================
# SYNTHETIC TYPE-LDD DATA GENERATION
# ================================================================

def generate_error_stream(
    drift_type,
    length=51
):

    base = np.random.uniform(
        0.005,
        0.08
    )


    values = np.zeros(
        length,
        dtype=np.float64
    )


    values[0] = base

    state = 0.0


    # ------------------------------------------------------------
    # Normal temporal variation
    # ------------------------------------------------------------

    for i in range(
        1,
        length
    ):

        state = (

            0.85 * state

            +

            np.random.normal(
                0,
                0.02
            )

        )


        values[i] = (

            values[i - 1]

            *

            np.exp(
                state * 0.05
            )

        )


    # ------------------------------------------------------------
    # NORMAL
    # ------------------------------------------------------------

    if drift_type == "NORMAL":

        true_location = 0


    # ------------------------------------------------------------
    # SUDDEN
    # ------------------------------------------------------------

    elif drift_type == "SUDDEN":

        p = np.random.randint(
            8,
            40
        )


        if np.random.rand() < 0.5:

            factor = np.random.uniform(
                2,
                5
            )

        else:

            factor = np.random.uniform(
                0.25,
                0.60
            )


        values[p:] *= factor


        true_location = p - 1


    # ------------------------------------------------------------
    # GRADUAL
    # ------------------------------------------------------------

    elif drift_type == "GRADUAL":

        s = np.random.randint(
            6,
            18
        )


        duration = np.random.randint(
            12,
            25
        )


        e = min(

            length - 1,

            s + duration

        )


        if np.random.rand() < 0.5:

            factor = np.random.uniform(
                2,
                5
            )

        else:

            factor = np.random.uniform(
                0.25,
                0.60
            )


        for i in range(
            s,
            e + 1
        ):

            values[i] *= (

                1

                +

                (

                    (i - s)

                    /

                    max(
                        1,
                        e - s
                    )

                )

                *

                (factor - 1)

            )


        if e + 1 < length:

            values[e + 1:] *= factor


        true_location = (

            (s + e) // 2

            - 1

        )


    # ------------------------------------------------------------
    # INCREMENTAL
    # ------------------------------------------------------------

    elif drift_type == "INCREMENTAL":

        s = np.random.randint(
            5,
            16
        )


        if np.random.rand() < 0.5:

            factor = np.random.uniform(
                2,
                5
            )

        else:

            factor = np.random.uniform(
                0.25,
                0.60
            )


        for i in range(
            s,
            length
        ):

            values[i] *= (

                1

                +

                (

                    (i - s)

                    /

                    max(
                        1,
                        length - 1 - s
                    )

                )

                *

                (factor - 1)

            )


        true_location = s - 1


    else:

        raise ValueError(
            drift_type
        )


    # Add realistic noise
    values *= np.exp(

        np.random.normal(
            0,
            0.025,
            length
        )

    )


    values = np.maximum(

        values,

        TYPE_LDD_MIN_ERROR_FLOOR

    )


    return (

        values.astype(
            np.float32
        ),

        int(
            true_location
        )

    )


# ================================================================
# CREATE TYPE-LDD DATASET
# ================================================================

def create_type_ldd_dataset(
    n
):

    X = []
    y = []
    loc = []
    raw = []


    for cid, drift_type in enumerate(
        TYPE_CLASSES
    ):

        for _ in range(n):

            raw_stream, location = (

                generate_error_stream(
                    drift_type
                )

            )


            X.append(

                convert_to_type_ldd_vector(
                    raw_stream
                )

            )


            y.append(cid)

            loc.append(location)

            raw.append(raw_stream)


    return (

        np.asarray(
            X,
            dtype=np.float32
        ),

        np.asarray(
            y,
            dtype=np.int64
        ),

        np.asarray(
            loc,
            dtype=np.float32
        ),

        np.asarray(
            raw,
            dtype=np.float32
        ),

    )


# ================================================================
# TYPE-LDD NETWORK
# ================================================================

class TypeLDDNetwork(
    nn.Module
):

    def __init__(
        self,
        input_size=50,
        embedding_size=30
    ):

        super().__init__()


        self.embedding = nn.Sequential(

            nn.Linear(
                input_size,
                250
            ),

            nn.ReLU(),

            nn.Linear(
                250,
                250
            ),

            nn.ReLU(),

            nn.Linear(
                250,
                embedding_size
            ),

        )


        self.location_network = nn.Sequential(

            nn.Linear(
                input_size
                + embedding_size,
                250
            ),

            nn.ReLU(),

            nn.Linear(
                250,
                250
            ),

            nn.ReLU(),

            nn.Linear(
                250,
                1
            ),

        )


    def get_embedding(
        self,
        x
    ):

        return self.embedding(x)


    def predict_location(
        self,
        x,
        embedding
    ):

        combined = torch.cat(

            [
                x,
                embedding
            ],

            dim=1

        )


        return (

            self.location_network(
                combined
            ).squeeze(1)

        )


# ================================================================
# TRAIN TYPE-LDD
# ================================================================

print(
    "\n"
    + "=" * 75
)

print(
    "TRAINING TYPE-LDD MODEL"
)

print(
    "=" * 75
)


X_train, y_train, loc_train, raw_train = (

    create_type_ldd_dataset(
        TYPE_LDD_TRAIN_SAMPLES_PER_CLASS
    )

)


print(
    "Training samples:",
    len(X_train)
)

print(
    "Training vector length:",
    X_train.shape[1]
)


Xt = torch.tensor(
    X_train,
    dtype=torch.float32
)


yt = torch.tensor(
    y_train,
    dtype=torch.long
)


lt = torch.tensor(
    loc_train,
    dtype=torch.float32
)


model = TypeLDDNetwork().to(
    DEVICE
)


optimizer = optim.Adam(

    model.parameters(),

    lr=TYPE_LDD_LEARNING_RATE

)


# ================================================================
# EPISODIC TYPE-LDD TRAINING
# ================================================================

for ep in range(
    TYPE_LDD_EPISODES
):

    model.train()


    support_x = []
    support_y = []

    query_x = []
    query_y = []


    for cid in range(4):

        indices = np.where(
            y_train == cid
        )[0]


        selected = np.random.choice(

            indices,

            TYPE_LDD_SUPPORT
            + TYPE_LDD_QUERY,

            replace=False

        )


        support_indices = selected[
            :TYPE_LDD_SUPPORT
        ]


        query_indices = selected[
            TYPE_LDD_SUPPORT:
        ]


        support_x.extend(

            Xt[
                support_indices
            ]

        )


        support_y += (

            [cid]
            * TYPE_LDD_SUPPORT

        )


        query_x.extend(

            Xt[
                query_indices
            ]

        )


        query_y += (

            [cid]
            * TYPE_LDD_QUERY

        )


    support_x = torch.stack(
        support_x
    )


    query_x = torch.stack(
        query_x
    )


    support_y = torch.tensor(

        support_y,

        dtype=torch.long

    )


    query_y = torch.tensor(

        query_y,

        dtype=torch.long

    )


    support_embedding = (

        model.get_embedding(
            support_x
        )

    )


    query_embedding = (

        model.get_embedding(
            query_x
        )

    )


    prototypes = torch.stack(

        [

            support_embedding[
                support_y == cid
            ].mean(0)

            for cid in range(4)

        ]

    )


    distances = torch.cdist(

        query_embedding,

        prototypes

    )


    class_loss = (

        nn.functional.cross_entropy(

            -distances,

            query_y

        )

    )


    # ------------------------------------------------------------
    # Location regression
    # ------------------------------------------------------------

    batch_size = min(
        64,
        len(X_train)
    )


    batch_indices = np.random.choice(

        len(X_train),

        batch_size,

        replace=False

    )


    batch_x = Xt[
        batch_indices
    ]


    batch_embedding = (

        model.get_embedding(
            batch_x
        )

    )


    predicted_location = (

        model.predict_location(

            batch_x,

            batch_embedding

        )

    )


    location_loss = (

        nn.functional.smooth_l1_loss(

            predicted_location,

            lt[
                batch_indices
            ]

        )

    )


    loss = (

        class_loss

        +

        TYPE_LDD_LOCATION_WEIGHT
        * location_loss

    )


    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


    if (
        (ep + 1) % 100
        == 0
    ):

        print(

            f"Episode {ep + 1}/"
            f"{TYPE_LDD_EPISODES} "
            f"Loss={loss.item():.4f} "
            f"ClassLoss={class_loss.item():.4f} "
            f"LocationLoss={location_loss.item():.4f}"

        )


# ================================================================
# FINAL TYPE-LDD PROTOTYPES
# ================================================================

model.eval()


with torch.no_grad():

    train_embeddings = (

        model.get_embedding(
            Xt
        )

    )


    final_prototypes = torch.stack(

        [

            train_embeddings[
                yt == cid
            ].mean(0)

            for cid in range(4)

        ]

    )


# ================================================================
# TYPE-LDD PREDICTION FUNCTION
# ================================================================

def predict_type_ldd(
    raw_window,
    floor_value
):

    vector = (

        convert_to_type_ldd_vector(

            raw_window,

            floor_value

        )

    )


    x = torch.tensor(

        vector,

        dtype=torch.float32

    ).unsqueeze(0)


    with torch.no_grad():

        embedding = (

            model.get_embedding(
                x
            )

        )


        distances = torch.cdist(

            embedding,

            final_prototypes

        ).squeeze(0)


        probabilities = torch.softmax(

            -distances,

            dim=0

        )


        cid = int(

            torch.argmax(
                probabilities
            )

        )


        confidence = float(

            probabilities[cid]

        )


        location = float(

            model.predict_location(

                x,

                embedding

            ).item()

        )


    return (

        TYPE_CLASSES[cid],

        confidence,

        float(

            np.clip(

                location,

                0,
                49

            )

        )

    )


# ================================================================
# SYNTHETIC VALIDATION
# ================================================================

print(
    "\n"
    + "=" * 75
)

print(
    "TYPE-LDD VALIDATION ON UNSEEN SYNTHETIC DATA"
)

print(
    "=" * 75
)


X_test, y_test, loc_test, raw_test = (

    create_type_ldd_dataset(

        TYPE_LDD_TEST_SAMPLES_PER_CLASS

    )

)


predicted_classes = []

predicted_locations = []

confidences = []


for raw_window in raw_test:

    predicted_type, confidence, location = (

        predict_type_ldd(

            raw_window,

            None

        )

    )


    predicted_classes.append(

        TYPE_CLASSES.index(
            predicted_type
        )

    )


    predicted_locations.append(
        location
    )


    confidences.append(
        confidence
    )


predicted_classes = np.asarray(
    predicted_classes
)


predicted_locations = np.asarray(
    predicted_locations
)


confidences = np.asarray(
    confidences
)


synthetic_accuracy = (

    100

    *

    np.mean(

        predicted_classes
        ==
        y_test

    )

)


synthetic_drift_mask = (
    y_test != 3
)


synthetic_location_mae = (

    mean_absolute_error(

        loc_test[
            synthetic_drift_mask
        ],

        predicted_locations[
            synthetic_drift_mask
        ]

    )

)


synthetic_confidence = float(
    confidences.mean()
)


print(

    f"\nOverall Type-LDD "
    f"synthetic accuracy: "
    f"{synthetic_accuracy:.2f}%"

)


print(
    "\nSYNTHETIC PER-CLASS ACCURACY"
)

print(
    "----------------------------"
)


for cid, drift_type in enumerate(
    TYPE_CLASSES
):

    class_mask = (
        y_test == cid
    )


    class_accuracy = (

        100

        *

        np.mean(

            predicted_classes[
                class_mask
            ]

            ==

            y_test[
                class_mask
            ]

        )

    )


    print(

        f"{drift_type:12s}: "
        f"{class_accuracy:.2f}%"

    )


# ================================================================
# SYNTHETIC CONFUSION MATRIX
# ================================================================

synthetic_cm = confusion_matrix(

    y_test,

    predicted_classes,

    labels=[0, 1, 2, 3]

)


print(
    "\nSYNTHETIC CONFUSION MATRIX"
)

print(
    "--------------------------"
)


print(

    pd.DataFrame(

        synthetic_cm,

        index=TYPE_CLASSES,

        columns=TYPE_CLASSES

    )

)


print(

    f"\nSynthetic location MAE: "
    f"{synthetic_location_mae:.2f} samples"

)


print(

    f"Synthetic average confidence: "
    f"{synthetic_confidence:.4f}"

)


# ================================================================
# SAVE SYNTHETIC RESULTS
# ================================================================

synthetic_results_df = pd.DataFrame({

    "sample":
        np.arange(
            len(y_test)
        ),

    "actual_type":
        [
            TYPE_CLASSES[i]
            for i in y_test
        ],

    "predicted_type":
        [
            TYPE_CLASSES[i]
            for i in predicted_classes
        ],

    "actual_location":
        loc_test,

    "predicted_location":
        predicted_locations,

    "location_error":
        np.abs(
            loc_test
            -
            predicted_locations
        ),

    "confidence":
        confidences,

    "classification_correct":
        (
            predicted_classes
            ==
            y_test
        ),

})


synthetic_results_df.to_csv(

    TYPE_LDD_VALIDATION_FILE,

    index=False

)


print(

    "\nSynthetic validation saved:",
    TYPE_LDD_VALIDATION_FILE

)


# ================================================================
# REAL CAUSAL TYPE-LDD EVALUATION
# ================================================================

print(
    "\n"
    + "=" * 75
)

print(
    "REAL 6G-O-RAN TYPE-LDD "
    "CAUSAL CHECKPOINT EVALUATION"
)

print(
    "=" * 75
)


# ================================================================
# TIME -> INDEX
# ================================================================

time_to_index = {

    int(time_value): index

    for index, time_value
    in enumerate(all_times)

}


real_rows = []


# ================================================================
# CHECKPOINT EVALUATION
# ================================================================

for event in DRIFT_EVENTS:

    for offset in REAL_EVAL_OFFSETS:

        checkpoint = (

            event["start"]
            +
            offset

        )


        if checkpoint > event["end"]:

            continue


        if checkpoint not in time_to_index:

            print(

                f"Skipping event "
                f"{event['event_id']}, "
                f"checkpoint "
                f"{checkpoint}s: "
                f"time not found."

            )

            continue


        end_idx = (

            time_to_index[
                checkpoint
            ]

        )


        start_idx = (

            end_idx

            -

            (
                TYPE_LDD_WINDOW_LENGTH
                - 1
            )

        )


        if start_idx < 0:

            print(

                f"Skipping event "
                f"{event['event_id']}, "
                f"checkpoint "
                f"{checkpoint}s: "
                f"insufficient history."

            )

            continue


        # --------------------------------------------------------
        # CAUSAL WINDOW
        # --------------------------------------------------------

        raw_window = (

            all_errors[
                start_idx:
                end_idx + 1
            ]

        )


        if len(raw_window) != 51:

            continue


        # --------------------------------------------------------
        # TYPE-LDD PREDICTION
        # --------------------------------------------------------

        (
            predicted_type,
            confidence,
            predicted_local
        ) = predict_type_ldd(

            raw_window,

            real_error_floor

        )


        predicted_local_integer = int(

            round(
                predicted_local
            )

        )


        # --------------------------------------------------------
        # MAP LOCAL LOCATION TO ABSOLUTE TIME
        # --------------------------------------------------------

        predicted_absolute_index = min(

            end_idx,

            start_idx
            + predicted_local_integer
            + 1

        )


        predicted_absolute_time = int(

            all_times[
                predicted_absolute_index
            ]

        )


        # --------------------------------------------------------
        # GROUND-TRUTH ONSET LOCATION
        # --------------------------------------------------------

        window_start_time = int(

            all_times[
                start_idx
            ]

        )


        ground_truth_local_gap = (

            event["start"]
            -
            window_start_time
            -
            1

        )


        location_error = abs(

            predicted_local_integer
            -
            ground_truth_local_gap

        )


        classification_correct = (

            predicted_type
            ==
            event["drift_type"]

        )


        if (

            confidence
            >=
            TYPE_LDD_CONFIDENCE_THRESHOLD

        ):

            reported_type = (
                predicted_type
            )

        else:

            reported_type = (
                "LOW_CONFIDENCE"
            )


        # --------------------------------------------------------
        # SAVE RESULT
        # --------------------------------------------------------

        real_rows.append({

            "event_id":
                event["event_id"],

            "ground_truth_type":
                event["drift_type"],

            "ground_truth_condition":
                event["condition"],

            "drift_start":
                event["start"],

            "drift_end":
                event["end"],

            "checkpoint_offset":
                offset,

            "checkpoint_time":
                checkpoint,

            "causal_window_start":
                window_start_time,

            "causal_window_end":
                int(
                    all_times[
                        end_idx
                    ]
                ),

            "predicted_type":
                predicted_type,

            "reported_type":
                reported_type,

            "confidence":
                confidence,

            "predicted_local_location":
                predicted_local_integer,

            "ground_truth_local_onset_location":
                ground_truth_local_gap,

            "predicted_absolute_time":
                predicted_absolute_time,

            "location_error_samples":
                location_error,

            "location_error_seconds":
                location_error,

            "classification_correct":
                classification_correct,

        })


        # --------------------------------------------------------
        # PRINT
        # --------------------------------------------------------

        print(

            f"\nEvent "
            f"{event['event_id']} | "
            f"{event['drift_type']} | "
            f"checkpoint="
            f"{checkpoint}s"

        )


        print(

            f"  Causal window: "
            f"{window_start_time}s "
            f"-> "
            f"{checkpoint}s"

        )


        print(

            f"  Predicted type: "
            f"{predicted_type}"

        )


        print(

            f"  Confidence: "
            f"{confidence:.4f}"

        )


        print(

            f"  Predicted location: "
            f"{predicted_absolute_time}s"

        )


        print(

            f"  Ground truth: "
            f"{event['drift_type']}"

        )


        print(

            f"  Classification correct: "
            f"{classification_correct}"

        )


        print(

            f"  Onset-location error: "
            f"{location_error}s"

        )


# ================================================================
# SAVE REAL CHECKPOINT RESULTS
# ================================================================

real_df = pd.DataFrame(
    real_rows
)


real_df.to_csv(

    TYPE_LDD_RESULTS_FILE,

    index=False

)


print(

    "\nReal causal checkpoint results saved:",
    TYPE_LDD_RESULTS_FILE

)


# ================================================================
# REAL TYPE-LDD PERFORMANCE
# ================================================================

print(
    "\n"
    + "=" * 75
)

print(
    "REAL Type-LDD PERFORMANCE"
)

print(
    "=" * 75
)


if len(real_df):

    real_accuracy = (

        100

        *

        real_df[
            "classification_correct"
        ].mean()

    )


    real_location_mae = float(

        real_df[
            "location_error_seconds"
        ].mean()

    )


    real_confidence = float(

        real_df[
            "confidence"
        ].mean()

    )

else:

    real_accuracy = np.nan

    real_location_mae = np.nan

    real_confidence = np.nan


print(

    f"\nReal Type-LDD accuracy: "
    f"{real_accuracy:.2f}%"

)


print(

    f"Real onset-location MAE: "
    f"{real_location_mae:.2f}s"

)


print(

    f"Real average confidence: "
    f"{real_confidence:.4f}"

)


# ================================================================
# REAL CONFUSION MATRIX
# ================================================================

real_labels = [
    "SUDDEN",
    "GRADUAL",
    "INCREMENTAL",
    "NORMAL",
]


if len(real_df):

    y_true = (
        real_df[
            "ground_truth_type"
        ].to_numpy()
    )


    y_pred = (
        real_df[
            "predicted_type"
        ].to_numpy()
    )


    real_cm = confusion_matrix(

        y_true,

        y_pred,

        labels=real_labels

    )


    confusion_df = pd.DataFrame(

        real_cm,

        index=real_labels,

        columns=real_labels

    )


    print(
        "\nREAL Type-LDD CONFUSION MATRIX"
    )

    print(
        "-------------------------------"
    )

    print(
        confusion_df
    )


    confusion_df.to_csv(

        TYPE_LDD_REAL_CONFUSION_FILE

    )


    # ------------------------------------------------------------
    # Precision / Recall / F1
    # ------------------------------------------------------------

    precision, recall, f1, support = (

        precision_recall_fscore_support(

            y_true,

            y_pred,

            labels=real_labels,

            zero_division=0

        )

    )


    per_class_df = pd.DataFrame({

        "type":
            real_labels,

        "precision":
            precision,

        "recall":
            recall,

        "f1_score":
            f1,

        "support":
            support,

    })


    print(
        "\nREAL PER-CLASS METRICS"
    )

    print(
        "----------------------"
    )

    print(

        per_class_df.to_string(
            index=False
        )

    )


    per_class_df.to_csv(

        TYPE_LDD_REAL_PER_CLASS_FILE,

        index=False

    )


    # ------------------------------------------------------------
    # Performance by drift type
    # ------------------------------------------------------------

    print(
        "\nREAL PERFORMANCE BY DRIFT TYPE"
    )

    print(
        "-------------------------------"
    )


    for drift_type in [
        "SUDDEN",
        "GRADUAL",
        "INCREMENTAL",
    ]:

        subset = real_df[

            real_df[
                "ground_truth_type"
            ]

            ==

            drift_type

        ]


        if len(subset):

            accuracy = (

                100

                *

                subset[
                    "classification_correct"
                ].mean()

            )


            location_mae = (

                subset[
                    "location_error_seconds"
                ].mean()

            )


            confidence = (

                subset[
                    "confidence"
                ].mean()

            )


            print(

                f"{drift_type:12s} | "
                f"Accuracy={accuracy:.2f}% | "
                f"Location MAE="
                f"{location_mae:.2f}s | "
                f"Confidence="
                f"{confidence:.4f} | "
                f"N={len(subset)}"

            )


else:

    pd.DataFrame(

        columns=[

            "type",
            "precision",
            "recall",
            "f1_score",
            "support",

        ]

    ).to_csv(

        TYPE_LDD_REAL_PER_CLASS_FILE,

        index=False

    )


# ================================================================
# RESEARCH SUMMARY CSV
# ================================================================

summary = pd.DataFrame(

    [

        [
            "RF_training_MAE",
            train_mae,
            "target_units",
        ],

        [
            "RF_training_RMSE",
            train_rmse,
            "target_units",
        ],

        [
            "RF_validation_MAE",
            validation_mae,
            "target_units",
        ],

        [
            "RF_validation_RMSE",
            validation_rmse,
            "target_units",
        ],

        [
            "ADWIN_detection_coverage",
            coverage,
            "percent",
        ],

        [
            "ADWIN_mean_detection_delay",
            mean_delay,
            "seconds",
        ],

        [
            "ADWIN_median_detection_delay",
            median_delay,
            "seconds",
        ],

        [
            "ADWIN_timely_detection_rate",
            100
            * timely
            / total_event_count,
            "percent",
        ],

        [
            "TypeLDD_synthetic_accuracy",
            synthetic_accuracy,
            "percent",
        ],

        [
            "TypeLDD_synthetic_location_MAE",
            synthetic_location_mae,
            "samples",
        ],

        [
            "TypeLDD_synthetic_average_confidence",
            synthetic_confidence,
            "probability",
        ],

        [
            "TypeLDD_real_accuracy",
            real_accuracy,
            "percent",
        ],

        [
            "TypeLDD_real_onset_location_MAE",
            real_location_mae,
            "seconds",
        ],

        [
            "TypeLDD_real_average_confidence",
            real_confidence,
            "probability",
        ],

    ],

    columns=[
        "metric",
        "value",
        "unit",
    ]

)


summary.to_csv(

    TYPE_LDD_REAL_EVALUATION_FILE,

    index=False

)


# ================================================================
# FINAL RESEARCH SUMMARY
# ================================================================

print(
    "\n"
    + "=" * 75
)

print(
    "FINAL RESEARCH SUMMARY"
)

print(
    "=" * 75
)


# ================================================================
# DATASET
# ================================================================

print(
    "\nDATASET"
)

print(
    "-------"
)

print(
    "Input file           :",
    INPUT_FILE
)

print(
    "Rows after lagging   :",
    len(df)
)

print(
    "Target               :",
    TARGET
)

print(
    "Selected RF features :",
    len(selected_features)
)


# ================================================================
# RF
# ================================================================

print(
    "\nRANDOM FOREST"
)

print(
    "-------------"
)

print(
    f"Trees                : "
    f"{RF_TREES}"
)

print(
    f"Training rows        : "
    f"{len(baseline_train_df)}"
)

print(
    f"Validation rows      : "
    f"{len(baseline_val_df)}"
)

print(
    f"Training MAE         : "
    f"{train_mae:.6f}"
)

print(
    f"Training RMSE        : "
    f"{train_rmse:.6f}"
)

print(
    f"Validation MAE       : "
    f"{validation_mae:.6f}"
)

print(
    f"Validation RMSE      : "
    f"{validation_rmse:.6f}"
)


# ================================================================
# ADWIN
# ================================================================

print(
    "\nADWIN"
)

print(
    "-----"
)

print(
    f"Delta                : "
    f"{ADWIN_DELTA}"
)

print(
    f"Total detections     : "
    f"{len(adwin_detections)}"
)

print(
    f"Event coverage       : "
    f"{coverage:.2f}%"
)

print(
    f"Mean onset delay     : "
    f"{mean_delay:.2f}s"
)

print(
    f"Median onset delay   : "
    f"{median_delay:.2f}s"
)

print(
    f"Timely event rate    : "
    f"{100 * timely / total_event_count:.2f}%"
)


# ================================================================
# SYNTHETIC TYPE-LDD
# ================================================================

print(
    "\nSYNTHETIC TYPE-LDD"
)

print(
    "------------------"
)

print(
    f"Accuracy             : "
    f"{synthetic_accuracy:.2f}%"
)

print(
    f"Location MAE         : "
    f"{synthetic_location_mae:.2f} samples"
)

print(
    f"Average confidence   : "
    f"{synthetic_confidence:.4f}"
)


# ================================================================
# REAL TYPE-LDD
# ================================================================

print(
    "\nREAL SIMULATOR Type-LDD"
)

print(
    "-----------------------"
)

print(
    f"Accuracy             : "
    f"{real_accuracy:.2f}%"
)

print(
    f"Onset-location MAE   : "
    f"{real_location_mae:.2f}s"
)

print(
    f"Average confidence   : "
    f"{real_confidence:.4f}"
)


# ================================================================
# RESEARCH INTERPRETATION
# ================================================================

print(
    "\nEXPERIMENTAL INTERPRETATION"
)

print(
    "==========================="
)

print(
    "1. RF is trained only on "
    "chronological baseline-training data."
)

print(
    "2. SelectKBest is fitted only "
    "on baseline-training data."
)

print(
    "3. Baseline validation is "
    "chronologically held out."
)

print(
    "4. RF validation metrics are "
    "calculated on unseen baseline data."
)

print(
    "5. RF remains frozen during streaming."
)

print(
    "6. ADWIN receives only stream "
    "prediction errors."
)

print(
    "7. Type-LDD receives causal "
    "prediction-error windows."
)

print(
    "8. Simulator labels are used "
    "only for post-hoc evaluation."
)

print(
    "9. ADWIN delay is measured "
    "from drift onset."
)

print(
    "10. Missed ADWIN events are "
    "reported explicitly."
)

print(
    "11. Type-LDD synthetic validation "
    "and real-simulator evaluation are separate."
)

print(
    "12. Real/synthetic performance differences "
    "can indicate a synthetic-to-simulator domain gap."
)

print(
    "13. Type-LDD location error represents "
    "simulator drift-onset localization error."
)

print(
    "14. For gradual/incremental drift, "
    "the location is a proxy rather than "
    "an instantaneous physical change point."
)

print(
    "15. Type-LDD error-floor calibration uses "
    "only baseline validation errors."
)


# ================================================================
# OUTPUT FILES
# ================================================================

print(
    "\nOUTPUT FILES"
)

print(
    "------------"
)


output_files = [

    ADWIN_RESULTS_FILE,

    ADWIN_DETECTIONS_FILE,

    ADWIN_EVENT_EVALUATION_FILE,

    TYPE_LDD_VALIDATION_FILE,

    TYPE_LDD_RESULTS_FILE,

    TYPE_LDD_REAL_EVALUATION_FILE,

    TYPE_LDD_REAL_CONFUSION_FILE,

    TYPE_LDD_REAL_PER_CLASS_FILE,

]


for filename in output_files:

    print(
        filename
    )


# ================================================================
# COMPLETE
# ================================================================

print(
    "\nPipeline completed successfully."
)

print(
    "=" * 75
)