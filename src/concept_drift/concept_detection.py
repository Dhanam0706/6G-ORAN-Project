"""
concept_drift_detector.py

Concept Drift Detection module for the O-RAN Non-RT RIC pipeline.

pip install pandas numpy scikit-learn scipy

--------------------------------------------------------------------------
WHAT "EXISTING MODEL" MEANS HERE
--------------------------------------------------------------------------
No external model file, API, or website is used. "Baseline model" means we
train scikit-learn's RandomForestRegressor -- a standard, off-the-shelf
algorithm -- on our OWN reference-window data. We are not inventing a new
ML algorithm; we are using an existing, well-understood one as a probe:
if a boring, standard model's prediction error starts behaving differently
on new data, that's evidence the underlying KPI relationships changed
(concept drift), independent of whether any single KPI's own distribution
moved (that's what the separate Data Drift module already checks).

--------------------------------------------------------------------------
PIPELINE
--------------------------------------------------------------------------
1. Pick a target KPI (y) and feature KPIs (X).
2. Split chronologically: reference window (train) vs current window
   (monitor only -- NEVER trained on).
3. Train RandomForestRegressor on the reference window only.
4. Compute baseline residuals (errors) on held-out reference data.
5. Run the SAME frozen model on the current window; compute current
   residuals.
6. Compare the two residual distributions with KS + Wasserstein + PSI
   (reusing the exact same statistical machinery as the Data Drift module,
   just applied to residuals instead of raw KPI values).
7. Require the elevated-error condition to persist across several
   consecutive windows before declaring concept drift (avoids single-window
   noise).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# ============================================================================
# Configuration (initial / configurable, not universal constants)
# ============================================================================

DEFAULT_TARGET = "DRB.UEThpDl"
MIN_SAMPLES_PER_WINDOW = 30

RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 8
RF_RANDOM_STATE = 42
VALIDATION_FRACTION = 0.2   # slice of the reference window held out, NOT trained on,
                            # used only to build the baseline error distribution

# Error-distribution comparison thresholds -- same style as the Data Drift module
KS_ALPHA = 0.05
PSI_BINS = 10
PSI_EPSILON = 1e-4
PSI_MODERATE = 0.10
PSI_MAJOR = 0.25
PSI_SATURATION = 0.50
WASSERSTEIN_SATURATION = 1.0

WEIGHT_KS = 0.4
WEIGHT_WASSERSTEIN = 0.3
WEIGHT_PSI = 0.3

SCORE_NO_DRIFT = 0.30
SCORE_DRIFT = 0.60

PERSISTENCE_WINDOWS = 3     # require this many consecutive elevated windows
                            # before declaring concept drift (configurable)


# ============================================================================
# Result containers
# ============================================================================

@dataclass
class BaselineModel:
    model: RandomForestRegressor
    feature_columns: list[str]
    target_column: str
    baseline_errors: np.ndarray       # residuals on held-out reference data
    val_mae: float
    val_r2: float


@dataclass
class MethodResult:
    name: str
    raw_value: float
    component_score: float
    individually_flagged: bool
    detail: str


# ============================================================================
# Step 1-2: Load data + chronological split
# ============================================================================

def load_data(csv_path: str, columns_needed: list[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in columns_needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {csv_path}. "
                          f"Available: {list(df.columns)}")

    for c in columns_needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    before = len(df)
    df = df.dropna(subset=columns_needed)
    df = df[np.all(np.isfinite(df[columns_needed].to_numpy()), axis=1)]
    after = len(df)
    if after < before:
        print(f"[load_data] Dropped {before - after} invalid/missing rows.")
    if after == 0:
        raise ValueError("No valid rows left after cleaning.")
    return df


def prepare_windows(
    df: pd.DataFrame,
    split_ratio: float = 0.5,
    method: str = "positional",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Same philosophy as the Data Drift module: 'positional' preserves row
    order as a stand-in for chronological order when no timestamp exists.
    In production with real timestamped KPM data, replace this with an
    explicit sort-by-timestamp + fixed calendar-time split.
    """
    n = len(df)
    split_idx = int(n * split_ratio)
    if method == "positional":
        return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
    elif method == "random":
        ref, cur = train_test_split(df, train_size=split_ratio, random_state=RF_RANDOM_STATE)
        return ref, cur
    else:
        raise ValueError(f"Unknown split method: {method}")


# ============================================================================
# Step 3-4: Train baseline model + establish baseline error distribution
# ============================================================================

def train_baseline_model(
    reference_df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    validation_fraction: float = VALIDATION_FRACTION,
) -> BaselineModel:
    """
    Trains RandomForestRegressor on the reference window ONLY.
    An internal (random, since it's all 'past' data) train/validation split
    is used purely to compute a trustworthy baseline error distribution --
    the model must NOT be evaluated on the same rows it was trained on.
    """
    X = reference_df[feature_columns].to_numpy(dtype=float)
    y = reference_df[target_column].to_numpy(dtype=float)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_fraction, random_state=RF_RANDOM_STATE
    )

    model = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RF_RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    baseline_errors = y_val - y_val_pred   # residuals, NOT absolute -- sign matters
                                            # (systematic over/under-prediction is itself a drift signal)

    val_mae = float(mean_absolute_error(y_val, y_val_pred))
    val_r2 = float(r2_score(y_val, y_val_pred))

    print(f"[train_baseline_model] Trained on {len(X_train)} rows, "
          f"validated on {len(X_val)} rows.")
    print(f"[train_baseline_model] Validation MAE={val_mae:.4f}, R^2={val_r2:.4f}")
    if val_r2 < 0.3:
        print("[train_baseline_model] WARNING: low R^2 -- the model barely explains the "
              "target from these features. Concept-drift error monitoring is only meaningful "
              "if the baseline model actually fits reasonably well; consider different/more features.")

    return BaselineModel(
        model=model,
        feature_columns=feature_columns,
        target_column=target_column,
        baseline_errors=baseline_errors,
        val_mae=val_mae,
        val_r2=val_r2,
    )


# ============================================================================
# Step 5: Run frozen model on current window
# ============================================================================

def compute_current_errors(baseline: BaselineModel, current_df: pd.DataFrame) -> np.ndarray:
    """
    Runs the ALREADY-TRAINED, FROZEN model on new data. The model is never
    refit here -- refitting on current data would hide the very drift we're
    trying to detect.
    """
    X_cur = current_df[baseline.feature_columns].to_numpy(dtype=float)
    y_cur = current_df[baseline.target_column].to_numpy(dtype=float)
    y_cur_pred = baseline.model.predict(X_cur)
    return y_cur - y_cur_pred


# ============================================================================
# Step 6: Compare baseline vs current error distributions
# (identical statistical machinery to the Data Drift module, applied to
#  residuals instead of raw KPI values)
# ============================================================================

def _ks(reference: np.ndarray, current: np.ndarray) -> MethodResult:
    stat, p_value = ks_2samp(reference, current)
    flagged = bool(p_value < KS_ALPHA)
    component = float(np.clip(stat, 0.0, 1.0))
    return MethodResult("KS", float(stat), component, flagged, f"D={stat:.4f}, p={p_value:.4g}")


def _wasserstein(reference: np.ndarray, current: np.ndarray) -> MethodResult:
    raw = float(wasserstein_distance(reference, current))
    q75, q25 = np.percentile(reference, [75, 25])
    scale = (q75 - q25) or (np.std(reference) or 1.0)
    component = float(np.clip(raw / (WASSERSTEIN_SATURATION * scale), 0.0, 1.0))
    flagged = component >= 0.5
    return MethodResult("Wasserstein", raw, component, flagged, f"raw={raw:.4f}, scale={scale:.4f}")


def _psi(reference: np.ndarray, current: np.ndarray) -> MethodResult:
    quantiles = np.linspace(0, 1, PSI_BINS + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 3:
        edges = np.linspace(reference.min(), reference.max() + 1e-9, PSI_BINS + 1)
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_pct = np.where(ref_counts == 0, PSI_EPSILON, ref_counts / max(len(reference), 1))
    cur_pct = np.where(cur_counts == 0, PSI_EPSILON, cur_counts / max(len(current), 1))
    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

    interpretation = "SIGNIFICANT_DRIFT" if psi >= PSI_MAJOR else \
                      "MODERATE_DRIFT" if psi >= PSI_MODERATE else "NO_DRIFT"
    component = float(np.clip(psi / PSI_SATURATION, 0.0, 1.0))
    flagged = interpretation != "NO_DRIFT"
    return MethodResult("PSI", psi, component, flagged, f"PSI={psi:.4f} -> {interpretation}")


def _fuse(ks_r: MethodResult, w_r: MethodResult, psi_r: MethodResult) -> tuple[float, str]:
    score = (WEIGHT_KS * ks_r.component_score
             + WEIGHT_WASSERSTEIN * w_r.component_score
             + WEIGHT_PSI * psi_r.component_score)
    score = float(np.clip(score, 0.0, 1.0))
    status = "NO_DRIFT" if score < SCORE_NO_DRIFT else \
             "WARNING" if score < SCORE_DRIFT else "DRIFT_DETECTED"
    return round(score, 4), status


def _reason(ks_r: MethodResult, w_r: MethodResult, psi_r: MethodResult) -> str:
    flagged = [n for n, r in [("KS", ks_r), ("Wasserstein", w_r), ("PSI", psi_r)] if r.individually_flagged]
    if not flagged:
        return "Insufficient consistent evidence of concept drift"
    if len(flagged) == 3:
        return "KS, Wasserstein and PSI consistently indicate a shift in model error behaviour"
    if len(flagged) == 1:
        return f"{flagged[0]} indicates a shift in model error behaviour"
    return " and ".join(flagged) + " indicate a shift in model error behaviour"


# ============================================================================
# Step 7 (simplified single-run version) + full orchestration
# ============================================================================

def detect_concept_drift(
    csv_path: str | None = None,
    reference_csv: str | None = None,
    current_csv: str | None = None,
    target_column: str = DEFAULT_TARGET,
    feature_columns: list[str] | None = None,
    split_ratio: float = 0.5,
    split_method: str = "positional",
) -> dict[str, Any]:
    if feature_columns is None:
        raise ValueError("feature_columns must be provided explicitly (which KPIs predict the target).")

    needed = [target_column] + feature_columns

    if reference_csv and current_csv:
        ref_df = load_data(reference_csv, needed)
        cur_df = load_data(current_csv, needed)
    elif csv_path:
        df = load_data(csv_path, needed)
        ref_df, cur_df = prepare_windows(df, split_ratio=split_ratio, method=split_method)
    else:
        raise ValueError("Provide either csv_path, or both reference_csv and current_csv.")

    if len(ref_df) < MIN_SAMPLES_PER_WINDOW or len(cur_df) < MIN_SAMPLES_PER_WINDOW:
        raise ValueError(f"Need >= {MIN_SAMPLES_PER_WINDOW} rows per window; "
                          f"got reference={len(ref_df)}, current={len(cur_df)}.")

    baseline = train_baseline_model(ref_df, target_column, feature_columns)
    current_errors = compute_current_errors(baseline, cur_df)

    ks_r = _ks(baseline.baseline_errors, current_errors)
    w_r = _wasserstein(baseline.baseline_errors, current_errors)
    psi_r = _psi(baseline.baseline_errors, current_errors)
    score, status = _fuse(ks_r, w_r, psi_r)
    reason = _reason(ks_r, w_r, psi_r)

    print(f"\nConcept Drift Analysis -- target: {target_column}")
    print("-" * 62)
    print(f"{'Method':<16}{'Value':<26}{'Result'}")
    print("-" * 62)
    print(f"{'KS':<16}{ks_r.detail:<26}{'DRIFT' if ks_r.individually_flagged else 'NO DRIFT'}")
    print(f"{'Wasserstein':<16}{f'{w_r.raw_value:.4f}':<26}{'DRIFT' if w_r.individually_flagged else 'NO DRIFT'}")
    print(f"{'PSI':<16}{f'{psi_r.raw_value:.4f}':<26}{'DRIFT' if psi_r.individually_flagged else 'NO DRIFT'}")
    print("-" * 62)
    print(f"{'Final Score':<16}{f'{score:.4f}':<26}{status}")
    print("-" * 62)
    print(f"(baseline val MAE={baseline.val_mae:.4f}, val R^2={baseline.val_r2:.4f} -- "
          f"sanity check that the model fits reasonably before trusting its errors)")

    return {
        "module": "concept_drift",
        "target_kpi": target_column,
        "drift_score": score,
        "status": status,
        "reason": reason,
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="O-RAN Concept Drift Detector (RF baseline + KS/Wasserstein/PSI on residuals).")
    parser.add_argument("--csv", default=None, help="Single-file mode: one CSV split into reference/current.")
    parser.add_argument("--reference", default=None, help="Two-file mode: reference/baseline CSV.")
    parser.add_argument("--current", default=None, help="Two-file mode: current/monitoring CSV.")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Target KPI to predict.")
    parser.add_argument("--features", nargs="+", required=True, help="Feature KPI columns used to predict the target.")
    parser.add_argument("--split-ratio", type=float, default=0.5)
    parser.add_argument("--split-method", choices=["positional", "random"], default="positional")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    result = detect_concept_drift(
        csv_path=args.csv,
        reference_csv=args.reference,
        current_csv=args.current,
        target_column=args.target,
        feature_columns=args.features,
        split_ratio=args.split_ratio,
        split_method=args.split_method,
    )

    print("\nFinal structured output:")
    print(json.dumps(result, indent=2))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved final JSON -> {args.out}")


if __name__ == "__main__":
    main()