<<<<<<< Updated upstream
=======
"""
oran_drift_detector.py

Data Drift Detection module (Member 1) for the O-RAN Non-RT RIC pipeline.

pip install pandas numpy scipy matplotlib

--------------------------------------------------------------------------
WHY THREE METHODS
--------------------------------------------------------------------------
- KS test          : classic hypothesis test for "did the shape/location of
                      the distribution change" -> gives a p-value.
- Wasserstein dist. : "how far did the distribution actually move" -> gives a
                      magnitude, in the KPI's own units, that KS's p-value
                      does not (p-values collapse to ~0 at large N even for
                      tiny shifts).
- PSI               : the industry-standard production-monitoring metric,
                      binned on the reference window, robust to sample-size
                      differences, and gives a human-readable severity scale.

No single test is treated as ground truth here; they are fused into one
Data Drift Score, and a human-readable "reason" documents which tests
agreed. All weights and thresholds below are INITIAL, CONFIGURABLE
parameters, not universal constants -- they should be calibrated against a
labeled validation window before being trusted in production.

--------------------------------------------------------------------------
WINDOWING NOTE (IMPORTANT)
--------------------------------------------------------------------------
This prototype splits a single static CSV into a reference and a current
window using a reproducible random or positional split (see
`prepare_windows`), because the sample dataset has no reliable timestamp
column to order rows by. THIS IS A PROTOTYPE SIMPLIFICATION.
In the final O-RAN deployment, once timestamped streaming KPM data is
available (e.g. from the Non-RT RIC data lake), the reference window MUST
be a fixed historical/baseline period and the current window MUST be the
most recent chronological period (e.g. last N minutes/hours), NOT a random
split. Random splitting is only valid here because we are demonstrating the
statistical methodology on a static file.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
import matplotlib.pyplot as plt


# ============================================================================
# Configuration (all initial / configurable, not universal constants)
# ============================================================================

DEFAULT_KPI = "DRB.UEThpDl"          # primary KPI; change freely, code is modular
MIN_SAMPLES_PER_WINDOW = 30

# --- KS test ---
KS_ALPHA = 0.05                      # significance level

# --- Wasserstein ---
# Normalization method for turning a raw Wasserstein distance (in the KPI's
# own units) into a [0,1] "component" score. "iqr" divides the raw distance
# by the reference window's inter-quartile range, which is a scale-free,
# distribution-aware normalizer. "range" divides by reference max-min.
# WASSERSTEIN_SATURATION controls at what multiple of that scale the
# component score saturates to 1.0 -- purely a configurable heuristic.
WASSERSTEIN_NORM_METHOD = "iqr"      # "iqr" | "range"
WASSERSTEIN_SATURATION = 1.0         # component reaches 1.0 when raw_dist >= SATURATION * scale

# --- PSI ---
PSI_BINS = 10
PSI_EPSILON = 1e-4                   # safe floor to avoid log(0) / div-by-0
PSI_MODERATE = 0.10                  # PSI >= this -> MODERATE_DRIFT
PSI_MAJOR = 0.25                     # PSI >= this -> SIGNIFICANT_DRIFT
# PSI component is normalized to [0,1] by capping at PSI_SATURATION.
PSI_SATURATION = 0.50

# --- Fusion weights (initial parameters, calibrate later) ---
WEIGHT_KS = 0.4
WEIGHT_WASSERSTEIN = 0.3
WEIGHT_PSI = 0.3

# --- Final score thresholds (initial, configurable) ---
SCORE_NO_DRIFT = 0.30                # score <  this            -> NO_DRIFT
SCORE_DRIFT = 0.60                   # score >= this            -> DRIFT_DETECTED
                                      # in between                -> WARNING


# ============================================================================
# Small result containers (internal use only -- NOT the final small JSON)
# ============================================================================

@dataclass
class MethodResult:
    name: str
    raw_value: float
    component_score: float           # normalized to [0,1]
    individually_flagged: bool
    detail: str


@dataclass
class DriftAnalysis:
    kpi: str
    methods: list[MethodResult] = field(default_factory=list)
    drift_score: float = 0.0
    status: str = "NO_DRIFT"
    reason: str = ""


# ============================================================================
# 1. Load data
# ============================================================================

def load_data(csv_path: str, kpi_column: str) -> pd.DataFrame:
    """
    Load the O-RAN KPM CSV and return a cleaned single-column-relevant
    DataFrame. Handles missing values / non-numeric junk safely.
    """
    df = pd.read_csv(csv_path)

    if kpi_column not in df.columns:
        raise ValueError(
            f"KPI column '{kpi_column}' not found in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    # Coerce to numeric; anything unparsable becomes NaN, then dropped.
    df[kpi_column] = pd.to_numeric(df[kpi_column], errors="coerce")
    before = len(df)
    df = df.dropna(subset=[kpi_column])
    df = df[np.isfinite(df[kpi_column])]
    after = len(df)

    if after == 0:
        raise ValueError(f"No valid numeric values left for KPI '{kpi_column}' after cleaning.")
    if after < before:
        print(f"[load_data] Dropped {before - after} invalid/missing rows for '{kpi_column}'.")

    return df


# ============================================================================
# 2. Prepare reference / current windows
# ============================================================================

def prepare_windows(
    df: pd.DataFrame,
    kpi_column: str,
    split_ratio: float = 0.5,
    method: str = "positional",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a single static dataset into reference (baseline) and current
    (monitoring) windows for the KPI of interest.

    method="positional" (default): first `split_ratio` fraction of rows (in
        the CSV's existing row order) = reference, remainder = current. This
        is a reproducible, order-preserving stand-in for a chronological
        split when no reliable timestamp column exists.
    method="random": reproducible random split (fixed random_state) purely
        for demonstrating the statistical methods on a static file.

    *** In production with timestamped streaming KPM data, replace this
    entirely with a true chronological split: reference = fixed historical
    baseline period, current = most recent period. ***
    """
    values = df[kpi_column].to_numpy(dtype=float)
    n = len(values)
    split_idx = int(n * split_ratio)

    if method == "random":
        rng = np.random.RandomState(random_state)
        idx = rng.permutation(n)
        ref_idx, cur_idx = idx[:split_idx], idx[split_idx:]
        reference, current = values[ref_idx], values[cur_idx]
    elif method == "positional":
        reference, current = values[:split_idx], values[split_idx:]
    else:
        raise ValueError(f"Unknown split method: {method}")

    if len(reference) < MIN_SAMPLES_PER_WINDOW or len(current) < MIN_SAMPLES_PER_WINDOW:
        raise ValueError(
            f"Insufficient samples per window (need >= {MIN_SAMPLES_PER_WINDOW} each); "
            f"got reference={len(reference)}, current={len(current)}."
        )

    return reference, current


# ============================================================================
# 3. KS test
# ============================================================================

def ks_test(reference: np.ndarray, current: np.ndarray, alpha: float = KS_ALPHA) -> MethodResult:
    stat, p_value = ks_2samp(reference, current)
    flagged = bool(p_value < alpha)
    # Component score = the KS D-statistic itself; it is already in [0,1]
    # by definition (max distance between two CDFs).
    component = float(np.clip(stat, 0.0, 1.0))
    detail = f"D={stat:.4f}, p={p_value:.4g}"
    return MethodResult("KS", raw_value=float(stat), component_score=component,
                         individually_flagged=flagged, detail=detail + f" (alpha={alpha})")


# ============================================================================
# 4. Wasserstein distance
# ============================================================================

def wasserstein_test(
    reference: np.ndarray,
    current: np.ndarray,
    norm_method: str = WASSERSTEIN_NORM_METHOD,
    saturation: float = WASSERSTEIN_SATURATION,
) -> MethodResult:
    """
    Raw Wasserstein distance is in the KPI's own units, so it is normalized
    to a dimensionless [0,1] component using a scale derived from the
    reference window (IQR by default). This normalization method and the
    saturation point are configurable heuristics -- a raw Wasserstein value
    does NOT universally mean "drift" on its own; scale matters.
    """
    raw = float(wasserstein_distance(reference, current))

    if norm_method == "iqr":
        q75, q25 = np.percentile(reference, [75, 25])
        scale = (q75 - q25) or (np.std(reference) or 1.0)
    elif norm_method == "range":
        scale = (reference.max() - reference.min()) or (np.std(reference) or 1.0)
    else:
        raise ValueError(f"Unknown normalization method: {norm_method}")

    component = float(np.clip(raw / (saturation * scale), 0.0, 1.0))
    # No fixed universal "flag" threshold for Wasserstein alone; treat it as
    # individually flagged once its normalized component is meaningfully
    # large (component >= 0.5 is a configurable heuristic, mirroring the
    # midpoint of KS's effect-size logic), used only for the "reason" text.
    flagged = component >= 0.5
    detail = f"raw={raw:.4f}, scale({norm_method})={scale:.4f}, component={component:.4f}"
    return MethodResult("Wasserstein", raw_value=raw, component_score=component,
                         individually_flagged=flagged, detail=detail)


# ============================================================================
# 5. PSI
# ============================================================================

def calculate_psi(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = PSI_BINS,
    epsilon: float = PSI_EPSILON,
    moderate: float = PSI_MODERATE,
    major: float = PSI_MAJOR,
    saturation: float = PSI_SATURATION,
) -> MethodResult:
    """
    PSI computed with bin edges derived from the REFERENCE distribution's
    quantiles, then applied unchanged to the current distribution (as
    required). Zero-probability bins are floored at `epsilon` to keep the
    log ratio finite/safe.
    """
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 3:
        edges = np.linspace(reference.min(), reference.max() + 1e-9, bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = np.where(ref_counts == 0, epsilon, ref_counts / max(len(reference), 1))
    cur_pct = np.where(cur_counts == 0, epsilon, cur_counts / max(len(current), 1))

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

    if psi >= major:
        interpretation = "SIGNIFICANT_DRIFT"
    elif psi >= moderate:
        interpretation = "MODERATE_DRIFT"
    else:
        interpretation = "NO_DRIFT"

    component = float(np.clip(psi / saturation, 0.0, 1.0))
    flagged = interpretation != "NO_DRIFT"
    detail = f"PSI={psi:.4f} -> {interpretation} (heuristic thresholds: mod>={moderate}, major>={major})"
    return MethodResult("PSI", raw_value=psi, component_score=component,
                         individually_flagged=flagged, detail=detail)


# ============================================================================
# 6. Combined Data Drift Score
# ============================================================================

def calculate_drift_score(
    ks_result: MethodResult,
    wasserstein_result: MethodResult,
    psi_result: MethodResult,
    w_ks: float = WEIGHT_KS,
    w_wasserstein: float = WEIGHT_WASSERSTEIN,
    w_psi: float = WEIGHT_PSI,
    no_drift_max: float = SCORE_NO_DRIFT,
    drift_min: float = SCORE_DRIFT,
) -> tuple[float, str]:
    """
    Weighted fusion of the three normalized [0,1] components.
    Weights (0.4 / 0.3 / 0.3 by default) are INITIAL parameters, not fixed
    scientific constants -- calibrate against validation data later.
    """
    total_weight = w_ks + w_wasserstein + w_psi
    score = (
        w_ks * ks_result.component_score
        + w_wasserstein * wasserstein_result.component_score
        + w_psi * psi_result.component_score
    ) / total_weight
    score = float(np.clip(score, 0.0, 1.0))

    if score < no_drift_max:
        status = "NO_DRIFT"
    elif score < drift_min:
        status = "WARNING"
    else:
        status = "DRIFT_DETECTED"

    return round(score, 4), status


# ============================================================================
# 7. Dynamic reason generation
# ============================================================================

def generate_reason(ks_result: MethodResult, wasserstein_result: MethodResult, psi_result: MethodResult) -> str:
    """
    Builds a short, human-readable reason string dynamically from which
    methods actually flagged drift on this run -- never hardcoded.
    """
    flagged_names = []
    if ks_result.individually_flagged:
        flagged_names.append("KS")
    if wasserstein_result.individually_flagged:
        flagged_names.append("Wasserstein")
    if psi_result.individually_flagged:
        flagged_names.append("PSI")

    phrase = {
        "KS": "significant distribution shift",
        "Wasserstein": "substantial distribution shift",
        "PSI": "significant population instability",
    }

    if len(flagged_names) == 0:
        return "Insufficient consistent evidence of drift"

    if len(flagged_names) == 1:
        m = flagged_names[0]
        return f"{m} test indicates {phrase[m]}" if m == "KS" else f"{m} indicates {phrase[m]}"

    if len(flagged_names) == len(["KS", "Wasserstein", "PSI"]):
        return "KS, Wasserstein and PSI consistently indicate distribution shift"

    # exactly two agree
    joined = " and ".join(flagged_names)
    return f"{joined} indicate significant distribution shift"


# ============================================================================
# 8. Top-level orchestration
# ============================================================================

def detect_data_drift(
    csv_path: str | None = None,
    kpi_column: str = DEFAULT_KPI,
    split_ratio: float = 0.5,
    split_method: str = "positional",
    random_state: int = 42,
    make_plots: bool = True,
    plot_prefix: str = "drift",
    reference_csv: str | None = None,
    current_csv: str | None = None,
    reference_data=None,
    current_data=None,
) -> dict[str, Any]:
    
    """
    Two supported input modes:

    1. Single-file mode (csv_path set, reference_csv/current_csv left None):
       one static CSV is split into reference/current windows via
       `prepare_windows` (positional or random split -- see that function's
       docstring for the chronological-windowing caveat).

    2. Two-file mode (reference_csv AND current_csv both set): the reference
       and current windows are already separate files -- e.g. a genuinely
       time-ordered "baseline period" file and a "monitoring period" file
       (this is the shape produced by generate_sample_data.py, and the shape
       a real chronological O-RAN pipeline would hand this module). No
       splitting is performed; each file is loaded and cleaned independently.
    """

    


    if reference_data is not None and current_data is not None:
        reference = np.asarray(reference_data, dtype=float)
        current = np.asarray(current_data, dtype=float)

        reference = reference[np.isfinite(reference)]
        current = current[np.isfinite(current)]

        if len(reference) < MIN_SAMPLES_PER_WINDOW:
            raise ValueError(
                f"Insufficient reference samples: {len(reference)}"
            )

        if len(current) < MIN_SAMPLES_PER_WINDOW:
            raise ValueError(
                f"Insufficient current samples: {len(current)}"
            )

    elif reference_csv and current_csv:

        ref_df = load_data(reference_csv, kpi_column)
        cur_df = load_data(current_csv, kpi_column)

        reference = ref_df[kpi_column].to_numpy(dtype=float)
        current = cur_df[kpi_column].to_numpy(dtype=float)

        if len(reference) < MIN_SAMPLES_PER_WINDOW or len(current) < MIN_SAMPLES_PER_WINDOW:
            raise ValueError(
                f"Insufficient samples per window (need >= {MIN_SAMPLES_PER_WINDOW} each); "
                f"got reference={len(reference)}, current={len(current)}."
            )

    elif csv_path:

        df = load_data(csv_path, kpi_column)

        reference, current = prepare_windows(
            df,
            kpi_column,
            split_ratio=split_ratio,
            method=split_method,
            random_state=random_state
        )

    else:
        raise ValueError(
            "Provide either streaming data, CSV reference/current files, "
            "or a single CSV file."
        )
    ks_result = ks_test(reference, current)
    wasserstein_result = wasserstein_test(reference, current)
    psi_result = calculate_psi(reference, current)

    score, status = calculate_drift_score(ks_result, wasserstein_result, psi_result)
    reason = generate_reason(ks_result, wasserstein_result, psi_result)

    _print_analysis_table(kpi_column, ks_result, wasserstein_result, psi_result, score, status)

    if make_plots:
        _plot_distributions(reference, current, kpi_column, plot_prefix)
        _plot_components(ks_result, wasserstein_result, psi_result, score, plot_prefix)

    final_output = {
        "module": "data_drift",
        "kpi": kpi_column,
        "drift_score": score,
        "status": status,
        "reason": reason,
    }
    return final_output


# ============================================================================
# Presentation helpers: analysis table + plots
# ============================================================================

def _print_analysis_table(
    kpi: str,
    ks_result: MethodResult,
    wasserstein_result: MethodResult,
    psi_result: MethodResult,
    score: float,
    status: str,
) -> None:
    print(f"\nData Drift Analysis -- KPI: {kpi}")
    print("-" * 58)
    print(f"{'Method':<16}{'Value':<20}{'Result'}")
    print("-" * 58)
    print(f"{'KS':<16}{ks_result.detail:<20}{'DRIFT' if ks_result.individually_flagged else 'NO DRIFT'}")
    print(f"{'Wasserstein':<16}{f'{wasserstein_result.raw_value:.4f}':<20}{'DRIFT' if wasserstein_result.individually_flagged else 'NO DRIFT'}")
    print(f"{'PSI':<16}{f'{psi_result.raw_value:.4f}':<20}{'DRIFT' if psi_result.individually_flagged else 'NO DRIFT'}")
    print("-" * 58)
    print(f"{'Final Score':<16}{f'{score:.4f}':<20}{status}")
    print("-" * 58)


def _plot_distributions(reference: np.ndarray, current: np.ndarray, kpi: str, prefix: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(reference, bins=30, alpha=0.55, label="Reference window", color="#4C72B0", density=True)
    ax.hist(current, bins=30, alpha=0.55, label="Current window", color="#DD8452", density=True)
    ax.set_title(f"Distribution Comparison: {kpi}")
    ax.set_xlabel(kpi)
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    out_path = f"{prefix}_distribution_comparison.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[plot] Saved distribution comparison -> {out_path}")


def _plot_components(
    ks_result: MethodResult,
    wasserstein_result: MethodResult,
    psi_result: MethodResult,
    score: float,
    prefix: str,
) -> None:
    labels = ["KS", "Wasserstein", "PSI", "Final Score"]
    values = [
        ks_result.component_score,
        wasserstein_result.component_score,
        psi_result.component_score,
        score,
    ]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Normalized score [0,1]")
    ax.set_title("Drift Detector Components")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    out_path = f"{prefix}_components.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[plot] Saved component bar chart -> {out_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="O-RAN KPM Data Drift Detector (KS + Wasserstein + PSI).")
    parser.add_argument("--csv", default=None,
                         help="Single-file mode: path to one CSV to split into reference/current windows.")
    parser.add_argument("--reference", default=None,
                         help="Two-file mode: path to the reference/baseline-window CSV.")
    parser.add_argument("--current", default=None,
                         help="Two-file mode: path to the current/monitoring-window CSV.")
    parser.add_argument("--kpi", default=DEFAULT_KPI, help="KPI column to analyze.")
    parser.add_argument("--split-ratio", type=float, default=0.5,
                         help="Single-file mode only: fraction of rows used as reference window.")
    parser.add_argument("--split-method", choices=["positional", "random"], default="positional",
                         help="Single-file mode only: 'positional' preserves row order (recommended stand-in "
                              "for chronological data); 'random' is a reproducible random split.")
    parser.add_argument("--out", default=None, help="Optional path to save the final small JSON result.")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots.")
    args = parser.parse_args()

    if not args.csv and not (args.reference and args.current):
        args.csv = "KPM Metrics-Table.csv"  # default single-file mode

    result = detect_data_drift(
        csv_path=args.csv,
        kpi_column=args.kpi,
        split_ratio=args.split_ratio,
        split_method=args.split_method,
        make_plots=not args.no_plots,
        reference_csv=args.reference,
        current_csv=args.current,
    )

    print("\nFinal structured output:")
    print(json.dumps(result, indent=2))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved final JSON -> {args.out}")


if __name__ == "__main__":
    main()
>>>>>>> Stashed changes
