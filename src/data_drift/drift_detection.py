"""
oran_drift_detector.py

Data drift detector for O-RAN AI models (xApps / rApps).

WHY THIS DESIGN (final algorithm choice)
-----------------------------------------
Data drift detection has three well-known families of tests, each with a
blind spot on its own:

  1. Kolmogorov-Smirnov (KS) test   -> catches shape/location changes in a
                                        single KPI, but is univariate only.
  2. Population Stability Index (PSI) -> the industry-standard metric used
                                        in production ML monitoring (banking,
                                        telecom OSS/BSS). Robust to sample
                                        size differences, gives a severity
                                        scale (not just a p-value), but is
                                        also univariate.
  3. Classifier Two-Sample Test (C2ST) -> trains a classifier to distinguish
                                        "reference" vs "current" rows using
                                        ALL features together. If the
                                        classifier can tell the two windows
                                        apart (AUC >> 0.5), the joint
                                        distribution has drifted -- this
                                        catches correlation/interaction
                                        drift between KPIs that univariate
                                        tests miss entirely (e.g. throughput
                                        and PRB utilization decoupling even
                                        though each marginal looks normal).
                                        This is the approach used by
                                        Lopez-Paz & Oquab (2017) and by
                                        production drift tools such as
                                        Evidently AI / NannyML.

No single test is "the" correct one for a live RAN, because KPI streams are
multivariate and xApps act on combinations of KPIs, not single features.
This detector therefore runs all three and fuses them:

  - Per-KPI drift  = KS test AND PSI (either one can flag a KPI; PSI severity
                      is reported so a human/agent can triage: PSI < 0.1 no
                      drift, 0.1-0.25 moderate, > 0.25 major -- this is the
                      standard PSI severity scale).
  - Dataset drift  = classifier AUC on held-out data. AUC ~0.5 = the model
                      can't tell reference from current -> no joint drift.
                      AUC significantly above 0.5 -> joint/interaction drift.
  - Overall verdict = drift flagged if EITHER a large fraction of KPIs drift
                      individually OR the joint classifier detects
                      interaction drift. This mirrors how the "Drift Type"
                      block in the Non-RT RIC evaluation system should feed
                      the Attribute Engine: a single boolean is not enough,
                      the agent needs per-KPI attribution to decide whether
                      to retrain one xApp's model or escalate.

OUTPUT
------
A single JSON-serializable dict (see `DriftReport`) designed to be dropped
straight onto the "Result" block in the Non-RT RIC pipeline and consumed by
a Decision Agent. It includes machine-actionable fields (`overall_drift`,
`recommended_action`, `severity`) as well as full per-KPI detail for
explainability / SHAP-style downstream analysis.

USAGE
-----
    python drift_detector.py --reference ref.csv --current cur.csv --out report.json

or import as a library:

    from drift_detector import ORANDriftDetector
    det = ORANDriftDetector(kpi_columns=["dl_throughput_mbps", "prb_util_pct", ...])
    report = det.detect(reference_df, current_df)
    report.to_json("report.json")
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Default thresholds. Tune these against your own validation window before
# trusting them on live traffic -- these are sane, widely used defaults, not
# guarantees for every deployment.
KS_ALPHA = 0.05                 # significance level for the KS test
KS_EFFECT_MIN = 0.10            # minimum KS D-statistic to count as practically
                                 # significant (guards against the classic large-N
                                 # problem: with thousands of samples, KS p-values
                                 # go to ~0 for microscopic, meaningless shifts).
                                 # Requiring D >= 0.1 alongside p < alpha keeps the
                                 # test both statistically AND practically significant.
PSI_MODERATE = 0.10             # PSI >= this -> "moderate" drift
PSI_MAJOR = 0.25                # PSI >= this -> "major" drift
KPI_DRIFT_RATIO_THRESHOLD = 0.3 # if >=30% of KPIs individually drift -> dataset flagged
CLASSIFIER_AUC_THRESHOLD = 0.65 # AUC above this -> joint/interaction drift
MIN_SAMPLES_PER_WINDOW = 30     # below this, statistics are unreliable


# --------------------------------------------------------------------------- #
# Data classes for structured, typed output
# --------------------------------------------------------------------------- #

@dataclass
class KPIDriftResult:
    kpi: str
    method: str = "KS+PSI"
    ks_statistic: float = 0.0
    p_value: float = 1.0
    psi: float = 0.0
    severity: str = "none"       # none | moderate | major
    drift_detected: bool = False
    reference_mean: float = 0.0
    current_mean: float = 0.0
    mean_shift_pct: float = 0.0


@dataclass
class MultivariateDriftResult:
    method: str = "classifier_two_sample_test"
    classifier: str = "RandomForestClassifier"
    auc: float = 0.5
    threshold: float = CLASSIFIER_AUC_THRESHOLD
    drift_detected: bool = False
    top_contributing_kpis: list[str] = field(default_factory=list)


@dataclass
class DriftReport:
    schema_version: str = "1.0"
    timestamp_utc: str = ""
    source: str = "oran-drift-detector"
    reference_window: dict[str, Any] = field(default_factory=dict)
    current_window: dict[str, Any] = field(default_factory=dict)
    kpi_results: list[KPIDriftResult] = field(default_factory=list)
    multivariate_result: MultivariateDriftResult = field(default_factory=MultivariateDriftResult)
    drifted_kpi_count: int = 0
    total_kpi_count: int = 0
    drifted_kpi_ratio: float = 0.0
    overall_drift: bool = False
    severity: str = "none"          # none | moderate | major
    recommended_action: str = "monitor"   # monitor | retrain_model | escalate_conflict_check
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_json(self, path: str | None = None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent)
        if path:
            with open(path, "w") as f:
                f.write(text)
        return text


# --------------------------------------------------------------------------- #
# Core detector
# --------------------------------------------------------------------------- #

class ORANDriftDetector:
    """
    Hybrid data-drift detector for O-RAN KPI streams (Non-RT RIC evaluation
    system / Member 1 module).

    Parameters
    ----------
    kpi_columns : list[str] | None
        Which numeric columns to treat as KPIs. If None, all numeric columns
        shared between reference and current dataframes are used.
    ks_alpha, psi_moderate, psi_major, kpi_drift_ratio_threshold,
    classifier_auc_threshold : override the module-level defaults per-instance.
    """

    def __init__(
        self,
        kpi_columns: list[str] | None = None,
        ks_alpha: float = KS_ALPHA,
        ks_effect_min: float = KS_EFFECT_MIN,
        psi_moderate: float = PSI_MODERATE,
        psi_major: float = PSI_MAJOR,
        kpi_drift_ratio_threshold: float = KPI_DRIFT_RATIO_THRESHOLD,
        classifier_auc_threshold: float = CLASSIFIER_AUC_THRESHOLD,
    ):
        self.kpi_columns = kpi_columns
        self.ks_alpha = ks_alpha
        self.ks_effect_min = ks_effect_min
        self.psi_moderate = psi_moderate
        self.psi_major = psi_major
        self.kpi_drift_ratio_threshold = kpi_drift_ratio_threshold
        self.classifier_auc_threshold = classifier_auc_threshold

    # ---- helpers ---------------------------------------------------------

    @staticmethod
    def _psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """
        Population Stability Index between two 1-D numeric arrays.
        PSI = sum( (cur_pct - ref_pct) * ln(cur_pct / ref_pct) ) over bins.
        Bin edges are derived from the reference window's quantiles so that
        each reference bin is roughly equal-population by construction.
        """
        reference = reference[~np.isnan(reference)]
        current = current[~np.isnan(current)]
        if len(reference) == 0 or len(current) == 0:
            return 0.0

        quantiles = np.linspace(0, 1, bins + 1)
        edges = np.unique(np.quantile(reference, quantiles))
        if len(edges) < 3:
            # Degenerate (near-constant) feature -- fall back to min/max split.
            edges = np.linspace(reference.min(), reference.max() + 1e-9, bins + 1)

        edges[0] = -np.inf
        edges[-1] = np.inf

        ref_counts, _ = np.histogram(reference, bins=edges)
        cur_counts, _ = np.histogram(current, bins=edges)

        ref_pct = np.where(ref_counts == 0, 1e-4, ref_counts / max(len(reference), 1))
        cur_pct = np.where(cur_counts == 0, 1e-4, cur_counts / max(len(current), 1))

        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        return round(psi, 4)

    @staticmethod
    def _severity_from_psi(psi: float, moderate: float, major: float) -> str:
        if psi >= major:
            return "major"
        if psi >= moderate:
            return "moderate"
        return "none"

    # ---- per-KPI univariate drift -----------------------------------------

    def _univariate_drift(self, ref: pd.DataFrame, cur: pd.DataFrame, kpis: list[str]) -> list[KPIDriftResult]:
        results = []
        for kpi in kpis:
            r = ref[kpi].dropna().to_numpy(dtype=float)
            c = cur[kpi].dropna().to_numpy(dtype=float)

            if len(r) < MIN_SAMPLES_PER_WINDOW or len(c) < MIN_SAMPLES_PER_WINDOW:
                results.append(KPIDriftResult(
                    kpi=kpi, method="insufficient_data",
                    drift_detected=False, severity="none",
                ))
                continue

            ks_stat, p_value = ks_2samp(r, c)
            psi = self._psi(r, c)
            severity = self._severity_from_psi(psi, self.psi_moderate, self.psi_major)

            ks_drift = (p_value < self.ks_alpha) and (ks_stat >= self.ks_effect_min)
            psi_drift = severity != "none"
            drift_detected = bool(ks_drift or psi_drift)

            ref_mean = float(np.mean(r))
            cur_mean = float(np.mean(c))
            mean_shift_pct = (
                round(100.0 * (cur_mean - ref_mean) / abs(ref_mean), 2)
                if ref_mean != 0 else 0.0
            )

            results.append(KPIDriftResult(
                kpi=kpi,
                method="KS+PSI",
                ks_statistic=round(float(ks_stat), 4),
                p_value=round(float(p_value), 6),
                psi=psi,
                severity=severity,
                drift_detected=drift_detected,
                reference_mean=round(ref_mean, 4),
                current_mean=round(cur_mean, 4),
                mean_shift_pct=mean_shift_pct,
            ))
        return results

    # ---- joint / multivariate drift (classifier two-sample test) ---------

    def _multivariate_drift(self, ref: pd.DataFrame, cur: pd.DataFrame, kpis: list[str]) -> MultivariateDriftResult:
        ref_x = ref[kpis].dropna()
        cur_x = cur[kpis].dropna()

        if len(ref_x) < MIN_SAMPLES_PER_WINDOW or len(cur_x) < MIN_SAMPLES_PER_WINDOW:
            return MultivariateDriftResult(auc=0.5, drift_detected=False)

        X = pd.concat([ref_x, cur_x], axis=0).to_numpy()
        y = np.concatenate([np.zeros(len(ref_x)), np.ones(len(cur_x))])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=42, n_jobs=-1
        )

        n_splits = min(5, min(np.bincount(y.astype(int))))
        n_splits = max(n_splits, 2)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        proba = cross_val_predict(clf, X_scaled, y, cv=cv, method="predict_proba")[:, 1]
        auc = float(roc_auc_score(y, proba))

        # Feature importance from a full-data fit, used only to explain which
        # KPIs the classifier leaned on (not for the AUC computation itself).
        clf.fit(X_scaled, y)
        importances = clf.feature_importances_
        top_idx = np.argsort(importances)[::-1][:3]
        top_kpis = [kpis[i] for i in top_idx]

        return MultivariateDriftResult(
            auc=round(auc, 4),
            threshold=self.classifier_auc_threshold,
            drift_detected=bool(auc >= self.classifier_auc_threshold),
            top_contributing_kpis=top_kpis,
        )

    # ---- public entry point ------------------------------------------------

    def detect(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        reference_label: str = "reference_window",
        current_label: str = "current_window",
    ) -> DriftReport:
        if self.kpi_columns is not None:
            kpis = [c for c in self.kpi_columns if c in reference_df.columns and c in current_df.columns]
        else:
            numeric_ref = set(reference_df.select_dtypes(include=[np.number]).columns)
            numeric_cur = set(current_df.select_dtypes(include=[np.number]).columns)
            kpis = sorted(numeric_ref & numeric_cur)

        if not kpis:
            raise ValueError("No shared numeric KPI columns found between reference and current data.")

        kpi_results = self._univariate_drift(reference_df, current_df, kpis)
        mv_result = self._multivariate_drift(reference_df, current_df, kpis)

        drifted = [r for r in kpi_results if r.drift_detected]
        drifted_ratio = len(drifted) / len(kpi_results) if kpi_results else 0.0

        ratio_flag = drifted_ratio >= self.kpi_drift_ratio_threshold
        overall_drift = bool(ratio_flag or mv_result.drift_detected)

        max_psi = max((r.psi for r in kpi_results), default=0.0)
        severity = self._severity_from_psi(max_psi, self.psi_moderate, self.psi_major)
        if mv_result.drift_detected and severity == "none":
            severity = "moderate"

        if not overall_drift:
            action = "monitor"
        elif ratio_flag and not mv_result.drift_detected:
            action = "retrain_model"
        elif mv_result.drift_detected and not ratio_flag:
            # Joint distribution changed but no single KPI drifted strongly on
            # its own -- classic signature of xApps interacting rather than a
            # genuine environment shift. Route to conflict-check rather than
            # straight to retraining.
            action = "escalate_conflict_check"
        else:
            action = "retrain_model"

        report = DriftReport(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            reference_window={"label": reference_label, "n_samples": int(len(reference_df))},
            current_window={"label": current_label, "n_samples": int(len(current_df))},
            kpi_results=kpi_results,
            multivariate_result=mv_result,
            drifted_kpi_count=len(drifted),
            total_kpi_count=len(kpi_results),
            drifted_kpi_ratio=round(drifted_ratio, 4),
            overall_drift=overall_drift,
            severity=severity,
            recommended_action=action,
            notes=(
                "overall_drift = (drifted_kpi_ratio >= "
                f"{self.kpi_drift_ratio_threshold}) OR (classifier AUC >= {self.classifier_auc_threshold}). "
                "recommended_action='escalate_conflict_check' means the joint distribution shifted "
                "without individual KPIs drifting -- check xApp conflict module before retraining."
            ),
        )
        return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="O-RAN data drift detector (KS + PSI + classifier two-sample test)."
    )

    parser.add_argument(
        "--reference",
        required=True,
        help="Path to reference-window CSV."
    )

    parser.add_argument(
        "--current",
        required=True,
        help="Path to current-window CSV."
    )

    parser.add_argument(
        "--kpis",
        nargs="*",
        default=None,
        help="Explicit list of KPI columns (default: all shared numeric columns)."
    )

    parser.add_argument(
        "--out",
        default=None,
        help="Path to write the JSON report."
    )

    args = parser.parse_args()

    # Load datasets
    ref_df = pd.read_csv(args.reference)
    cur_df = pd.read_csv(args.current)

    # Create detector
    detector = ORANDriftDetector(kpi_columns=args.kpis)

    # Detect drift
    report = detector.detect(
        ref_df,
        cur_df,
        reference_label=args.reference,
        current_label=args.current
    )

    # Display clean drift detection result
    for result in report.kpi_results:

        print("\n====================================================")
        print("                 DATA DRIFT DETECTION")
        print("====================================================")

        if result.kpi == "dl_throughput_mbps":
            kpi_name = "Throughput"
        else:
            kpi_name = result.kpi.replace("_", " ").title()

        print(f"KPI                 : {kpi_name}")
        print("Detection Method    : Kolmogorov-Smirnov Test")
        print()
        print(f"KS Statistic        : {result.ks_statistic:.4f}")
        print(f"p-value             : {result.p_value:.4f}")
        print(f"Drift Threshold     : {KS_ALPHA}")
        print()
        print(f"Drift Score         : {result.ks_statistic:.4f}")
        print(
            f"Status              : "
            f"{'DRIFT DETECTED' if result.drift_detected else 'NO DRIFT'}"
        )

        print("====================================================")

    # Save full JSON report if requested
    if args.out:
        report.to_json(args.out)
        print(f"\nFull JSON report saved to: {args.out}")


if __name__ == "__main__":
    main()