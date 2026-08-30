"""
generate_sample_data.py

Creates a synthetic O-RAN Near-RT RIC KPI dataset (per-UE, per-TTI-window
aggregated metrics, similar in shape to what a KPM xApp reports over E2) so
that drift_detector.py can be run and demoed immediately, without waiting on
a real dataset download.

Two CSVs are produced:
  reference_window.csv  -- "normal" operating conditions
  current_window.csv    -- same KPIs, generated 30 minutes later, with a
                            realistic drift deliberately injected into two
                            KPIs (dl_throughput_mbps, prb_utilization_pct)
                            to simulate a cell going into congestion, while
                            two other KPIs (cqi, rsrp_dbm) are left
                            unchanged as a negative control.

This is for DEVELOPMENT/TESTING of the drift-detection pipeline only. For
your actual review results, use a real O-RAN KPI dataset (see README.md).
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_REF = 2000
N_CUR = 2000


def generate_reference(n=N_REF) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-08-01 00:00:00", periods=n, freq="1s"),
        "dl_throughput_mbps": RNG.normal(loc=45, scale=6, size=n).clip(0),
        "ul_throughput_mbps": RNG.normal(loc=12, scale=3, size=n).clip(0),
        "prb_utilization_pct": RNG.normal(loc=40, scale=8, size=n).clip(0, 100),
        "latency_ms": RNG.normal(loc=8, scale=1.5, size=n).clip(0),
        "cqi": RNG.normal(loc=11, scale=2, size=n).clip(1, 15),
        "rsrp_dbm": RNG.normal(loc=-85, scale=5, size=n),
        "packet_loss_pct": RNG.exponential(scale=0.3, size=n).clip(0, 100),
        "active_ues": RNG.poisson(lam=25, size=n),
    })


def generate_current(n=N_CUR) -> pd.DataFrame:
    """Same generative process, EXCEPT dl_throughput and PRB utilization are
    drifted (congestion scenario: cell load rises, throughput per UE falls),
    which also nudges latency up as a downstream effect. cqi and rsrp are
    left as a negative control (no drift injected)."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-08-01 00:33:20", periods=n, freq="1s"),
        # Drifted: mean shift + heavier tail (congestion -> lower/variable throughput)
        "dl_throughput_mbps": RNG.normal(loc=29, scale=9, size=n).clip(0),
        "ul_throughput_mbps": RNG.normal(loc=12, scale=3, size=n).clip(0),
        # Drifted: PRB utilization rises toward saturation
        "prb_utilization_pct": RNG.normal(loc=72, scale=10, size=n).clip(0, 100),
        # Mildly affected downstream of congestion
        "latency_ms": RNG.normal(loc=14, scale=3, size=n).clip(0),
        # Left unchanged (negative control)
        "cqi": RNG.normal(loc=11, scale=2, size=n).clip(1, 15),
        "rsrp_dbm": RNG.normal(loc=-85, scale=5, size=n),
        "packet_loss_pct": RNG.exponential(scale=0.9, size=n).clip(0, 100),
        "active_ues": RNG.poisson(lam=48, size=n),
    })
    return df


if __name__ == "__main__":
    ref = generate_reference()
    cur = generate_current()
    ref.to_csv("reference_window.csv", index=False)
    cur.to_csv("current_window.csv", index=False)
    print(f"Wrote reference_window.csv ({len(ref)} rows) and current_window.csv ({len(cur)} rows)")