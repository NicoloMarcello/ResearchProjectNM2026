#!/usr/bin/env python3
"""Test participant-level MEG sensor slopes against zero."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp


SLOPE_TABLE_FILE = Path(__file__).resolve().with_name("participant_slope_table.csv")
OUTPUT_FILE = Path(__file__).resolve().with_name("sensor_slope_ttest_results.csv")
EXPECTED_N_PARTICIPANTS = 16


def benjamini_hochberg(p_values):
    """Return Benjamini-Hochberg adjusted p-values, preserving NaNs."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan)
    valid_indices = np.flatnonzero(np.isfinite(p_values))

    if not len(valid_indices):
        return adjusted

    order = np.argsort(p_values[valid_indices])
    sorted_p = p_values[valid_indices][order]
    ranks = np.arange(1, len(sorted_p) + 1)
    sorted_adjusted = sorted_p * len(sorted_p) / ranks
    sorted_adjusted = np.minimum.accumulate(sorted_adjusted[::-1])[::-1]
    adjusted[valid_indices[order]] = np.clip(sorted_adjusted, 0, 1)
    return adjusted


def main():
    # Each input row is one participant and each column is one MEG sensor.
    slope_table = pd.read_csv(SLOPE_TABLE_FILE)
    if slope_table.shape[0] != EXPECTED_N_PARTICIPANTS:
        raise ValueError(
            "Expected "
            + str(EXPECTED_N_PARTICIPANTS)
            + " participant rows in "
            + str(SLOPE_TABLE_FILE.name)
            + ", found "
            + str(slope_table.shape[0])
            + "."
        )
    if slope_table.columns.duplicated().any():
        raise ValueError("The slope table contains duplicate sensor columns.")

    # Convert the participant × sensor table to sensors × participants for axis=1.
    slope_table = slope_table.apply(pd.to_numeric, errors="coerce")
    slope_by_sensor = slope_table.to_numpy(dtype=float).T
    n_participants = np.isfinite(slope_by_sensor).sum(axis=1)
    if np.any(n_participants < 2):
        bad_sensors = slope_table.columns[np.flatnonzero(n_participants < 2)]
        raise ValueError(
            "At least two finite participant slopes are required for each sensor. "
            + "Insufficient data for: "
            + ", ".join(map(str, bad_sensors))
        )

    # Run a two-sided one-sample t-test for every sensor against zero.
    test_results = ttest_1samp(
        slope_by_sensor,
        popmean=0.0,
        axis=1,
        nan_policy="omit",
    )

    # Summarize estimates and adjust for testing across all MEG sensors.
    results = pd.DataFrame({
        "sensor": slope_table.columns,
        "n_participants": n_participants,
        "mean_slope_rad_per_press": np.nanmean(slope_by_sensor, axis=1),
        "std_slope_rad_per_press": np.nanstd(slope_by_sensor, axis=1, ddof=1),
        "t_statistic": test_results.statistic,
        "p_value": test_results.pvalue,
    })
    results["p_value_fdr"] = benjamini_hochberg(results["p_value"].to_numpy())
    results.to_csv(OUTPUT_FILE, index=False)

    print("Participants in each test:", EXPECTED_N_PARTICIPANTS)
    print("Sensors tested:", len(results))
    print("Saved results to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
