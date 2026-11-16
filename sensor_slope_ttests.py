#!/usr/bin/env python3
"""Run sensor-wise one-sample tests on participant slope estimates."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp


# Use the participant slope table beside this script as input and save results there.
participant_slope_table_file = Path(__file__).resolve().with_name(
    "participant_slope_table.csv"
)
sensor_slope_ttest_results_file = Path(__file__).resolve().with_name(
    "sensor_slope_ttest_results.csv"
)
n_participants_expected = 16


def benjamini_hochberg(p_values):
    """Adjust sensor-wise p-values while leaving missing values as NaN."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted_p_values = np.full(p_values.shape, np.nan)
    valid_indices = np.flatnonzero(np.isfinite(p_values))

    if not len(valid_indices):
        return adjusted_p_values

    # Rank valid p-values, scale by the number of tests, and enforce monotonicity.
    sort_order = np.argsort(p_values[valid_indices])
    sorted_p_values = p_values[valid_indices][sort_order]
    ranks = np.arange(1, len(sorted_p_values) + 1)
    sorted_adjusted_p_values = sorted_p_values * len(sorted_p_values) / ranks
    sorted_adjusted_p_values = np.minimum.accumulate(
        sorted_adjusted_p_values[::-1]
    )[::-1]
    adjusted_p_values[valid_indices[sort_order]] = np.clip(
        sorted_adjusted_p_values, 0, 1
    )
    return adjusted_p_values


def main():
    # Load one participant per row and one MEG sensor per column.
    participant_slope_table = pd.read_csv(participant_slope_table_file)

    # Check the expected participant count and unique sensor names.
    if participant_slope_table.shape[0] != n_participants_expected:
        raise ValueError(
            "Expected "
            + str(n_participants_expected)
            + " participant rows in "
            + participant_slope_table_file.name
            + ", found "
            + str(participant_slope_table.shape[0])
            + "."
        )
    if participant_slope_table.columns.duplicated().any():
        raise ValueError("The slope table contains duplicate sensor columns.")

    # Convert the table to numeric data and arrange it as sensors × participants.
    participant_slope_table = participant_slope_table.apply(
        pd.to_numeric, errors="coerce"
    )
    sensor_slope_by_participant = participant_slope_table.to_numpy(
        dtype=float
    ).T

    # Count the valid participant values available for each sensor.
    n_participants_by_sensor = np.isfinite(
        sensor_slope_by_participant
    ).sum(axis=1)
    if np.any(n_participants_by_sensor < 2):
        insufficient_sensors = participant_slope_table.columns[
            np.flatnonzero(n_participants_by_sensor < 2)
        ]
        raise ValueError(
            "At least two finite participant slopes are required for each sensor. "
            + "Insufficient data for: "
            + ", ".join(map(str, insufficient_sensors))
        )

    # Test each sensor's participant slopes against zero with a two-sided t-test.
    sensor_slope_ttest = ttest_1samp(
        sensor_slope_by_participant,
        popmean=0.0,
        axis=1,
        nan_policy="omit",
    )

    # Summarize each sensor and adjust p-values across the sensor tests.
    sensor_slope_test_results = pd.DataFrame({
        "sensor": participant_slope_table.columns,
        "n_participants": n_participants_by_sensor,
        "mean_slope_rad_per_press": np.nanmean(
            sensor_slope_by_participant, axis=1
        ),
        "std_slope_rad_per_press": np.nanstd(
            sensor_slope_by_participant, axis=1, ddof=1
        ),
        "t_statistic": sensor_slope_ttest.statistic,
        "p_value": sensor_slope_ttest.pvalue,
    })
    sensor_slope_test_results["p_value_fdr"] = benjamini_hochberg(
        sensor_slope_test_results["p_value"].to_numpy()
    )

    # Save the results and report how many tests were run.
    sensor_slope_test_results.to_csv(
        sensor_slope_ttest_results_file, index=False
    )
    print("Participants in source table:", len(participant_slope_table))
    print("Sensors tested:", len(sensor_slope_test_results))
    print("Saved results to:", sensor_slope_ttest_results_file)


if __name__ == "__main__":
    main()
