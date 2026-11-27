#!/usr/bin/env python3
"""Run sensor-wise one-sample tests on participant slope estimates."""

import argparse
import os
from pathlib import Path

import pandas as pd

from meg_data import load_sensor_info
from sensor_stats import sensor_wise_one_sample_ttest
from sensor_topomaps import plot_sensor_test_topomaps


script_folder = Path(__file__).resolve().parent
participant_slope_table_file = script_folder / "participant_slope_table_4pi.csv"
sensor_slope_ttest_results_file = script_folder / "sensor_slope_ttest_results_4pi.csv"
sensor_slope_ttest_topomap_file = script_folder / "sensor_slope_ttest_topomap_4pi.png"
n_participants_expected = 16
reference_subject_id = "05"


def main():
    parser = argparse.ArgumentParser(
        description="Run one-sample t-tests on participant slope values by sensor."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ["MEG_DATA_ROOT"])
        if os.environ.get("MEG_DATA_ROOT") else None,
        help="MEG export directory used to load sensor locations.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_folder,
        help="Directory for test results and topomaps.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load one participant per row and one MEG sensor per column.
    participant_slope_table = pd.read_csv(participant_slope_table_file)
    if participant_slope_table.shape[0] != n_participants_expected:
        raise ValueError(
            "Expected " + str(n_participants_expected)
            + " participant rows in " + participant_slope_table_file.name
            + ", found " + str(participant_slope_table.shape[0]) + "."
        )

    # Use the shared one-sample t-test and FDR implementation.
    sensor_slope_test_results = sensor_wise_one_sample_ttest(
        participant_slope_table,
        value_name="slope_rad_per_press",
        popmean=0.0,
    )
    result_file = args.output_dir / sensor_slope_ttest_results_file.name
    sensor_slope_test_results.to_csv(result_file, index=False)

    print("Participants in source table:", len(participant_slope_table))
    print("Sensors tested:", len(sensor_slope_test_results))
    print(
        "Sensors with raw p < 0.01:",
        int((sensor_slope_test_results["p_value"] < 0.01).sum()),
    )
    print("Saved results to:", result_file)

    # Load channel geometry and plot the mean slopes with raw p < 0.01 marked.
    sensor_info = load_sensor_info(
        participant_slope_table.columns, args.data_root
    )
    plot_sensor_test_topomaps(
        sensor_slope_test_results,
        sensor_info,
        args.output_dir / sensor_slope_ttest_topomap_file.name,
        mean_column="mean_slope_rad_per_press",
        colorbar_label="Mean slope (radians per press)",
        title_prefix="Mean slope",
        p_value_column="p_value",
        threshold=0.01,
    )


if __name__ == "__main__":
    main()
