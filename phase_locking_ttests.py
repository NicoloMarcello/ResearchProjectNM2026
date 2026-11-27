#!/usr/bin/env python3
"""Run sensor-wise one-sample tests on participant phase-locking values."""

import argparse
import os
from pathlib import Path

import pandas as pd

from meg_data import load_sensor_info
from sensor_stats import sensor_wise_one_sample_ttest
from sensor_topomaps import plot_sensor_test_topomaps


script_folder = Path(__file__).resolve().parent
participant_phase_locking_table_file = script_folder / "participant_phase_locking_table.csv"
phase_locking_ttest_results_file = script_folder / "phase_locking_ttest_results.csv"
phase_locking_topomap_file = script_folder / "phase_locking_ttest_topomap.png"
n_participants_expected = 16
reference_subject_id = "05"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run one two-sided one-sample t-test per MEG sensor on "
            "participant mean phase-locking values."
        )
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
        help="Directory for the test results and topomaps.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load participant means: one row per participant and one column per sensor.
    participant_phase_locking_table = pd.read_csv(
        participant_phase_locking_table_file
    )
    if participant_phase_locking_table.shape[0] != n_participants_expected:
        raise ValueError(
            "Expected " + str(n_participants_expected)
            + " participant rows in "
            + participant_phase_locking_table_file.name
            + ", found "
            + str(participant_phase_locking_table.shape[0]) + "."
        )

    # Use the shared one-sample t-test and FDR implementation.
    phase_locking_test_results = sensor_wise_one_sample_ttest(
        participant_phase_locking_table,
        value_name="phase_locking",
        popmean=0.0,
    )
    phase_locking_test_results.to_csv(
        args.output_dir / phase_locking_ttest_results_file.name,
        index=False,
    )

    print("Participants in source table:", len(participant_phase_locking_table))
    print("Sensors tested:", len(phase_locking_test_results))
    print(
        "Sensors with raw p < 0.01:",
        int((phase_locking_test_results["p_value"] < 0.01).sum()),
    )
    print(
        "Sensors with FDR p < 0.05:",
        int((phase_locking_test_results["p_value_fdr"] < 0.05).sum()),
    )
    print(
        "Saved results to:",
        args.output_dir / phase_locking_ttest_results_file.name,
    )

    # Plot participant mean phase-locking strength and mark raw p < 0.01 sensors.
    sensor_info = load_sensor_info(
        participant_phase_locking_table.columns, args.data_root
    )
    plot_sensor_test_topomaps(
        phase_locking_test_results,
        sensor_info,
        args.output_dir / phase_locking_topomap_file.name,
        mean_column="mean_phase_locking",
        colorbar_label="Mean resultant length",
        title_prefix="Mean phase-locking strength",
        p_value_column="p_value",
        threshold=0.01,
    )


if __name__ == "__main__":
    main()
