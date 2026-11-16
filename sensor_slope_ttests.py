#!/usr/bin/env python3
"""Run sensor-wise one-sample tests on participant slope estimates."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp


# Use the participant slope table beside this script as input and save results there.
participant_slope_table_file = Path(__file__).resolve().with_name(
    "participant_slope_table_4pi.csv"
)
sensor_slope_ttest_results_file = Path(__file__).resolve().with_name(
    "sensor_slope_ttest_results_4pi.csv"
)
sensor_slope_ttest_topomap_file = Path(__file__).resolve().with_name(
    "sensor_slope_ttest_topomap_4pi.png"
)
n_participants_expected = 16

# Use the shared MEG data location from the phase-precession notebook by default.
meg_data_root = Path(os.environ.get(
    "MEG_DATA_ROOT",
    r"//rdp.arc.ucl.ac.uk/ritd-ag-project-rd01pz-dbush99/Katja Button Press Data",
))
reference_subject_id = "05"


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


def load_sensor_info(sensor_names):
    # Read channel locations from one participant's epochs without loading MEG data.
    subject_number = str(int(reference_subject_id))
    file_name = "Subject" + subject_number + "_spm_mne-epo.fif"
    default_epochs_file = (
        meg_data_root / ("s" + reference_subject_id) / file_name
    )
    reference_epochs_file = Path(os.environ.get(
        "MEG_REFERENCE_EPOCHS_FILE", str(default_epochs_file)
    ))

    # Match the phase-precession notebook's fallback for a FIF file in data_root.
    epochs_file_candidates = [reference_epochs_file]
    if "MEG_REFERENCE_EPOCHS_FILE" not in os.environ:
        epochs_file_candidates.append(meg_data_root / file_name)
    epochs_file = next(
        (candidate for candidate in epochs_file_candidates if candidate.is_file()),
        None,
    )
    if epochs_file is None:
        raise FileNotFoundError(
            "Could not find the reference epochs file. Set "
            "MEG_REFERENCE_EPOCHS_FILE or MEG_DATA_ROOT."
        )

    # MNE reads the epochs metadata only; preload=False avoids loading the MEG data.
    epochs = mne.read_epochs(epochs_file, preload=False, verbose="ERROR")
    missing_sensors = [
        sensor for sensor in sensor_names if sensor not in epochs.ch_names
    ]
    if missing_sensors:
        raise ValueError(
            "The reference epochs file is missing slope-table sensors: "
            + ", ".join(missing_sensors[:10])
        )

    # Keep the sensor order identical to the columns in the slope table.
    sensor_picks = mne.pick_channels(
        epochs.ch_names, include=list(sensor_names), ordered=True
    )
    return mne.pick_info(epochs.info, sensor_picks, copy=True)


def plot_sensor_slope_results(sensor_slope_test_results, sensor_info):
    
    # Align the test results with the sensor order in the MNE metadata.
    sensor_slope_test_results = sensor_slope_test_results.set_index(
        "sensor"
    ).loc[sensor_info["ch_names"]]
    mean_slope = sensor_slope_test_results[
        "mean_slope_rad_per_press"
    ].to_numpy(dtype=float)
    p_values = sensor_slope_test_results["p_value"].to_numpy(dtype=float)

    # Highlight sensors with the requested unadjusted p-value threshold.
    significant_sensor = np.isfinite(p_values) & (p_values < 0.01)
    sensor_types = [
        mne.channel_type(sensor_info, index)
        for index in range(len(sensor_info["ch_names"]))
    ]

    # Plot magnetometers and gradiometers separately, as their units differ.
    plotted_sensor_types = []
    for sensor_type in ("mag", "grad"):
        sensor_picks = np.array([
            index for index, kind in enumerate(sensor_types)
            if kind == sensor_type
        ])
        if not len(sensor_picks):
            continue

        type_info = mne.pick_info(sensor_info, sensor_picks, copy=True)
        fig, ax = plt.subplots(figsize=(7, 6))
        image, _ = mne.viz.plot_topomap(
            mean_slope[sensor_picks],
            type_info,
            ch_type=sensor_type,
            axes=ax,
            cmap="RdBu_r",
            contours=0,
            mask=significant_sensor[sensor_picks],
            mask_params={
                "marker": "o",
                "markerfacecolor": "yellow",
                "markeredgecolor": "black",
                "linestyle": "None",
                "linewidth": 0.7,
                "markersize": 7,
            },
            show=False,
        )

        # Label the topography and save one map for this sensor type.
        ax.set_title(
            "Mean slope; circles mark raw p < 0.01 (" + sensor_type + ")"
        )
        fig.colorbar(image, ax=ax, label="Mean slope (radians per press)")
        plt.tight_layout()
        plot_file = sensor_slope_ttest_topomap_file.with_name(
            sensor_slope_ttest_topomap_file.stem
            + "_" + sensor_type
            + sensor_slope_ttest_topomap_file.suffix
        )
        fig.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        plotted_sensor_types.append(sensor_type)
        print("Saved scalp map to:", plot_file)

    # MNE topomaps support the scalp MEG sensor types plotted above.
    if not plotted_sensor_types:
        raise ValueError("No magnetometer or gradiometer sensors were found.")


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

    significant_sensors = sensor_slope_ttest.pvalue < 0.01
    print(f"Number of significant sensors: {np.sum(significant_sensors)}")

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

    # Save the test results before creating the sensor maps.
    sensor_slope_test_results.to_csv(
        sensor_slope_ttest_results_file, index=False
    )
    print("Participants in source table:", len(participant_slope_table))
    print("Sensors tested:", len(sensor_slope_test_results))
    print("Saved results to:", sensor_slope_ttest_results_file)

    # Load channel geometry and plot the mean slopes with significant sensors marked.
    sensor_info = load_sensor_info(participant_slope_table.columns)
    plot_sensor_slope_results(sensor_slope_test_results, sensor_info)


if __name__ == "__main__":
    main()
