"""Run sensor-wise one-sample tests on participant theta-power changes.

The power calculation follows MEG_powerplot_all.ipynb: theta-band power in
button-press epochs is baseline-corrected in dB and averaged across valid presses
within each participant. Each participant contributes one value per sensor to the
test, avoiding treating repeated presses as independent observations. The script
tests each sensor's participant values against 0 dB and saves per-sensor Q-Q plots.
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm

from meg_data import load_participant_data
from meg_power import epochs_maker, mean_band_power, press_time_flattener
from sensor_stats import sensor_wise_one_sample_ttest
from sensor_topomaps import plot_sensor_test_topomaps


# Use data paths and outputs in the directory.
script_folder = Path(__file__).resolve().parent
participant_power_table_file = script_folder / "participant_power_change_dB.csv"
power_ttest_results_file = script_folder / "power_ttest_results_4pi.csv"
power_ttest_qq_file = script_folder / "power_ttest_qqplots_4pi.pdf"
power_ttest_topomap_file = script_folder / "power_ttest_topomap_4pi.png"

# Match the participant list and power settings in MEG_powerplot_all.ipynb.
sub_ID = [
    "05", "06", "08", "09", "10", "13", "14", "15",
    "16", "17", "18", "20", "21", "24", "25", "28",
]
excluded_subjects = []
theta_freq_band = np.array([2.0, 5.0])       # theta band edges in Hz
press_time_wind = np.array([-0.5, 0.5])      # epoch window around each button press
baseline_time_wind = np.array([-1.0, 0.0])   # baseline window before each trial
mean_time_offset = -0.0468                   # MEG trigger offset in seconds
sens_delay = 2.8 + mean_time_offset          # cue time relative to epoch start
n_press_slots = 5


def participant_theta_power(id, data_root):
    # Load the participant's behavioral press times and matching MEG epochs.
    participant_data = load_participant_data(
        data_root,
        id,
        preload=True,
        verbose="ERROR",
        include_press_ids=False,
    )
    press_times = participant_data["press_times"]
    epochs = participant_data["epochs"]
    MEG_sens = participant_data["meg_sensors"]
    sensor_info = participant_data["sensor_info"]
    sfreq = participant_data["sfreq"]

    # Reshape the MEG epochs into a continuous-style RawArray.
    MEG_data = epochs.get_data(picks=MEG_sens)
    n_epochs, n_channels, n_times = MEG_data.shape
    trial_time = n_times / sfreq
    MEG_data = MEG_data.transpose(1, 0, 2).reshape(
        n_channels, n_epochs * n_times
    )
    raw = mne.io.RawArray(MEG_data, sensor_info, verbose="ERROR")

    press_times_shifted, trial_cutoff_times, press_trial_indices = (
        press_time_flattener(press_times, n_epochs, trial_time, sens_delay)
    )
    press_epochs = epochs_maker(
        raw, press_times_shifted, press_time_wind, sfreq, "press"
    )
    baseline_epochs = epochs_maker(
        raw, trial_cutoff_times, baseline_time_wind, sfreq, "baseline"
    )

    # Calculate theta power for press and baseline epochs.
    theta_power_by_press = mean_band_power(
        press_epochs, theta_freq_band, MEG_sens
    )
    theta_power_baseline = mean_band_power(
        baseline_epochs, theta_freq_band, MEG_sens
    )

    # Match each press to the baseline from the same trial.
    theta_baseline_by_trial = np.full(
        (len(trial_cutoff_times), len(MEG_sens)), np.nan
    )
    theta_baseline_by_trial[baseline_epochs.selection] = theta_power_baseline
    press_trial_ids = press_trial_indices[press_epochs.selection]
    theta_baseline_for_press = theta_baseline_by_trial[press_trial_ids]

    # Keep presses with an available trial baseline.
    has_baseline = np.isfinite(theta_baseline_for_press).any(axis=1)
    theta_power_by_press = theta_power_by_press[has_baseline]
    theta_baseline_for_press = theta_baseline_for_press[has_baseline]
    n_presses_used = int(has_baseline.sum())
    if n_presses_used == 0:
        raise RuntimeError("No presses have both a valid press epoch and baseline")

    # Convert each press-to-baseline power ratio to dB.
    valid_power = (
        np.isfinite(theta_power_by_press)
        & np.isfinite(theta_baseline_for_press)
        & (theta_power_by_press > 0)
        & (theta_baseline_for_press > 0)
    )
    theta_db_by_press = np.full(theta_power_by_press.shape, np.nan)
    power_ratio = np.ones_like(theta_power_by_press)
    np.divide(
        theta_power_by_press,
        theta_baseline_for_press,
        out=power_ratio,
        where=valid_power,
    )
    theta_db_by_press[valid_power] = 10.0 * np.log10(power_ratio[valid_power])

    # Average valid presses within this participant for each sensor.
    sensor_counts = np.isfinite(theta_db_by_press).sum(axis=0)
    theta_db_mean = np.divide(
        np.nansum(theta_db_by_press, axis=0),
        sensor_counts,
        out=np.full(len(MEG_sens), np.nan),
        where=sensor_counts > 0,
    )
    print("Participant", id, "- retained button presses:", n_presses_used)
    return theta_db_mean, MEG_sens, sensor_info, n_presses_used


def save_qq_plots(sensor_power_by_participant, sensor_names, output_file):
    """Save normal Q-Q plots for each sensor in a multi-page PDF."""
    sensors_per_page = 16
    rows, columns = 4, 4

    with PdfPages(output_file) as pdf:
        for start in range(0, len(sensor_names), sensors_per_page):
            stop = min(start + sensors_per_page, len(sensor_names))
            fig, axes = plt.subplots(
                rows, columns, figsize=(11, 8.5), squeeze=False
            )
            axes_flat = axes.ravel()

            for local_index, sensor_index in enumerate(range(start, stop)):
                ax = axes_flat[local_index]
                values = sensor_power_by_participant[sensor_index]
                values = values[np.isfinite(values)]
                if len(values) >= 2:
                    # Pair observed quantiles with theoretical standard-normal quantiles.
                    probabilities = (
                        np.arange(1, len(values) + 1, dtype=float) - 0.5
                    ) / len(values)
                    normal_quantiles = norm.ppf(probabilities)
                    observed_quantiles = np.sort(values)
                    slope, intercept = np.polyfit(
                        normal_quantiles, observed_quantiles, 1
                    )
                    ax.scatter(
                        normal_quantiles, observed_quantiles,
                        s=10, color="tab:blue",
                    )
                    ax.plot(
                        normal_quantiles,
                        slope * normal_quantiles + intercept,
                        color="tab:red",
                        linewidth=1,
                    )
                    ax.set_title(
                        str(sensor_names[sensor_index]) + " (n=" + str(len(values)) + ")",
                        fontsize=8,
                    )
                    ax.set_xlabel("Theoretical normal quantiles", fontsize=6)
                    ax.set_ylabel("Observed dB quantiles", fontsize=6)
                    ax.tick_params(labelsize=6)
                else:
                    ax.set_title(str(sensor_names[sensor_index]), fontsize=8)
                    ax.text(
                        0.5, 0.5, "Not enough values",
                        ha="center", va="center", transform=ax.transAxes,
                    )
                    ax.set_axis_off()

            for ax in axes_flat[stop - start:]:
                ax.set_visible(False)

            fig.suptitle(
                "Normal Q-Q plots of participant mean dB power "
                + " (sensors " + str(start + 1) + "–" + str(stop) + ")"
            )
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig)
            plt.close(fig)

    print("Saved sensor Q-Q plots to:", output_file)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate participant-level theta power changes, run one-sample "
            "tests per sensor, and save normal Q-Q plots."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default= os.environ.get("MEG_DATA_ROOT", r"//rdp.arc.ucl.ac.uk/ritd-ag-project-rd01pz-dbush99/Katja Button Press Data"),
        help="MEG data directory containing beh_data and participant folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_folder,
        help="Directory for participant data, test results, Q-Q plots and maps.",
    )
    args = parser.parse_args()

    if args.data_root is None:
        parser.error("Set MEG_DATA_ROOT or pass --data-root")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    subjects_to_use = [
        id for id in sub_ID if id not in excluded_subjects
    ]
    if not subjects_to_use:
        raise ValueError("No participants remain after applying excluded_subjects.")

    subject_power = []
    subjects_used = []
    retained_press_counts = []
    group_sensors = None
    group_info = None

    for id in subjects_to_use:
        theta_db_mean, MEG_sens, sensor_info, n_presses = (
            participant_theta_power(id, args.data_root)
        )

        # Keep channel order fixed so values cannot be assigned to mismatched sensors.
        if group_sensors is None:
            group_sensors = MEG_sens
            group_info = sensor_info
        elif MEG_sens != group_sensors:
            raise ValueError("MEG sensor names or order differ for participant " + id)

        subject_power.append(theta_db_mean)
        subjects_used.append(id)
        retained_press_counts.append(n_presses)

    # Rows are participants; columns are MEG sensors.
    participant_power_table = pd.DataFrame(
        subject_power, index=subjects_used, columns=group_sensors
    )
    participant_power_table.index.name = "participant"
    participant_power_table.to_csv(
        args.output_dir / participant_power_table_file.name
    )

    # Use the shared sensor-wise t-test and FDR implementation.
    sensor_power_by_participant = participant_power_table.to_numpy(
        dtype=float
    ).T
    results = sensor_wise_one_sample_ttest(
        participant_power_table,
        value_name="power_change_dB",
        popmean=0.0,
    )
    results.to_csv(
        args.output_dir / power_ttest_results_file.name, index=False
    )

    qq_file = args.output_dir / power_ttest_qq_file.name
    save_qq_plots(sensor_power_by_participant, group_sensors, qq_file)
    plot_sensor_test_topomaps(
        results,
        group_info,
        args.output_dir / power_ttest_topomap_file.name,
        mean_column="mean_power_change_dB",
        colorbar_label="Mean power change (dB)",
        title_prefix="Mean baseline-corrected theta power",
        p_value_column="p_value",
        threshold=0.01,
    )

    participant_summary = pd.DataFrame({
        "participant": subjects_used,
        "retained_press_count": retained_press_counts,
    })

    print("Participants tested:", len(participant_power_table))
    print("Sensors tested:", len(results))
    print("Sensors with raw p < 0.01:", int((results["p_value"] < 0.01).sum()))
    print("Saved participant power table to:",
          args.output_dir / participant_power_table_file.name)
    print("Saved test results to:",
          args.output_dir / power_ttest_results_file.name)
    print("Saved retained-press counts:")
    print(participant_summary.to_string(index=False))


if __name__ == "__main__":
    main()
