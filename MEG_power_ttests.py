#!/usr/bin/env python3
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

import h5py
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm, ttest_1samp


# Use data paths and outputs in the script directory by default.
script_folder = Path(__file__).resolve().parent
participant_power_table_file = script_folder / "participant_power_change_dB.csv"
power_ttest_results_file = script_folder / "power_ttest_results_4pi.csv"
power_ttest_qq_file = script_folder / "power_ttest_qqplots_4pi.pdf"
power_ttest_topomap_file = script_folder / "power_ttest_topomap_4pi.png"

# Match the participant list and power settings in MEG_powerplot_all.ipynb.
sub_ID = [
    "05", "06", "08", "09", "10", "11", "13", "14", "15",
    "16", "17", "18", "20", "21", "24", "25", "28",
]
excluded_subjects = []
theta_freq_band = np.array([2.0, 5.0])       # theta band edges in Hz
press_time_wind = np.array([-0.5, 0.5])      # epoch window around each button press
baseline_time_wind = np.array([-1.0, 0.0])   # baseline window before each trial
mean_time_offset = -0.0468                   # MEG trigger offset in seconds
sens_delay = 2.8 + mean_time_offset          # cue time relative to epoch start
n_press_slots = 5


def benjamini_hochberg(p_values):
    """Adjust sensor-wise p-values while preserving missing values as NaN."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted_p_values = np.full(p_values.shape, np.nan)
    valid_indices = np.flatnonzero(np.isfinite(p_values))

    if not len(valid_indices):
        return adjusted_p_values

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


def filename_loader(id, data_root):
    # Load cue-relative button press times and the participant's MEG epochs.
    button_press_folder = data_root / "beh_data" / ("s" + id)
    behaviour_file = button_press_folder / "behDataMEG.mat"
    target_file = button_press_folder / "targetMEG.mat"

    with h5py.File(behaviour_file, "r") as behaviour:
        timings = np.asarray(behaviour["A/timing"][:]).T
    with h5py.File(target_file, "r") as target:
        cue_duration = np.asarray(target["T/cueDur"][:]).T

    # Add each cue duration and convert the behavioral times from ms to seconds.
    press_times = (
        timings + np.tile(cue_duration, n_press_slots)
    ) / 1000.0

    subject_folder = data_root / ("s" + id)
    file_name = "Subject" + str(int(id)) + "_spm_mne-epo.fif"
    epochs_file = subject_folder / file_name
    if not epochs_file.is_file():
        epochs_file = data_root / file_name
    if not epochs_file.is_file():
        raise FileNotFoundError("Could not find " + file_name + " under " + str(data_root))

    epochs = mne.read_epochs(epochs_file, preload=True, verbose="ERROR")

    # Subject 16 has 222 saved epochs although its behavioral file has 240 rows.
    if id == "16" and len(epochs) == 222 and press_times.shape[0] == 240:
        print("Subject 16: using the first 222 behavioral trial rows.")
        press_times = press_times[:len(epochs), :]
    elif len(epochs) != press_times.shape[0]:
        raise ValueError(
            "Subject " + id + " has " + str(len(epochs))
            + " MEG epochs but " + str(press_times.shape[0])
            + " behavioral trials"
        )

    MEG_sens = [
        channel for channel in epochs.ch_names if channel.startswith("M")
    ]
    if not MEG_sens:
        raise ValueError("No MEG sensors found for subject " + id)

    sensor_picks = mne.pick_channels(
        epochs.ch_names, include=MEG_sens, ordered=True
    )
    sensor_info = mne.pick_info(epochs.info, sensor_picks, copy=True)
    return press_times, epochs, MEG_sens, sensor_info


def press_time_flattener(press_times, trial_time):
    # Shift press times onto the concatenated timeline and retain each trial ID.
    n_trials, n_slots = press_times.shape
    trial_offsets = np.arange(n_trials, dtype=float) * trial_time
    press_times_shifted = press_times + sens_delay + trial_offsets[:, None]
    trial_cutoff_times = sens_delay + trial_offsets

    valid = np.isfinite(press_times_shifted)
    press_trial_indices = np.broadcast_to(
        np.arange(n_trials)[:, None], (n_trials, n_slots)
    )[valid]
    return press_times_shifted[valid], trial_cutoff_times, press_trial_indices


def epochs_maker(raw, events, window, sfreq, name):
    # Convert event times to samples and extract the requested time window.
    samples = np.rint(np.asarray(events, dtype=float) * sfreq).astype(int)
    event_array = np.column_stack([
        samples,
        np.zeros(len(samples), dtype=int),
        np.ones(len(samples), dtype=int),
    ])
    return mne.Epochs(
        raw,
        event_array,
        event_id={"event": 1},
        tmin=float(window[0]),
        tmax=float(window[1]) - 1.0 / sfreq,
        baseline=None,
        preload=True,
        verbose="ERROR",
    )


def mean_band_power(epochs, band, MEG_sens):
    # Get data in epochs x sensors x time order.
    x = epochs.get_data(picks=MEG_sens)
    sfreq = float(epochs.info["sfreq"])
    n_times = x.shape[-1]
    if not np.isclose(n_times / sfreq, 1.0):
        raise ValueError("Expected a 1-second epoch")

    # Calculate a Hann-windowed, one-sided power spectral density.
    window = np.hanning(n_times)
    x = x - x.mean(axis=-1, keepdims=True)
    fft = np.fft.rfft(x * window, axis=-1)
    psd = (np.abs(fft) ** 2) / (sfreq * np.sum(window ** 2))
    freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)

    # Integrate the frequency bins within the requested band.
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    band_psd = psd[..., in_band]
    band_freqs = freqs[in_band]
    return np.sum(
        0.5 * (band_psd[..., 1:] + band_psd[..., :-1]) * np.diff(band_freqs),
        axis=-1,
    )


def participant_theta_power(id, data_root):
    press_times, epochs, MEG_sens, sensor_info = filename_loader(id, data_root)
    sfreq = float(epochs.info["sfreq"])

    # Reshape the MEG epochs into a continuous-style RawArray.
    MEG_data = epochs.get_data(picks=MEG_sens)
    n_epochs, n_channels, n_times = MEG_data.shape
    trial_time = n_times / sfreq
    MEG_data = MEG_data.transpose(1, 0, 2).reshape(
        n_channels, n_epochs * n_times
    )
    raw = mne.io.RawArray(MEG_data, sensor_info, verbose="ERROR")

    press_times_shifted, trial_cutoff_times, press_trial_indices = (
        press_time_flattener(press_times, trial_time)
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
    print(
        "Participant", id,
        "- retained button presses:", n_presses_used,
    )
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


def plot_sensor_power_results(results, sensor_info, output_base):
    # Align the results to the sensor order in the MNE metadata.
    results = results.set_index("sensor").loc[sensor_info["ch_names"]]
    mean_power = results["mean_power_change_dB"].to_numpy(dtype=float)
    p_values = results["p_value"].to_numpy(dtype=float)
    significant_sensor = np.isfinite(p_values) & (p_values < 0.01)
    sensor_types = [
        mne.channel_type(sensor_info, index)
        for index in range(len(sensor_info["ch_names"]))
    ]

    # Plot magnetometers and gradiometers separately, as their units differ.
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
            mean_power[sensor_picks],
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

        ax.set_title(
            "Mean baseline-corrected theta power; circles mark raw p < 0.01 ("
            + sensor_type + ")"
        )
        fig.colorbar(image, ax=ax, label="Mean power change (dB)")
        fig.tight_layout()
        plot_file = output_base.with_name(
            output_base.stem + "_" + sensor_type + output_base.suffix
        )
        fig.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("Saved scalp map to:", plot_file)


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
        default=Path(os.environ["MEG_DATA_ROOT"]) if os.environ.get("MEG_DATA_ROOT") else None,
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

    # Arrange the independent participant means as sensors x participants.
    sensor_power_by_participant = participant_power_table.to_numpy(
        dtype=float
    ).T
    n_participants_by_sensor = np.isfinite(
        sensor_power_by_participant
    ).sum(axis=1)
    if np.any(n_participants_by_sensor < 2):
        insufficient_sensors = np.asarray(group_sensors)[
            np.flatnonzero(n_participants_by_sensor < 2)
        ]
        raise ValueError(
            "At least two finite participant values are required for every "
            "sensor. Insufficient data for: "
            + ", ".join(map(str, insufficient_sensors[:10]))
        )

    # Test whether mean participant power change differs from 0 dB for each sensor.
    sensor_power_ttest = ttest_1samp(
        sensor_power_by_participant,
        popmean=0.0,
        axis=1,
        nan_policy="omit",
    )
    results = pd.DataFrame({
        "sensor": group_sensors,
        "n_participants": n_participants_by_sensor,
        "mean_power_change_dB": np.nanmean(
            sensor_power_by_participant, axis=1
        ),
        "std_power_change_dB": np.nanstd(
            sensor_power_by_participant, axis=1, ddof=1
        ),
        "t_statistic": sensor_power_ttest.statistic,
        "p_value": sensor_power_ttest.pvalue,
    })
    results["p_value_fdr"] = benjamini_hochberg(
        results["p_value"].to_numpy()
    )
    results.to_csv(
        args.output_dir / power_ttest_results_file.name, index=False
    )

    qq_file = args.output_dir / power_ttest_qq_file.name
    save_qq_plots(sensor_power_by_participant, group_sensors, qq_file)
    plot_sensor_power_results(
        results, group_info, args.output_dir / power_ttest_topomap_file.name
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
