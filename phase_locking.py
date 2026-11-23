"""Measure theta phase locking at button presses for each MEG sensor and trial.

For each included participant, this script filters MEG epochs to the theta band,
extracts phase at the five button-press times, and applies scipy.stats.directional_stats
to the corresponding unit vectors. The mean resultant length ranges from 0 to 1;
larger values indicate that press phases are more concentrated within that trial.
These are descriptive phase-locking measures, not significance tests.
"""

import argparse
import os

import mne
import numpy as np
import pandas as pd
from scipy.stats import directional_stats


# Set MEG_DATA_ROOT or pass --data-root to select a local MEG export.
data_root = os.environ.get("MEG_DATA_ROOT")

# Keep the button-press timing file alongside this script by default.
script_folder = os.path.dirname(os.path.abspath(__file__))
button_press_times_file = os.path.join(
    script_folder, "UPPT002_button_press_times.npz"
)
output_file = os.path.join(script_folder, "phase_locking_results.csv")

# Define the theta frequency band and analysis settings.
theta_freq_band = [2.0, 5.0]                 # theta band edges in Hz
excluded_subjects = ["11"]                   # participants excluded from analysis
channel_batch_size = 8                       # channels loaded for each MNE processing batch


# Match IDs from the NPZ file to the two-digit labels used by the MEG notebooks.
def normalise_subject_id(id):
    return str(int(str(id).strip().lower().lstrip("s"))).zfill(2)


# Read and check the button-press timing arrays and their metadata.
def load_button_press_times(file_name):
    with np.load(file_name, allow_pickle=False) as press_file:
        UPPT002_times = np.asarray(press_file["uppt002_times"], dtype=float)
        subject_ID = np.asarray(
            [normalise_subject_id(id) for id in press_file["subject_ID"]]
        )
        n_trials_by_subject = np.asarray(
            press_file["n_trials_by_subject"], dtype=int
        )
        press_slot = np.asarray(press_file["press_slot"], dtype=int)
        cue_onset_after_epoch_start_s = float(
            np.asarray(press_file["cue_onset_after_epoch_start_s"]).item()
        )
        time_reference = str(press_file["time_reference"].item())

    # Check the time reference and array layout before matching participants.
    if "relative to cue onset" not in time_reference.lower():
        raise ValueError(
            "Expected button-press times in seconds relative to cue onset"
        )
    if UPPT002_times.ndim != 3 or UPPT002_times.shape[0] != len(subject_ID):
        raise ValueError(
            "Unexpected participant x trial x press-slot array shape"
        )
    if UPPT002_times.shape[2] != len(press_slot):
        raise ValueError("Press-slot labels do not match the timing array")
    if len(n_trials_by_subject) != len(subject_ID):
        raise ValueError("Trial counts do not match the participant list")
    if not np.array_equal(press_slot, np.arange(1, len(press_slot) + 1)):
        raise ValueError("Expected press slots numbered in sequence from 1")

    return (
        UPPT002_times,
        subject_ID,
        n_trials_by_subject,
        press_slot,
        cue_onset_after_epoch_start_s,
    )


# Locate the participant's saved epochs using the repository's subject-folder naming.
def filename_loader(id, data_root):
    subject_number = str(int(id))
    subject_folder = "s" + id
    file_name = "Subject" + subject_number + "_spm_mne-epo.fif"
    epochs_file = os.path.join(data_root, subject_folder, file_name)

    # Also allow a local export with the FIF file directly in data_root.
    if not os.path.isfile(epochs_file):
        epochs_file = os.path.join(data_root, file_name)
    if not os.path.isfile(epochs_file):
        raise FileNotFoundError(
            "Could not find " + file_name + " under " + data_root
        )

    # The participant function reads its MEG data once.
    epochs = mne.read_epochs(epochs_file, preload=False, verbose="ERROR")
    MEG_sens = [
        channel for channel in epochs.ch_names if channel.startswith("M")
    ]
    if not MEG_sens:
        raise ValueError("No MEG sensors found in " + epochs_file)

    return epochs, MEG_sens


# Calculate one phase-locking result for each selected trial and MEG sensor.
def participant_phase_locking(
    id,
    UPPT002_times,
    n_trials,
    cue_onset_after_epoch_start_s,
    press_slot,
    data_root,
    trial_number=None,
):
    # Load this participant's epochs and confirm the timing rows match.
    epochs, MEG_sens = filename_loader(id, data_root)
    if len(epochs) != n_trials:
        raise ValueError(
            "Participant " + id + " has " + str(len(epochs))
            + " MEG epochs but " + str(n_trials) + " timing rows"
        )

    # Read the required MEG sensor data once.
    MEG_data = epochs.get_data(picks=MEG_sens)

    # Convert cue-relative press times to MNE epoch sample indices.
    press_times = UPPT002_times[:n_trials]
    cue_onset_epoch_s = epochs.times[0] + cue_onset_after_epoch_start_s
    event_times = press_times + cue_onset_epoch_s
    sample_indices = np.zeros(press_times.shape, dtype=int)
    valid_press = np.isfinite(event_times)
    sample_indices[valid_press] = np.rint(
        (event_times[valid_press] - epochs.times[0]) * epochs.info["sfreq"]
    ).astype(int)
    valid_press &= (
        (sample_indices >= 0) & (sample_indices < len(epochs.times))
    )

    # Use the same complete-press-trial rule as phase_real.ipynb.
    complete_trial = valid_press.sum(axis=1) == len(press_slot)
    if trial_number is None:
        trials_to_process = np.arange(n_trials)
    else:
        if trial_number < 1:
            raise ValueError("Trial numbers start at 1")
        if trial_number > n_trials:
            return None, MEG_sens, 0
        trials_to_process = np.asarray([trial_number - 1], dtype=int)

    # Store one phase vector and resultant length per selected trial and sensor.
    trial_resultant_x = np.full(
        (len(trials_to_process), len(MEG_sens)), np.nan
    )
    trial_resultant_y = np.full_like(trial_resultant_x, np.nan)
    trial_direction_x = np.full_like(trial_resultant_x, np.nan)
    trial_direction_y = np.full_like(trial_resultant_x, np.nan)
    trial_resultant_length = np.full_like(trial_resultant_x, np.nan)
    trial_preferred_phase = np.full_like(trial_resultant_x, np.nan)
    trial_phase_sample_count = np.zeros(
        (len(trials_to_process), len(MEG_sens)), dtype=int
    )
    trial_status = np.full(
        (len(trials_to_process), len(MEG_sens)),
        "incomplete_press_times",
        dtype=object,
    )

    # Mark complete trials before filtering; sensors with valid samples are updated below.
    for output_index, trial in enumerate(trials_to_process):
        if complete_trial[trial]:
            trial_status[output_index, :] = "no_finite_phase_samples"

    # Filter and transform a small channel batch at a time to limit working memory.
    for start in range(0, len(MEG_sens), channel_batch_size):
        channel_batch = MEG_sens[start:start + channel_batch_size]
        channel_indices = mne.pick_channels(
            epochs.ch_names, include=channel_batch, ordered=True
        )

        # Slice this channel batch from the participant data already in memory.
        channel_data = MEG_data[:, start:start + len(channel_batch), :]
        batch_info = mne.pick_info(epochs.info, channel_indices, copy=True)

        # Wrap the loaded data with matching sensor metadata for MNE processing.
        epochs_batch = mne.EpochsArray(
            channel_data,
            batch_info,
            events=epochs.events,
            tmin=epochs.tmin,
            baseline=None,
            verbose="ERROR",
        )

        # Filter to theta and form the complex analytic signal to extract phase.
        epochs_batch.filter(
            l_freq=theta_freq_band[0],
            h_freq=theta_freq_band[1],
            verbose="ERROR",
        ).apply_hilbert(envelope=False, verbose="ERROR")
        analytic = epochs_batch.get_data()

        # Find the circular phase vector across the five press slots for each trial.
        for output_index, trial in enumerate(trials_to_process):
            if not complete_trial[trial]:
                continue

            samples = sample_indices[trial]
            phase_by_sensor = np.angle(analytic[trial][:, samples])
            n_finite = np.isfinite(phase_by_sensor).sum(axis=1)
            trial_phase_sample_count[
                output_index, start:start + len(channel_batch)
            ] = n_finite

            valid_sensor = n_finite == len(press_slot)
            partial_sensor = (n_finite > 0) & ~valid_sensor
            trial_status[
                output_index,
                start + np.flatnonzero(partial_sensor),
            ] = "incomplete_phase_samples"

            if not valid_sensor.any():
                continue

            # Convert each press phase to a unit vector in the complex plane.
            valid_phase = phase_by_sensor[valid_sensor]
            phase_vectors = np.stack(
                (np.cos(valid_phase), np.sin(valid_phase)), axis=-1
            )

            # Each sensor contributes five unit vectors; summarize across press slots.
            directional_summary = directional_stats(
                phase_vectors, axis=1, normalize=False
            )
            mean_direction = np.asarray(
                directional_summary.mean_direction, dtype=float
            )
            mean_resultant_length = np.asarray(
                directional_summary.mean_resultant_length, dtype=float
            )

            # Reconstruct the mean resultant vector; its length is the locking strength.
            mean_resultant_vector = np.zeros_like(mean_direction)
            np.multiply(
                mean_direction,
                mean_resultant_length[:, None],
                out=mean_resultant_vector,
                where=np.isfinite(mean_direction),
            )
            sensor_indices = start + np.flatnonzero(valid_sensor)
            trial_resultant_x[output_index, sensor_indices] = (
                mean_resultant_vector[:, 0]
            )
            trial_resultant_y[output_index, sensor_indices] = (
                mean_resultant_vector[:, 1]
            )
            trial_direction_x[output_index, sensor_indices] = mean_direction[:, 0]
            trial_direction_y[output_index, sensor_indices] = mean_direction[:, 1]
            trial_resultant_length[output_index, sensor_indices] = (
                mean_resultant_length
            )
            trial_preferred_phase[output_index, sensor_indices] = np.arctan2(
                mean_direction[:, 1], mean_direction[:, 0]
            )
            trial_status[output_index, sensor_indices] = "ok"

        # Release this batch's temporary arrays before processing the next channel set.
        del analytic
        del epochs_batch
        del channel_data

    # Release the full participant array before returning the trial results.
    del MEG_data

    # Organize one result row per trial and sensor.
    n_sensors = len(MEG_sens)
    trial_labels = trials_to_process + 1
    results = pd.DataFrame(
        {
            "participant": id,
            "trial": np.repeat(trial_labels, n_sensors),
            "sensor": np.tile(MEG_sens, len(trials_to_process)),
            "n_valid_press_times": np.repeat(
                valid_press[trials_to_process].sum(axis=1), n_sensors
            ),
            "n_phase_samples": trial_phase_sample_count.reshape(-1),
            "resultant_vector_x": trial_resultant_x.reshape(-1),
            "resultant_vector_y": trial_resultant_y.reshape(-1),
            "mean_direction_x": trial_direction_x.reshape(-1),
            "mean_direction_y": trial_direction_y.reshape(-1),
            "mean_resultant_length": trial_resultant_length.reshape(-1),
            "preferred_phase_rad": trial_preferred_phase.reshape(-1),
            "status": trial_status.reshape(-1),
        }
    )

    return results, MEG_sens, int(complete_trial[trials_to_process].sum())


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate theta phase-locking vectors at button presses for "
            "each included participant and MEG sensor."
        )
    )
    parser.add_argument(
        "--data-root",
        default=data_root,
        help="MEG export directory (defaults to MEG_DATA_ROOT).",
    )
    parser.add_argument(
        "--button-press-times-file",
        default=button_press_times_file,
        help="NPZ file containing cue-relative button-press times.",
    )
    parser.add_argument(
        "--output-file",
        default=output_file,
        help="CSV file for trial-by-sensor phase-locking results.",
    )
    parser.add_argument(
        "--trial",
        type=int,
        default=None,
        help="Optional 1-based trial number; by default, process all trials.",
    )
    args = parser.parse_args()

    if not args.data_root:
        parser.error("Set MEG_DATA_ROOT or pass --data-root")
    if args.trial is not None and args.trial < 1:
        raise ValueError("Trial numbers start at 1")
    if os.path.abspath(args.output_file) == os.path.abspath(
        args.button_press_times_file
    ):
        raise ValueError("The output file cannot overwrite the timing NPZ file")

    (
        UPPT002_times,
        subject_ID,
        n_trials_by_subject,
        press_slot,
        cue_onset_after_epoch_start_s,
    ) = load_button_press_times(args.button_press_times_file)

    # Keep two-digit subject labels and omit subject 11, as in phase_real.ipynb.
    sub_ID = [
        id for id in subject_ID if id not in excluded_subjects
    ]
    subject_index = {id: index for index, id in enumerate(subject_ID)}
    if not sub_ID:
        raise ValueError("No participants remain after applying excluded_subjects.")

    if args.trial is not None and args.trial > int(n_trials_by_subject.max()):
        raise ValueError("The requested trial exceeds the timing file trial counts")

    rows_written = 0
    for id in sub_ID:
        data_index = subject_index[id]
        n_trials = int(n_trials_by_subject[data_index])
        results, MEG_sens, n_complete_trials = participant_phase_locking(
            id,
            UPPT002_times[data_index],
            n_trials,
            cue_onset_after_epoch_start_s,
            press_slot,
            args.data_root,
            trial_number=args.trial,
        )

        if results is None:
            print(
                "Participant", id, "- requested trial", args.trial,
                "is not present; skipping participant."
            )
            continue

        # Keep sensor order fixed so every participant uses consistent labels.
        if rows_written == 0:
            group_sensors = MEG_sens
        elif MEG_sens != group_sensors:
            raise ValueError("MEG sensor order differs for participant " + id)

        results.to_csv(
            args.output_file,
            index=False,
            mode="w" if rows_written == 0 else "a",
            header=(rows_written == 0),
        )
        rows_written += len(results)
        print(
            "Participant", id,
            "- complete trials:", n_complete_trials,
            "- sensor-trial rows:", len(results),
        )

    if rows_written == 0:
        raise ValueError("No participant trial results were written")

    print("Saved phase-locking results to", args.output_file)


if __name__ == "__main__":
    main()
