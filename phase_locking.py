"""Save participant-level mean theta phase-locking values for MEG sensors.

For each included participant and trial, this script filters MEG epochs to the theta
band, extracts phase at the five button-press times, and uses scipy.stats.directional_stats
to calculate the mean direction vector and mean resultant length. It then averages the
trial-wise mean resultant lengths for each participant and sensor and saves a wide
participant-by-sensor CSV table. Larger values indicate more concentrated press phases.
The values are descriptive and are not significance tests.
"""

import argparse
import os

import mne
import numpy as np
import pandas as pd
from scipy.stats import directional_stats


# Set MEG_DATA_ROOT or pass --data-root to select a local MEG export.
data_root = os.environ.get("MEG_DATA_ROOT", r"//rdp.arc.ucl.ac.uk/ritd-ag-project-rd01pz-dbush99/Katja Button Press Data")


# Keep the button-press timing file and output table alongside this script by default.
script_folder = os.path.dirname(os.path.abspath(__file__))
button_press_times_file = os.path.join(
    script_folder, "UPPT002_button_press_times.npz"
)
output_file = os.path.join(
    script_folder, "participant_phase_locking_table.csv"
)

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


# Average finite trial values and preserve NaN when no valid values are available.
def mean_finite(values, axis=0):
    values = np.asarray(values)
    counts = np.isfinite(values).sum(axis=axis)
    return np.divide(
        np.nansum(values, axis=axis),
        counts,
        out=np.full(
            counts.shape, np.nan, dtype=np.result_type(values.dtype, float)
        ),
        where=counts > 0,
    )


# Calculate and average trial-wise phase-locking values for one participant.
def participant_phase_locking(
    id,
    UPPT002_times,
    n_trials,
    cue_onset_after_epoch_start_s,
    press_slot,
    data_root,
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

    # Store the trial-wise resultant length for averaging; invalid trials remain NaN.
    trial_resultant_length = np.full((n_trials, len(MEG_sens)), np.nan)

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

        # Find a circular phase vector across the five press slots for each trial.
        for trial in np.flatnonzero(complete_trial):
            samples = sample_indices[trial]
            phase_by_sensor = np.angle(analytic[trial][:, samples])
            valid_sensor = np.isfinite(phase_by_sensor).all(axis=1)
            if not valid_sensor.any():
                continue

            # Convert each press phase to a unit vector in the complex plane.
            valid_phase = phase_by_sensor[valid_sensor]
            phase_vectors = np.stack(
                (np.cos(valid_phase), np.sin(valid_phase)), axis=-1
            )

            # Summarize each sensor's press vectors and retain its locking strength.
            directional_summary = directional_stats(
                phase_vectors, axis=1, normalize=False
            )
            sensor_indices = start + np.flatnonzero(valid_sensor)
            trial_resultant_length[trial, sensor_indices] = (
                directional_summary.mean_resultant_length
            )

        # Release this batch's temporary arrays before processing the next channel set.
        del analytic
        del epochs_batch
        del channel_data

    # Average valid trial values separately for each sensor.
    participant_mean = mean_finite(trial_resultant_length, axis=0)
    n_trials_with_phase = int(
        np.isfinite(trial_resultant_length).any(axis=1).sum()
    )

    # Release the full participant array before returning the sensor means.
    del MEG_data

    return (
        participant_mean,
        MEG_sens,
        int(complete_trial.sum()),
        n_trials_with_phase,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Save mean theta phase-locking values across trials for each "
            "included participant and MEG sensor."
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
        help="Wide participant-by-sensor CSV output file.",
    )
    args = parser.parse_args()

    if not args.data_root:
        parser.error("Set MEG_DATA_ROOT or pass --data-root")
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
    sub_ID = [id for id in subject_ID if id not in excluded_subjects]
    subject_index = {id: index for index, id in enumerate(subject_ID)}
    if not sub_ID:
        raise ValueError("No participants remain after applying excluded_subjects.")

    participant_means = []
    group_sensors = None
    for id in sub_ID:
        data_index = subject_index[id]
        n_trials = int(n_trials_by_subject[data_index])
        participant_mean, MEG_sens, n_complete_trials, n_trials_with_phase = (
            participant_phase_locking(
                id,
                UPPT002_times[data_index],
                n_trials,
                cue_onset_after_epoch_start_s,
                press_slot,
                args.data_root,
            )
        )

        # Keep sensor order fixed so every participant contributes to the same columns.
        if group_sensors is None:
            group_sensors = MEG_sens
        elif MEG_sens != group_sensors:
            raise ValueError("MEG sensor order differs for participant " + id)

        participant_means.append(participant_mean)
        print(
            "Participant", id,
            "- complete trials:", n_complete_trials,
            "- trials with phase data:", n_trials_with_phase,
        )

    # Rows are participants and columns are MEG sensors, matching the slope table layout.
    participant_phase_locking_table = pd.DataFrame(
        participant_means, index=sub_ID, columns=group_sensors
    )
    participant_phase_locking_table.to_csv(args.output_file, index=False)
    print("Saved participant mean phase-locking table to", args.output_file)


if __name__ == "__main__":
    main()
