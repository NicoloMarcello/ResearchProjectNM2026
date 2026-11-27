"""Shared loaders for participant behavior files, epochs, and timing exports."""

from pathlib import Path

import h5py
import mne
import numpy as np


def normalise_subject_id(subject_id):
    """Return a participant ID in the repository's two-digit format."""
    return str(int(str(subject_id).strip().lower().lstrip("s"))).zfill(2)


def find_epochs_file(data_root, subject_id):
    """Locate a participant's saved MNE epochs file."""
    data_root = Path(data_root)
    subject_id = normalise_subject_id(subject_id)
    file_name = "Subject" + str(int(subject_id)) + "_spm_mne-epo.fif"
    candidates = [
        data_root / ("s" + subject_id) / file_name,
        data_root / file_name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find " + file_name + " under " + str(data_root)
    )


def find_behavior_files(data_root, subject_id):
    """Locate the behavioral timing and target files for a participant."""
    data_root = Path(data_root)
    subject_id = normalise_subject_id(subject_id)
    folders = [
        data_root / "beh_data" / ("s" + subject_id),
        data_root / ("s" + subject_id),
    ]
    for folder in folders:
        behavior_file = folder / "behDataMEG.mat"
        target_file = folder / "targetMEG.mat"
        if behavior_file.is_file() and target_file.is_file():
            return behavior_file, target_file
    raise FileNotFoundError(
        "Could not find behDataMEG.mat and targetMEG.mat for participant "
        + subject_id + " under " + str(data_root)
    )


def load_epochs(data_root, subject_id, preload=False, verbose="ERROR"):
    """Load one participant's MNE epochs from the repository folder layout."""
    epochs_file = find_epochs_file(data_root, subject_id)
    return mne.read_epochs(
        epochs_file, preload=preload, verbose=verbose
    )


def meg_sensor_names(epochs, prefix="M"):
    """Return channel names for sensors whose names use the given prefix."""
    sensor_names = [
        channel for channel in epochs.ch_names if channel.startswith(prefix)
    ]
    if not sensor_names:
        raise ValueError("No " + prefix + "-prefixed sensors were found")
    return sensor_names


def load_participant_data(
    data_root,
    subject_id,
    preload=True,
    verbose="ERROR",
    include_press_ids=True,
    trim_subject_16=True,
):
    """Load behavioral press times and the matching participant epochs.

    Behavioral times include cue duration and are returned in seconds, in
    trials-by-press-slots order. Subject 16's known 240-to-222 row mismatch is
    handled by retaining the first 222 behavioral trials when requested.
    """
    subject_id = normalise_subject_id(subject_id)
    behavior_file, target_file = find_behavior_files(data_root, subject_id)

    with h5py.File(behavior_file, "r") as behavior:
        timings_ms = np.asarray(behavior["A/timing"][:]).T
        presses = (
            np.asarray(behavior["A/press"][:]).T
            if include_press_ids else None
        )
    with h5py.File(target_file, "r") as target:
        cue_duration_ms = np.asarray(target["T/cueDur"][:]).reshape(-1)

    if timings_ms.ndim != 2:
        raise ValueError(
            "Expected A/timing to have trial-by-press-slot layout after transpose"
        )
    if timings_ms.shape[0] != len(cue_duration_ms):
        raise ValueError(
            "A/timing and T/cueDur have different trial counts for participant "
            + subject_id
        )
    if presses is not None and presses.shape != timings_ms.shape:
        raise ValueError(
            "A/press and A/timing have different shapes for participant "
            + subject_id
        )

    press_times = (timings_ms + cue_duration_ms[:, None]) / 1000.0
    epochs = load_epochs(
        data_root, subject_id, preload=preload, verbose=verbose
    )

    if (
        trim_subject_16
        and subject_id == "16"
        and len(epochs) == 222
        and press_times.shape[0] == 240
    ):
        print("Subject 16: using the first 222 behavioral trial rows.")
        press_times = press_times[:len(epochs), :]
        timings_ms = timings_ms[:len(epochs), :]
        cue_duration_ms = cue_duration_ms[:len(epochs)]
        if presses is not None:
            presses = presses[:len(epochs), :]
    elif len(epochs) != press_times.shape[0]:
        raise ValueError(
            "Participant " + subject_id + " has " + str(len(epochs))
            + " MEG epochs but " + str(press_times.shape[0])
            + " behavioral trials"
        )

    sensors = meg_sensor_names(epochs)
    sensor_picks = mne.pick_channels(
        epochs.ch_names, include=sensors, ordered=True
    )
    sensor_info = mne.pick_info(epochs.info, sensor_picks, copy=True)

    return {
        "participant": subject_id,
        "behavior_file": behavior_file,
        "target_file": target_file,
        "epochs_file": find_epochs_file(data_root, subject_id),
        "timings_ms": timings_ms,
        "presses": presses,
        "cue_duration_ms": cue_duration_ms,
        "press_times": press_times,
        "epochs": epochs,
        "meg_sensors": sensors,
        "sensor_info": sensor_info,
        "sfreq": float(epochs.info["sfreq"]),
    }


def load_button_press_times(file_name):
    """Read and validate the UPPT002 timing NPZ used by phase analyses."""
    with np.load(file_name, allow_pickle=False) as press_file:
        times = np.asarray(press_file["uppt002_times"], dtype=float)
        subject_ids = np.asarray([
            normalise_subject_id(subject_id)
            for subject_id in press_file["subject_ID"]
        ])
        trial_counts = np.asarray(
            press_file["n_trials_by_subject"], dtype=int
        )
        press_slots = np.asarray(press_file["press_slot"], dtype=int)
        cue_onset = float(
            np.asarray(press_file["cue_onset_after_epoch_start_s"]).item()
        )
        time_reference = str(press_file["time_reference"].item())

    if "relative to cue onset" not in time_reference.lower():
        raise ValueError(
            "Expected button-press times in seconds relative to cue onset"
        )
    if times.ndim != 3 or times.shape[0] != len(subject_ids):
        raise ValueError(
            "Unexpected participant x trial x press-slot array shape"
        )
    if times.shape[2] != len(press_slots):
        raise ValueError("Press-slot labels do not match the timing array")
    if len(trial_counts) != len(subject_ids):
        raise ValueError("Trial counts do not match the participant list")
    if not np.array_equal(
        press_slots, np.arange(1, len(press_slots) + 1)
    ):
        raise ValueError("Expected press slots numbered in sequence from 1")

    return {
        "times": times,
        "subject_ids": subject_ids,
        "trial_counts": trial_counts,
        "press_slots": press_slots,
        "cue_onset_after_epoch_start_s": cue_onset,
        "time_reference": time_reference,
    }
