"""Shared helpers for press-aligned MEG band-power calculations."""

import mne
import numpy as np


def press_time_flattener(press_times, n_epochs, trial_time, sens_delay):
    """Shift cue-relative press times onto a concatenated epoch timeline."""
    n_trials, n_press_slots = press_times.shape
    if n_trials != n_epochs:
        raise ValueError(
            "Press-time rows do not match the MEG epoch count"
        )
    trial_offsets = np.arange(n_trials, dtype=float) * trial_time
    press_times_shifted = press_times + sens_delay + trial_offsets[:, None]
    trial_cutoff_times = sens_delay + trial_offsets

    valid = np.isfinite(press_times_shifted)
    press_trial_indices = np.broadcast_to(
        np.arange(n_trials)[:, None], (n_trials, n_press_slots)
    )[valid]
    print("Removed", int((~valid).sum()), "missing/non-finite press times")
    return press_times_shifted[valid], trial_cutoff_times, press_trial_indices


def epochs_maker(raw, events, window, sfreq, name="event"):
    """Extract one MNE epoch per event using the MEG power notebook settings."""
    samples = np.rint(np.asarray(events, dtype=float) * sfreq).astype(int)
    event_array = np.column_stack([
        samples,
        np.zeros(len(samples), dtype=int),
        np.ones(len(samples), dtype=int),
    ])
    epochs = mne.Epochs(
        raw,
        event_array,
        event_id={"event": 1},
        tmin=float(window[0]),
        tmax=float(window[1]) - 1.0 / sfreq,
        baseline=None,
        preload=True,
        verbose="ERROR",
    )
    print(name, "epochs:", len(epochs))
    return epochs


def mean_band_power(epochs, band, sensor_names):
    """Integrate Hann-windowed PSD within a frequency band for each epoch."""
    x = epochs.get_data(picks=sensor_names)
    sfreq = float(epochs.info["sfreq"])
    n_times = x.shape[-1]
    if not np.isclose(n_times / sfreq, 1.0):
        raise ValueError("Expected a 1-second epoch")

    window = np.hanning(n_times)
    x = x - x.mean(axis=-1, keepdims=True)
    fft = np.fft.rfft(x * window, axis=-1)
    psd = (np.abs(fft) ** 2) / (sfreq * np.sum(window ** 2))
    freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)

    in_band = (freqs >= band[0]) & (freqs <= band[1])
    band_psd = psd[..., in_band]
    band_freqs = freqs[in_band]
    return np.sum(
        0.5 * (band_psd[..., 1:] + band_psd[..., :-1])
        * np.diff(band_freqs),
        axis=-1,
    )
