"""Shared UPPT002 event detection and behavior-to-signal matching helpers."""

import numpy as np


def robust_sigma(values):
    """Estimate noise scale with a median absolute deviation."""
    values = np.asarray(values, dtype=float)
    center = np.median(values)
    return 1.4826 * np.median(np.abs(values - center))


def detect_uppt002_changes(
    signal,
    epoch_times,
    cue_onset,
    noise_sigma_multiplier=8.0,
    relative_step_fraction=0.02,
    persistence_s=0.020,
    refractory_s=0.020,
):
    """Detect persistent UPPT002 signal steps before and after cue onset."""
    signal = np.asarray(signal, dtype=float)
    epoch_times = np.asarray(epoch_times, dtype=float)
    baseline_window = (
        (epoch_times >= epoch_times[0] + 0.05)
        & (epoch_times < cue_onset - 0.05)
    )
    baseline_values = signal[baseline_window]
    if len(baseline_values) < 3:
        raise ValueError("UPPT002 baseline window is too short to estimate noise")

    baseline = np.median(baseline_values)
    baseline_noise = robust_sigma(baseline_values)
    step_noise = robust_sigma(np.diff(baseline_values))
    excursion = np.percentile(np.abs(signal - baseline), 99.9)
    numerical_floor = (
        np.finfo(float).eps * max(1.0, np.max(np.abs(signal))) * 10
    )
    step_threshold = max(
        noise_sigma_multiplier * step_noise,
        relative_step_fraction * excursion,
        numerical_floor,
    )
    active_threshold = max(
        3 * baseline_noise, 0.25 * step_threshold, numerical_floor
    )

    sfreq = 1 / np.median(np.diff(epoch_times))
    persistence = max(3, int(np.ceil(persistence_s * sfreq)))
    refractory = max(1, int(np.ceil(refractory_s * sfreq)))
    edges = np.where(
        np.abs(np.diff(signal, prepend=signal[0])) >= step_threshold
    )[0]

    event_samples = []
    last_event = -refractory
    for sample in edges:
        if sample < persistence or sample + persistence > len(signal):
            continue
        if sample - last_event < refractory:
            continue

        before = np.median(signal[sample - persistence:sample])
        after = np.median(signal[sample:sample + persistence])
        if abs(after - before) < step_threshold:
            continue
        if abs(after - baseline) <= active_threshold:
            continue

        event_samples.append(sample)
        last_event = sample

    event_samples = np.asarray(event_samples, dtype=int)
    event_times = epoch_times[event_samples] - cue_onset
    return event_times, {
        "baseline_noise": baseline_noise,
        "step_threshold": step_threshold,
    }


def match_events_in_order(behavior_times, uppt_times, tolerance):
    """Match behavioral press times to signal events in chronological order."""
    behavior_times = np.asarray(behavior_times, dtype=float)
    uppt_times = np.asarray(uppt_times, dtype=float)
    behavior_order = np.flatnonzero(np.isfinite(behavior_times))
    behavior_order = behavior_order[
        np.argsort(behavior_times[behavior_order], kind="stable")
    ]
    uppt_order = np.argsort(uppt_times, kind="stable")

    behavior = behavior_times[behavior_order]
    uppt = uppt_times[uppt_order]
    n_behavior, n_uppt = len(behavior), len(uppt)
    count = np.zeros((n_behavior + 1, n_uppt + 1), dtype=int)
    cost = np.zeros((n_behavior + 1, n_uppt + 1))
    step = np.zeros((n_behavior + 1, n_uppt + 1), dtype=int)

    for i in range(1, n_behavior + 1):
        for j in range(1, n_uppt + 1):
            options = [
                (count[i - 1, j], cost[i - 1, j], 1),
                (count[i, j - 1], cost[i, j - 1], 2),
            ]
            difference = abs(behavior[i - 1] - uppt[j - 1])
            if difference <= tolerance:
                options.append((
                    count[i - 1, j - 1] + 1,
                    cost[i - 1, j - 1] + difference,
                    3,
                ))
            best = max(options, key=lambda value: (
                value[0], -value[1], value[2] == 3
            ))
            count[i, j], cost[i, j], step[i, j] = best

    pairs = []
    i, j = n_behavior, n_uppt
    while i > 0 and j > 0:
        if step[i, j] == 3:
            pairs.append((behavior_order[i - 1], uppt_order[j - 1]))
            i -= 1
            j -= 1
        elif step[i, j] == 1:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]
