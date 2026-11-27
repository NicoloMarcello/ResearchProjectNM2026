"""Shared sensor-wise one-sample tests and multiple-test correction."""

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp


def benjamini_hochberg(p_values):
    """Adjust p-values while leaving missing values as NaN."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan)
    valid_indices = np.flatnonzero(np.isfinite(p_values))
    if not len(valid_indices):
        return adjusted

    order = np.argsort(p_values[valid_indices])
    sorted_p = p_values[valid_indices][order]
    ranks = np.arange(1, len(sorted_p) + 1)
    sorted_adjusted = sorted_p * len(sorted_p) / ranks
    sorted_adjusted = np.minimum.accumulate(sorted_adjusted[::-1])[::-1]
    adjusted[valid_indices[order]] = np.clip(sorted_adjusted, 0, 1)
    return adjusted


def sensor_wise_one_sample_ttest(
    participant_by_sensor,
    value_name,
    popmean=0.0,
):
    """Test participant values against popmean separately for every sensor.

    Parameters
    ----------
    participant_by_sensor : pandas.DataFrame
        Rows are participants and columns are sensors. Missing participant
        measurements may be NaN.
    value_name : str
        Suffix used for the mean and standard-deviation result columns.
    popmean : float, default 0
        Null-hypothesis mean for the one-sample, two-sided t-test.

    Returns
    -------
    pandas.DataFrame
        One result row per sensor, including sample count, mean, standard
        deviation, t-statistic, raw p-value, and Benjamini-Hochberg p-value.
    """
    if not isinstance(participant_by_sensor, pd.DataFrame):
        participant_by_sensor = pd.DataFrame(participant_by_sensor)
    if participant_by_sensor.empty:
        raise ValueError("Participant-by-sensor input table is empty")
    if participant_by_sensor.columns.duplicated().any():
        raise ValueError("Participant-by-sensor table has duplicate sensors")

    numeric = participant_by_sensor.apply(pd.to_numeric, errors="coerce")
    sensor_by_participant = numeric.to_numpy(dtype=float).T
    n_by_sensor = np.isfinite(sensor_by_participant).sum(axis=1)
    if np.any(n_by_sensor < 2):
        insufficient = numeric.columns[np.flatnonzero(n_by_sensor < 2)]
        raise ValueError(
            "At least two finite participant values are required for each "
            "sensor. Insufficient data for: "
            + ", ".join(map(str, insufficient[:10]))
        )

    test = ttest_1samp(
        sensor_by_participant,
        popmean=popmean,
        axis=1,
        nan_policy="omit",
        alternative="two-sided",
    )
    results = pd.DataFrame({
        "sensor": numeric.columns,
        "n_participants": n_by_sensor,
        "mean_" + value_name: np.nanmean(sensor_by_participant, axis=1),
        "std_" + value_name: np.nanstd(
            sensor_by_participant, axis=1, ddof=1
        ),
        "t_statistic": test.statistic,
        "p_value": test.pvalue,
    })
    results["p_value_fdr"] = benjamini_hochberg(
        results["p_value"].to_numpy()
    )
    return results
