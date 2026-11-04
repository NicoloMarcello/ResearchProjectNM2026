import numpy as np
import mne
from functools import lru_cache
from scipy.signal import hilbert


def circ_regress_db(x, t, k=1000, ppdir=0, phaseLim=2*np.pi):
    """
    Circular–linear regression by exhaustive search.
    Ported from Daniel Bush (2020) MATLAB version.

    Parameters
    ----------
    x : array (n,)
        Linear variable.
    t : array (n,)
        Phase angles in radians (0–2π).
    k : int
        Number of permutations for p-value.
    ppdir : int
        -1 = negative slope only
         0 = unconstrained
         1 = positive slope only
    phaseLim : float
        Maximum slope range.

    Returns
    -------
    dict with keys:
        slope_opt
        phase_opt
        corr
        p
    """

    # --- clean input and flatten ---
    x = np.asarray(x).flatten()
    t = np.asarray(t).flatten()

    mask = ~np.isnan(x) & ~np.isnan(t)
    x = x[mask]
    t = t[mask]
    n = len(x)

    # --- construct slope search range (as in MATLAB: m_range) ---
    if ppdir == -1:
        m_range = np.arange(-phaseLim, 0 + 1e-12, np.pi/500)
    elif ppdir == 0:
        m_range = np.arange(-phaseLim, phaseLim + 1e-12, np.pi/500)
    else:
        m_range = np.arange(0, phaseLim + 1e-12, np.pi/500)

    # --- evaluate cost function for all slopes (vectorized) ---
    alpha = t[:, None] - x[:, None] * m_range[None, :]
    cost = -np.abs(np.sum(np.exp(1j*alpha), axis=0) / n)

    # --- pick best slope (min cost; if ties choose smallest abs value) ---
    slope_candidates = m_range[cost == np.min(cost)]
    slope_opt = slope_candidates[np.argmin(np.abs(slope_candidates))]

    # --- phase offset ---
    phase_opt = np.arctan2(
        np.sum(np.sin(t - slope_opt*x)),
        np.sum(np.cos(t - slope_opt*x))
    )

    # --- Kempter circular correlation ---
    def circ_mean(a):
        mu = np.arctan2(np.nanmean(np.sin(a)), np.nanmean(np.cos(a)))
        return mu % (2*np.pi)

    t_bar = t - circ_mean(t)
    theta = np.mod(np.abs(slope_opt) * x, 2*np.pi)
    theta_bar = theta - circ_mean(theta)

    corr = np.sum(np.sin(t_bar) * np.sin(theta_bar)) / np.sqrt(
        np.sum(np.sin(t_bar)**2) * np.sum(np.sin(theta_bar)**2)
    )

    # --- permutation testing ---
    if k > 0:
        # efficiently shuffle t
        idx = np.vstack([np.random.permutation(n) for _ in range(k)])
        t_shuf = t[idx]

        # compute cost for each shuffle
        # alpha_shuf: (n, k, slopes)
        alpha_shuf = (
            t_shuf.T[:, :, None] -
            (x[:, None, None] * m_range[None, None, :])
        )

        # cost_shuf: (k, slopes)
        cost_shuf = -np.abs(np.sum(np.exp(1j*alpha_shuf), axis=0) / n)

        # pick optimal slope per permutation
        opt_idx = np.argmin(cost_shuf, axis=1)
        m_shuf = m_range[opt_idx]

        # compute permuted circular correlations
        corr_shuf = []
        for j in range(k):
            theta_j = np.mod(np.abs(m_shuf[j]) * x, 2*np.pi)

            t_bar_j = t_shuf[j] - circ_mean(t)
            theta_bar_j = theta_j - circ_mean(theta_j)

            corr_j = np.sum(np.sin(t_bar_j) * np.sin(theta_bar_j)) / np.sqrt(
                np.sum(np.sin(t_bar_j)**2) * np.sum(np.sin(theta_bar_j)**2)
            )
            corr_shuf.append(corr_j)

        corr_shuf = np.array(corr_shuf)

        # compute p-value
        if np.isnan(corr):
            p = np.nan
        elif corr > 0:
            p = np.sum(corr_shuf > corr) / k
        else:
            p = np.sum(corr_shuf < corr) / k

    else:
        p = np.nan

    return {
        "slope_opt": slope_opt,
        "phase_opt": phase_opt,
        "corr": corr,
        "p": p
    }






@lru_cache(maxsize=16)
def _cached_slope_grid_and_basis(x_values, phaseLim):
    # Cache the shared slope grid and linear-to-circular basis for repeated batches.
    x = np.asarray(x_values, dtype=float)
    m_range = np.arange(-phaseLim, phaseLim + 1e-12, np.pi / 500)
    slope_basis = np.exp(-1j * x[:, None] * m_range[None, :])
    return m_range, slope_basis


def circ_regress_db_batch(x, t, phaseLim=2 * np.pi):
    """
    Vectorized circular-linear regression for multiple phase series.

    Parameters
    ----------
    x : array (n,)
        Linear variable shared by all rows.
    t : array (n_series, n)
        Phase angles in radians. Each row is fitted independently.
    phaseLim : float
        Maximum slope range.

    Returns
    -------
    slopes, phase_offsets, correlations : arrays (n_series,)
        Fitted values corresponding to circ_regress_db. Permutation p-values
        are intentionally omitted for this batched calculation.
    """
    x = np.asarray(x, dtype=float).flatten()
    t = np.asarray(t, dtype=float)
    if t.ndim == 1:
        t = t[None, :]
    if t.ndim != 2 or t.shape[1] != len(x):
        raise ValueError("t must have shape (n_series, len(x))")
    if not len(x) or not np.isfinite(x).all() or not np.isfinite(t).all():
        raise ValueError("x and t must contain finite values")

    # Factor the slope search across series instead of repeating it per sensor.
    m_range, slope_basis = _cached_slope_grid_and_basis(
        tuple(x), float(phaseLim)
    )
    resultants = np.exp(1j * t) @ slope_basis
    cost = -np.abs(resultants / len(x))

    # Match circ_regress_db tie-breaking: among best fits choose the
    # smallest absolute slope, then the first candidate in the search grid.
    best_cost = cost.min(axis=1)
    tied = cost == best_cost[:, None]
    abs_slope = np.abs(m_range)
    best_abs_slope = np.min(
        np.where(tied, abs_slope[None, :], np.inf), axis=1
    )
    best_index = np.argmax(
        tied & (abs_slope[None, :] == best_abs_slope[:, None]),
        axis=1,
    )
    slopes = m_range[best_index]

    # Calculate each fitted phase offset and circular correlation.
    residual = t - slopes[:, None] * x[None, :]
    phase_offsets = np.arctan2(
        np.sin(residual).sum(axis=1),
        np.cos(residual).sum(axis=1),
    )
    t_mean = np.arctan2(
        np.sin(t).mean(axis=1), np.cos(t).mean(axis=1)
    ) % (2 * np.pi)
    theta = np.mod(np.abs(slopes[:, None]) * x[None, :], 2 * np.pi)
    theta_mean = np.arctan2(
        np.sin(theta).mean(axis=1), np.cos(theta).mean(axis=1)
    ) % (2 * np.pi)
    t_bar = t - t_mean[:, None]
    theta_bar = theta - theta_mean[:, None]

    numerator = (np.sin(t_bar) * np.sin(theta_bar)).sum(axis=1)
    denominator = np.sqrt(
        (np.sin(t_bar) ** 2).sum(axis=1)
        * (np.sin(theta_bar) ** 2).sum(axis=1)
    )
    correlations = np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan),
        where=denominator > 0,
    )
    return slopes, phase_offsets, correlations


def extract_phase_epochs(epochs,low_freq=2,high_freq=40): 

    # 1. Filter raw data or epochs in a specific frequency band (e.g., 8-12 Hz)
    epochs.load_data()
    epochs_filtered = epochs.copy().filter(
        l_freq=low_freq, h_freq=high_freq, fir_design="firwin"
    )

    # 2. Extract data array: shape is (n_epochs, n_channels, n_times)
    data = epochs_filtered.get_data()

    # 3. Compute Hilbert transform along the time axis (-1) to get analytic signal
    analytic_signal = hilbert(data, axis=-1)

    # 4. Extract instantaneous phase
    phase_data = np.angle(analytic_signal)

    # 5. Optionally store phase back into an EpochsArray for plotting/inspection
    phase_epochs = mne.EpochsArray(phase_data, epochs.info, tmin=epochs.tmin)
