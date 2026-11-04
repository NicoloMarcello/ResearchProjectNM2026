import numpy as np


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
