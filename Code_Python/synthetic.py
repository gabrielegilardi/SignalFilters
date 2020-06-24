"""
Utility functions for ????.

Copyright (c) 2020 Gabriele Gilardi
"""

import numpy as np


def normalize_data(X, param=(), ddof=0):
    """
    If mu and sigma are not defined, returns a column-normalized version of
    X with zero mean and standard deviation equal to one. If mu and sigma are
    defined returns a column-normalized version of X using mu and sigma.

    X           Input dataset
    Xn          Column-normalized input dataset
    param       Tuple with mu and sigma
    mu          Mean
    sigma       Standard deviation
    ddof        Delta degrees of freedom (if ddof = 0 then divide by m, if
                ddof = 1 then divide by m-1, with m the number of data in X)
    """
    # Column-normalize using mu and sigma
    if (len(param) > 0):
        Xn = (X - param[0]) / param[1]
        return Xn

    # Column-normalize using mu=0 and sigma=1
    else:
        mu = X.mean(axis=0)
        sigma = X.std(axis=0, ddof=ddof)
        Xn = (X - mu) / sigma
        param = (mu, sigma)
        return Xn, param


def scale_data(X, param=()):
    """
    If X_min and X_max are not defined, returns a column-scaled version of
    X in the interval (-1,+1). If X_min and X_max are defined returns a
    column-scaled version of X using X_min and X_max.

    X           Input dataset
    Xs          Column-scaled input dataset
    param       Tuple with X_min and X_max
    X_min       Min. value along the columns (features) of the input dataset
    X_max       Max. value along the columns (features) of the input dataset
    """
    # Column-scale using X_min and X_max
    if (len(param) > 0):
        Xs = -1.0 + 2.0 * (X - param[0]) / (param[1] - param[0])
        return Xs

    # Column-scale using X_min=-1 and X_max=+1
    else:
        X_min = np.amin(X, axis=0)
        X_max = np.amax(X, axis=0)
        Xs = -1.0 + 2.0 * (X - X_min) / (X_max - X_min)
        param = (X_min, X_max)
        return Xs, param


def value2diff(X, percent=True):
    """
    from value to difference in abs or %
    dx is reduced by 1 row
    """
    # Difference in percent
    if (percent):
        dX = X[1:, :] / X[:-1, :] - 1.0
    
    # Difference in value
    else:
        dX = X[1:, :] - X[:-1, :]
    
    return dX


def diff2value(dX, percent=True):
    """
    from difference in abs or % to value
    X is increased by one row

    Value from percent: first row set to one. If X0 defines the starting
    values, then X0*X would be the ???

    Value from difference: first row set to zero. If X0 defines the starting
    values, then X0+X would be the ???
    """
    n_rows, n_cols = dX.shape
    X = np.zeros((n_rows+1, n_cols))

    # Value from percent
    # X[0, :] = 1
    # X[1, :] = X[0, :] * (1 + dX[1, :]) = (1 + dX[1, :])
    # X[2, :] = X[1, :] * (1 + dX[2, :])
    #         = X[0, :] * (1 + dX[1, :]) * (1 + dX[2, :])
    #         = (1 + dX[1, :]) * (1 + dX[2, :])
    # ....
    if (percent):
        X[0, :] = 1.0
        X[1:, :] = np.cumprod((1.0 + dX), axis=0)
    
    # Value from difference
    # X[0, :] = 0
    # X[1, :] = X[0, :] + dX[1, :] = dX[1, :]
    # X[2, :] = X[0, :] + dX[1, :] + dX[2, :] = dX[1, :] + dX[2, :]
    # ....
    else:
        # First row already set to zero
        X[1:, :] = np.cumsum(dX, axis=0)
    
    return X


def synthetic_wave(P, A=None, phi=None, num=1000):
    """
    Generates a multi-sinewave.
    P = [P_1, P_2, ... P_n]                 Periods
    A = [A_1, A_2, ... A_n]                 Amplitudes
    phi = [phi_1, phi_2, ... phi_n]         Phases (rad)

    Default amplitudes are ones
    Default phases are zeros
    Time is from 0 to largest period (default 1000 steps)
    """
    n_waves = len(P)
    P = np.asarray(P)

    # Define amplitudes
    if (A is None):
        A = np.ones(n_waves)
    else:
        A = np.asarray(A)

    # Define phases
    if (phi is None):
        phi = np.zeros(n_waves)
    else:
        phi = np.asarray(phi)

    # Add all the waves
    t = np.linspace(0.0, np.amax(P), num=num)
    f = np.zeros(len(t))
    for i in range(n_waves):
        f = f + A[i] * np.sin(2.0 * np.pi * t / P[i] + phi[i])

    return t, f


def synthetic_FFT(X, multiv=False):
    """
    - univariate and single time-series
    - univariate and multi-time series (can be used to generate multi from same)
    - multi-variate multi-time series

    generate more than n_samples, remome multi-time series for uni-variate
    use 3d matrix looping on the time-series for multi-variate

    """
    n_samples, n_series = X.shape

    # The number of samples must be odd (if the number is even drop the last value)
    if ((n_samples % 2) == 0):
        print("Warning: data reduced by one (even number of samples)")
        n_samples = n_samples - 1
        X = X[0:n_samples, :]

    # FFT of the original data
    X_fft = np.fft.fft(X, axis=0)

    # Parameters
    half_len = (n_samples - 1) // 2
    idx1 = np.arange(1, half_len+1, dtype=int)
    idx2 = np.arange(half_len+1, n_samples, dtype=int)

    # If multivariate the random phases is the same
    if (multiv):
        phases = np.random.rand(half_len, 1)
        phases1 = np.tile(np.exp(2.0 * np.pi * 1j * phases), (1, n_series))
        phases2 = np.conj(np.flipud(phases1))

    # If univariate the random phases is different
    else:
        phases = np.random.rand(half_len, n_series)
        phases1 = np.exp(2.0 * np.pi * 1j * phases)
        phases2 = np.conj(np.flipud(phases1))

    # FFT of the synthetic data
    synt_fft = X_fft.copy()
    synt_fft[idx1, :] = X_fft[idx1, :] * phases1
    synt_fft[idx2, :] = X_fft[idx2, :] * phases2

    # Inverse FFT of the synthetic data
    X_synt = np.real(np.fft.ifft(synt_fft, axis=0))

    return X_synt


def synthetic_sampling(X, n_reps=1, replace=True):
    """
    X must be (n_samples, 1)
    """
    n_samples = X.shape[0]

    # Sampling with replacement
    if (replace):
        idx = np.random.randint(0, n_samples, size=(n_samples, n_reps))
        i = np.arange(n_reps)
        Xe = np.tile(X,(1, n_reps))
        X_synt = Xe[idx, i]

    # Sampling without replacement
    else:
        idx = np.zeros(n_samples, n_reps)
        for j in range(n_reps):
            idx[:, j] = np.random.permutation(n_samples)
        i = np.arange(n_reps)
        Xe = np.tile(X,(1, n_reps))
        X_synt = Xe[idx, i]

    return X_synt


def synthetic_MEboot(X, alpha=0.1, bounds=True, scale=False):
    """
    """
    n_samples, n_series = X.shape
    X_synt = np.zeros_like(X)

    # Loop over time-series iterate over number of desired series or set parallel
    # if possile
    n = n_samples
    for ts in range(n_series):
        
        # Sort the time series keeping track of the original position
        idx = np.argsort(X[:, ts])
        Y = X[idx, ts]

        # Compute the trimmed mean 
        # trimmed_mean = np.abs(np.diff(x)).mean()
        g = int(np.floor(n * alpha))
        r = n * alpha - g
        m_trm = ((1.0 - r) * (Y[g] + Y[n-g-1]) + Y[g+1:n-g-1].sum()) \
                 / (n * (1.0 - 2.0 * alpha))

        # Compute the intermediate points
        Z = np.zeros(n+1)
        Z[0] = Y[0] - m_trm
        Z[1:-1] = (Y[0:-1] + Y[1:]) / 2.0
        Z[n] = Y[n-1] + m_trm

        # Compute the interval means
        mt = np.zeros(n)
        mt[0] = 0.75 * Y[0] + 0.25 * Y[1]
        mt[1:n-1] = 0.25 * Y[0:n-2] + 0.5 * Y[1:n-1] + 0.25 * Y[2:n]
        mt[n-1] = 0.25 * Y[n-2] + 0.75 * Y[n-1]

        # Randomly generate new points
        # t_w = np.random.rand(n)
        t_w = np.array([0.12, 0.83, 0.53, 0.59, 0.11])
        t_w = np.sort(t_w)

        # Interpolate new points
        t = np.linspace(0.0, 1.0, num=n+1)
        w_int = np.interp(t_w, t, Z)

        # Correct the new points (first and last interval) to satisfy mass constraint
        idw = (np.floor_divide(t_w, 1.0 / n)).astype(int)
        corr = np.where(idw == 0, mt[idw] - (Z[idw] + Z[idw+1]) / 2.0, 0.0)
        w_corr = w_int + corr
        if (n > 1):
            corr = np.where(idw == n-1, mt[idw] - (Z[idw] + Z[idw+1]) / 2.0, 0.0)
            w_corr += corr

        # Enforce limits
        if (bounds):
            w_corr = np.fmin(np.fmax(w_corr, Z[0]), Z[n])

        # Recovery the time-dependencies (done all togheter?)
        X_synt = np.zeros(n)
        X_synt[idx] = w_corr

        # Scale (done all together?)
        if (scale):
            var_Z = np.diff(Z) ** 2.0 / 12.0
            X_mean = X.mean(axis=0)
            var_ME = (((mt - X_mean) ** 2).sum() + var_Z.sum()) / n
            std_X= X.std(axis=0, ddof=1)
            std_ME = np.sqrt(var_ME)
            k_scale = std_X / std_ME - 1.0
            X_synt = X_synt + k_scale * (X_synt - X_mean)

        return X_synt


