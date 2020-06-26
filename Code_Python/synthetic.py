"""
Signal Filtering/Smoothing and Generation of Synthetic Time-Series.

Copyright (c) 2020 Gabriele Gilardi
"""

import numpy as np


def synthetic_wave(P, A=None, phi=None, num=1000):
    """
    Generates a multi-sine wave given a periods, amplitudes, and phases.

    P           (n, )               Periods
    A           (n, )               Amplitudes
    phi         (n, )               Phases (rad)
    t           (num, )             Time
    f           (num, )             Multi-sine wave

    The default value for the amplitudes is 1 and for the phases is zero. The
    time goes from zero to the largest period.
    """
    n_waves = len(P)                # Number of waves
    P = np.asarray(P)

    # Amplitudes
    if (A is None):
        A = np.ones(n_waves)        # Default is 1
    else:
        A = np.asarray(A)

    # Phases
    if (phi is None):
        phi = np.zeros(n_waves)     # Default is 0
    else:
        phi = np.asarray(phi)

    # Time
    t = np.linspace(0.0, np.amax(P), num=num)

    # Add up all the sine waves
    f = np.zeros(len(t))
    for i in range(n_waves):
        f = f + A[i] * np.sin(2.0 * np.pi * t / P[i] + phi[i])

    return t, f


def synthetic_FFT(X, n_reps=1):
    """
    Generates surrogates of the time-serie X using the phase-randomized
    Fourier-transform algorithm. Input X needs to be a 1D array.

    X           (n, )               Original time-series
    X_fft       (n, )               FFT of the original time-series
    X_synt_fft  (n_reps, n)         FFT of the synthetic time-series
    X_synt      (n_reps, n)         Synthetic time-series
    """
    X = X.flatten()             # Reshape to (n, )
    n = len(X)

    # The number of samples must be odd
    if ((n % 2) == 0):
        print("Warning: data reduced by one (even number of samples)")
        n = n - 1
        X = X[0:n, :]

    # FFT of the original time-serie
    X_fft = np.fft.fft(X)

    # Parameters
    half_len = (n - 1) // 2
    idx1 = np.arange(1, half_len+1, dtype=int)          # 1st half
    idx2 = np.arange(half_len+1, n, dtype=int)          # 2nd half

    # Generate the random phases
    phases = np.random.rand(n_reps, half_len)
    phases1 = np.exp(2.0 * np.pi * 1j * phases)
    phases2 = np.conj(np.flipud(phases1))

    # FFT of the synthetic time-series (1st sample is unchanged)
    X_synt_fft = np.zeros((n_reps, n), dtype=complex)
    X_synt_fft[:, 0] = X_fft[0]
    X_synt_fft[:, idx1] = X_fft[idx1] * phases1         # 1st half
    X_synt_fft[:, idx2] = X_fft[idx2] * phases2         # 2nd half

    # Synthetic time-series
    X_synt = np.real(np.fft.ifft(X_synt_fft, axis=1))

    return X_synt


def synthetic_sampling(X, n_reps=1, replace=True):
    """
    Generates surrogates of the time-serie X using randomized-sampling
    (bootstrap) with or without replacement. Input X needs to be a 1D array.

    X           (n, )               Original time-series
    idx         (n_reps, n)         Random index of X
    X_synt      (n_reps, n)         Synthetic time-series
    """
    X = X.flatten()             # Reshape to (n, )
    n = len(X)

    # Sampling with replacement
    if (replace):
        idx = np.random.randint(0, n, size=(n_reps, n))

    # Sampling without replacement
    else:
        idx = np.argsort(np.random.rand(n_reps, n), axis=1)

    # Synthetic time-series
    X_synt = X[idx]

    return X_synt


def synthetic_MEboot(X, n_reps=1, alpha=0.1, bounds=False, scale=False):
    """
    Generates surrogates of the time-serie X using the maximum entropy
    bootstrap algorithm. Input X needs to be a 1D array.

    X       (n, )           Original time-series
    idx     (n, )           Original order of X
    y       (n, )           Sorted original time-series
    z       (n+1, )         Intermediate points
    mt      (n, )           Interval means
    t_w     (n_reps, n)     Random new points
    w_int   (n_reps, n)     Interpolated new points
    w_corr  (n_reps, n)     Interpolated new points with corrections for first
                            and last interval
    X_synt  (n_reps, n)     Synthetic time-series
    """
    X = X.flatten()             # Reshape to (n, )
    n = len(X)

    # Sort the time series keeping track of the original order
    idx = np.argsort(X)
    y = X[idx]

    # Trimmed mean
    g = int(np.floor(n * alpha))
    r = n * alpha - g
    m_trm = ((1.0 - r) * (y[g] + y[n-g-1]) + y[g+1:n-g-1].sum()) \
            / (n * (1.0 - 2.0 * alpha))

    # Intermediate points
    z = np.zeros(n+1)
    z[0] = y[0] - m_trm
    z[1:-1] = (y[0:-1] + y[1:]) / 2.0
    z[n] = y[n-1] + m_trm

    # Interval means
    mt = np.zeros(n)
    mt[0] = 0.75 * y[0] + 0.25 * y[1]
    mt[1:n-1] = 0.25 * y[0:n-2] + 0.5 * y[1:n-1] + 0.25 * y[2:n]
    mt[n-1] = 0.25 * y[n-2] + 0.75 * y[n-1]

    # Generate randomly new points and sort them
    t_w = np.random.rand(n_reps, n)
    t_w = np.sort(t_w, axis=1)

    # Interpolate the new points
    t = np.linspace(0.0, 1.0, num=n+1)
    w_int = np.interp(t_w, t, z)

    # Correct the new points in the first and last interval to satisfy
    # the mass constraint
    idw = (np.floor_divide(t_w, 1.0 / n)).astype(int)
    corr = np.where(idw == 0, mt[idw] - (z[idw] + z[idw+1]) / 2.0, 0.0)
    w_corr = w_int + corr
    if (n > 1):
        corr = np.where(idw == n-1, mt[idw] - (z[idw] + z[idw+1]) / 2.0, 0.0)
        w_corr += corr

    # Enforce limits (if requested)
    if (bounds):
        w_corr = np.fmin(np.fmax(w_corr, z[0]), z[n])

    # Recovery the time-dependency of the original time-series
    X_synt = np.zeros((n_reps, n))
    X_synt[:, idx] = w_corr

    # Scale to force equal variance (if requested)
    if (scale):
        var_z = np.diff(z) ** 2.0 / 12.0
        X_mean = X.mean(axis=0)
        var_ME = (((mt - X_mean) ** 2).sum() + var_z.sum()) / n
        std_X = X.std(ddof=1)
        std_ME = np.sqrt(var_ME)
        k_scale = std_X / std_ME - 1.0
        X_synt = X_synt + k_scale * (X_synt - X_mean)

    return X_synt


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
    Returns the 1st discrete difference of array X.

    X           (n, )           Input dataset
    dX          (n-1, )         1st discrete differences

    Notes:
    - the discrete difference can be calculated in percent or in value.
    - dX is one element shorter than X.
    - X needs to be a 1D array.
    """
    X = X.flatten()             # Reshape to (n, )

    # Discrete difference in percent
    if (percent):
        dX = X[1:] / X[:-1] - 1.0

    # Discrete difference in value
    else:
        dX = X[1:] - X[:-1]

    return dX


def diff2value(dX, X0, percent=True):
    """
    Returns array X from the 1st discrete difference using X0 as initial value.

    dX          (n, )           Discrete differences
    X0          scalar          Initial value
    X           (n+1, )         Output dataset

    Notes:
    - the discrete difference can be in percent or in value.
    - X is one element longer than dX.
    - dX needs to be a 1D array.

    If the discrete difference is in percent:
        X[0] = X0
        X[1] = X[0] * (1 + dX[0])
        X[2] = X[1] * (1 + dX[1]) = X[0] * (1 + dX[0]) * (1 + dX[1])
        ....

    If the discrete difference is in value:
        X[0] = X0
        X[1] = X[0] + dX[0]
        X[2] = X[1] + dX[1] = X[0] + dX[0] + dX[1]
        ....
    """
    dX = dX.flatten()               # Reshape to (n, )
    X = np.zeros(len(dX) + 1)
    X[0] = X0                       # Initial value

    # Discrete difference in percent
    if (percent):
        X[1:] = X0 * np.cumprod(1.0 + dX)

    # Discrete difference in value
    else:
        X[1:] = X0 + np.cumsum(dX)

    return X
