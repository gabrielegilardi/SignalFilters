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


def value2diff(X, mode=None):
    """
    from value to difference in abs or %
    diff in value first element is zero
    diff in % first element is one
    """

    # Difference in value
    if (mode == 'V'):
        dX = np.zeros_like(X)
        dX[1:, :] = X[1:, :] - X[:-1, :]
    
    # Difference in percent
    else:
        dX = np.ones_like(X)
        dX[1:, :] = X[1:, :] / X[:-1, :] - 1.0
    
    return dX


def diff2value(dX, mode=None):
    """
    from difference in abs or % to value (first row should be all zeros but
    will be over-written

    Reference X[0,:] is assumed to be zero. If X0[0,:] is the desired
    reference, the actual vector X can be determined as X0+X

    Reference X[0,:] is assumed to be one. If X0[0,:] is the desired
    reference, the actual vector X can be determined as X0*X
    """
    # Value from the difference (first row equal to zero)
    # X[0, :] = 0
    # X[1, :] = X[0, :] + dX[1, :] = dX[1, :]
    # X[2, :] = X[0, :] + dX[1, :] + dX[2, :] = dX[1, :] + dX[2, :]
    # ....
    if (mode == 'V'):
        X = np.zeros_like(dX)
        X[1:, :] = np.cumsum(dX[1:, :], axis=0)
    
    # Value from percent (first row equal to 1)
    # X[0, :] = 1
    # X[1, :] = X[0, :] * (1 + dX[1, :]) = (1 + dX[1, :])
    # X[2, :] = X[1, :] * (1 + dX[2, :])
    #         = X[0, :] * (1 + dX[1, :]) * (1 + dX[2, :])
    #         = (1 + dX[1, :]) * (1 + dX[2, :])
    # ....
    else:
        X = np.ones_like(dX)
        X[1:, :] = np.cumprod((1.0 + dX), axis=0)
    
    return X


def synthetic_wave(per, amp=None, pha=None, num=1000):
    """
    Generates a multi-sinewave.
    P = [ P1 P2 ... Pn ]      Periods
    A = [ A1 A2 ... An ]      Amplitudes
    PH = [PH1 PH2 ... PHn]    Phases (rad)

    Default amplitudes are ones
    Default phases are zeros
    Time is from 0 to largest period (default 1000 steps)
    """
    n_waves = len(per)
    per = np.asarray(per)

    # Define amplitudes and phases
    if (amp is None):
        amp = np.ones(n_waves)
    else:
        amp = np.asarray(amp)
    if (pha is None):
        pha = np.zeros(n_waves)
    else:
        pha = np.asarray(pha)

    # Add all the waves
    t = np.linspace(0.0, np.amax(per), num=num)
    f = np.zeros(len(t))
    for i in range(n_waves):
        f = f + amp[i] * np.sin(2.0 * np.pi * t / per[i] + pha[i])

    return t, f


def synthetic_series(X, multiv=False):
    """
    """
    n_samples, n_series = data.shape

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
