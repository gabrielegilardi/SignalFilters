"""
Utility functions for ????.

Copyright (c) 2020 Gabriele Gilardi
"""

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt


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


def plot_signals(signals, idx_start=0, idx_end=None):
    """
    """
    if (idx_end is None):
        idx_end = len(signals[0])
    t = np.arange(idx_start, idx_end)
    names = []
    count = 0
    for signal in signals:
        plt.plot(t, signal[idx_start:idx_end])
        names.append(str(count))
        count += 1
    plt.grid(b=True)
    plt.legend(names)
    plt.show()


def plot_frequency_response(b, a=1.0):
    """
    """
    b = np.asarray(b)
    a = np.asarray(a)

    w, h = signal.freqz(b, a)
    h_db = 20.0 * np.log10(abs(h))
    wf = w / (2.0 * np.pi)

    plt.plot(wf, h_db)
    plt.axhline(-3.0, lw=1.5, ls='--', C='r')
    plt.grid(b=True)
    plt.xlim(np.amin(wf), np.amax(wf))
    # plt.ylim(-40.0, 0.0)
    plt.xlabel('$\omega$ [rad/sample]')
    plt.ylabel('$h$ [db]')
    plt.show()


def plot_lag_response(b, a=1.0):
    """
    """
    b = np.asarray(b)
    a = np.asarray(a)

    w, gd = signal.group_delay((b, a))
    wf = w / (2.0 * np.pi)

    plt.plot(wf, gd)
    plt.grid(b=True)
    plt.xlim(np.amin(wf), np.amax(wf))
    plt.xlabel('$\omega$ [rad/sample]')
    plt.ylabel('$gd$ [samples]')
    plt.show()


def synthetic_wave(P, A=None, PH=None, num=1000):
    """
    Generates a multi-sinewave.
    P = [ P1 P2 ... Pn ]      Periods
    A = [ A1 A2 ... An ]      Amplitudes
    PH = [PH1 PH2 ... PHn]    Phases (rad)

    Default amplitudes are ones
    Default phases are zeros
    Time is from 0 to largest period (default 1000 steps)
    """
    n_waves = len(P)
    P = np.asarray(P)

    # Define amplitudes and phases
    if (A is None):
        A = np.ones(n_waves)
    else:
        A = np.asarray(A)
    if (PH is None):
        PH = np.zeros(n_waves)
    else:
        PH = np.asarray(PH)

    # Add all the waves
    t = np.linspace(0.0, np.amax(P), num=num)
    f = np.zeros(len(t))
    for i in range(n_waves):
        f = f + A[i] * np.sin(2.0 * np.pi * t / P[i] + PH[i])

    return t, f


def synthetic_series(data, multivariate=False):
    """
    """
    n_samples, n_series = data.shape

    # The number of samples must be odd (if the number is even drop the last value)
    if ((n_samples % 2) == 0):
        print("Warning: data reduced by one (even number of samples)")
        n_samples = n_samples - 1
        data = data[0:n_samples, :]

    # FFT of the original data
    data_fft = np.fft.fft(data, axis=0)

    # Parameters
    half_len = (n_samples - 1) // 2
    idx1 = np.arange(1, half_len+1, dtype=int)
    idx2 = np.arange(half_len+1, n_samples, dtype=int)

    # If multivariate the random phases is the same
    if (multivariate):
        phases = np.random.rand(half_len, 1)
        phases1 = np.tile(np.exp(2.0 * np.pi * 1j * phases), (1, n_series))
        phases2 = np.conj(np.flipud(phases1))

    # If univariate the random phases is different
    else:
        phases = np.random.rand(half_len, n_series)
        phases1 = np.exp(2.0 * np.pi * 1j * phases)
        phases2 = np.conj(np.flipud(phases1))

    # FFT of the synthetic data
    synt_fft = data_fft.copy()
    synt_fft[idx1, :] = data_fft[idx1, :] * phases1
    synt_fft[idx2, :] = data_fft[idx2, :] * phases2

    # Inverse FFT of the synthetic data
    synt_data = np.real(np.fft.ifft(synt_fft, axis=0))

    return synt_data
