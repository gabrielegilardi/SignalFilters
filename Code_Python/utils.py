"""
Utility functions for ????.

Copyright (c) 2020 Gabriele Gilardi
"""

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


def calc_rmse(a, b):
    """
    Calculates the root-mean-square-error of arrays <a> and <b>. If the arrays
    are multi-column, the RMSE is calculated as all the columns are one single
    vector.
    """
    # Convert to (n, ) dimension
    a = a.flatten()
    b = b.flatten()

    # Root-mean-square-error
    rmse = np.sqrt(((a - b) ** 2).sum() / len(a))

    return rmse


def calc_corr(a, b):
    """
    Calculates the correlation between arrays <a> and <b>. If the arrays are
    multi-column, the correlation is calculated as all the columns are one
    single vector.
    """
    # Convert to (n, ) dimension
    a = a.flatten()
    b = b.flatten()

    # Correlation
    corr = np.corrcoef(a, b)[0, 1]

    return corr


def calc_accu(a, b):
    """
    Calculates the accuracy (in %) between arrays <a> and <b>. The two arrays
    must be column/row vectors.
    """
    # Convert to (n, ) dimension
    a = a.flatten()
    b = b.flatten()

    # Correlation
    accu = 100.0 * (a == b).sum() / len(a)

    return accu


def plot_signals(signals):
    """
    """
    for signal in signals:
        plt.plot(signal)

    plt.grid(b=True)
    plt.show()
