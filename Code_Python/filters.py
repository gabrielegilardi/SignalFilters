"""
Class for filter/smooth data.

Copyright (c) 2020 Gabriele Gilardi


ToDo:
- generalize to multidimensional input arrays
- use NaN or input values for points not filtered?
- plot filtered data
- add plot filter

"""

import sys
import numpy as np


def Generalized(X, b, a):
    """
    Applies a generic filter

    Inputs:
    X           Data to filter
    b           Transfer response coefficients (numerator)
    a           Transfer response coefficients (denominator)

    Outputs:
    Y           Filtered data
    idx         Index first element in Y actually filtered

    Elements from 0 to (idx-1) are set equal to NaN.
    """
    # Initialize
    nel_X = len(X)
    nel_b = len(b)
    nel_a = len(a)
    idx = np.amax([0, nel_b-1, nel_a-1])
    Y = X.copy()

    # Apply filter
    for i in range(idx, nel_X):
        tmp = 0.0

        # Contribution from [b] (numerator)
        for j in range(nel_b):
            tmp = tmp + b[j] * X[i-j]

        # Contribution from [a] (denominator)
        for j in range(1, nel_a):
            tmp = tmp - a[j] * Y[i-j]

        # Filtered value
        Y[i] = tmp / a[0]

        # Set elements from 0 to (idx-1) equal to NaN
        Y[0:idx] = np.nan

    return Y, idx


class Filter:

    def __init__(self, X):
        """
        """
        self.X = np.asarray(X)
        self.nel = len(X)
        self.idx = 0

    def SMA(self, N=10):
        """
        Simple Moving Average
        N = order/smoothing factor
        """
        b = np.ones(float(N)) / float(N)
        a = np.array([1.0])
        Y, self.idx = Generalized(self.X, b, a)
        return Y

    def EMA(self, N=10):
        """
        Exponential Moving Average
        N = order/smoothing factor
        The damping term <alpha> is determined as equivalent to a N-SMA
        """
        alpha = 2.0 / (float(N) + 1.0)
        b = np.array([alpha])
        a = np.array([1.0, alpha-1.0])
        Y, self.idx = Generalized(self.X, b, a)
        return Y

    def InstTrend(self, alpha=0.5):
        """
        Instantaneous Trendline (2nd order, IIR, low pass, Ehlers)
        alpha = damping term
        """
        b = np.array([(alpha-alpha**2/4.0), (alpha**2/2.0),
                      -(alpha-3.0*alpha**2/4.0)])
        a = np.array([1.0, -2.0*(1.0-alpha), (1.0-alpha)**2])
        Y, self.idx = Generalized(self.X, b, a)
        return Y

    def PassBand(self, P=5, delta=0.3):
        """
        Pass Band
        P = cut-off period (50% power loss, -3 dB)
        delta = band centered in P and in percent
                (Example: 0.3 => 30% of P => 0.3*P, if P = 10 => 0.3*10 = 3)
        """
        beta = np.cos(2.0 * pi / float(P))
        gamma = np.cos(2.0*pi*(2.0*delta)/float(P))
        alpha = 1.0 / gamma - np.sqrt(1.0 / gamma ** 2 - 1.0)
        b = np.array([(1.0-alpha)/2.0, 0.0, -(1.0-alpha)/2.0])
        a = np.array([1.0, -beta*(1.0+alpha), alpha])
        Y, self.idx = Generalized(self.X, b, a)
        return Y

    def StopBand(self, P=5, delta=0.3):
        """
        Stop Band
        P = cut-off period (50% power loss, -3 dB)
        delta = band centered in P and in percent
                (Example: 0.3 => 30% of P => 0.3*P, if P = 10 => 0.3*10 = 3)
        """
        beta = cos(2.0*pi/float(P))
        gamma = cos(2.0*pi*(2.0*delta)/float(P))
        alpha = 1.0/gamma - sqrt(1.0/gamma**2 - 1.0)
        b = np.array([(1.0+alpha)/2.0, -2.0*beta*(1.0+alpha)/2.0,
                      (1.0+alpha)/2.0])
        a = np.array([1.0, -beta*(1.0+alpha), alpha])
        Y, self.idx = Generalized(self.X, b, a)
        return Y

    def GaussLow(self, P=2, N=1):
        """
        Gauss Low (low pass, IIR, N-th order, must be P > 1)
        P = cut-off period (50% power loss, -3 dB)
        N = times the filter is called (order)
        """
        P = np.array([2, P], dtype=int).max()       # or error? warning?
        A = 2.0**(1.0/float(N)) - 1.0
        B = 4.0*sin(pi/float(P))**2.0
        C = 2.0*(cos(2.0*pi/float(P)) - 1.0)
        delta = sqrt(B**2.0 - 4.0*A*C)
        alpha = (-B + delta)/(2.0*A)
        b = np.array([alpha])
        a = np.array([1.0, -(1.0-alpha)])
        Y = np.copy(self.X)
        for i in range(N):
            Y, self.idx = Generalized(Y, b, a)
        return Y

    def ZEMA1(self, N=10, K=1.0, Vn=5):
        """
        Zero lag Exponential Moving Average (type 1)
        N = order/smoothing factor
        K = coefficient/gain
        Vn = look back bar for the momentum
        The damping term <alpha> is determined as equivalent to a N-SMA
        """
        alpha = 2.0 / (float(N) + 1.0)
        b = np.zeros(Vn+1)
        b[0] = alpha * (1.0 + K)
        b[-1] = - alpha * K
        a = np.array([1.0, -(1.0-alpha)])
        Y, self.idx = Generalized(self.X, b, a)
        return Y
