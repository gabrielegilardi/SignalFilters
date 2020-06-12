"""
Class for filter/smooth data.

Copyright (c) 2020 Gabriele Gilardi


N = order/smoothing factor/number of past bars
        alpha = damping term

General          b,a        Generic case (param = [b; a])    
SMA              N          Simple Moving Average
EMA              N          Exponential Moving Average
PassBand         P,delta    Pass band filter
StopBand         P,delta    Stop band filter
InstTrendline    alpha      Instantaneous trendline
GaussLow         P,N        Gauss, low pass (must be P > 1)
ZEMA1            N,K,Vn     Zero-lag EMA (type 1)

ZEMA2            N,K        Zero-lag EMA (type 2)
LWMA             N          Linearly Weighted Moving Average
MSMA             N          Modified Simple Moving Average
MLSQ             N          Modified Least-Squares Quadratic
GaussHigh        P,N        Gauss, high pass (must be P > 4)
ButterOrig       P,N        Butterworth original, order N (2 or 3)
ButterMod        P,N        Butterworth modified, order N (2 or 3)
SuperSmoother    P, N       Super smoother
SincFunction     N          Sinc function (N > 1, cut off at 0.5/N)

b            Coefficients at the numerator
a            Coefficients at the denominator
P            Cut of period (50% power loss, -3 dB)
N            Order/smoothing factor
K            Coefficient/gain
Vn           Look back bar for the momentum
delta        Band centered in P and in percent 
            (0.3 => 30% of P, = 0.3*P, if P = 10 => 0.3*10 = 3)
alpha        Damping term
nt           Times the filter is called (order)


"""

import sys
import numpy as np


def filter_data(X, b, a):
    """
    Applies a generic filter.

    Inputs:
    X           (n_samples, n_series)          Data to filter
    b           Transfer response coefficients (numerator)
    a           Transfer response coefficients (denominator)

    Outputs:
    Y           Filtered data
    idx         Index of the first element in Y filtered

    Notes:
    - the filter is applied from element 0 to len(X).
    - elements from 0 to (idx-1) are set equal to the original input.
    """
    n_samples, n_series = X.shape
    Nb = len(b)
    Na = len(a)
    idx = np.amax([0, Nb-1, Na-1])
    Y = X.copy()

    # Apply filter
    for i in range(idx, n_samples):

        tmp = np.zeros(n_series)

        # Contribution from [b] (numerator)
        for j in range(Nb):
            tmp = tmp + b[j] * X[i-j,:]

        # Contribution from [a] (denominator)
        for j in range(1, Na):
            tmp = tmp - a[j] * Y[i-j, :]

        # Filtered value
        Y[i,:] = tmp / a[0]

    return Y, idx


class Filter:

    def __init__(self, X):
        """
        X       (nel_X, )          Data to filter
        """
        self.X = np.asarray(X)
        self.n_samples, self.n_series = X.shape
        self.idx = 0

    def SMA(self, N=10):
        """
        Simple moving average (type ???).
        """
        b = np.ones(N) / N
        a = np.array([1.0])
        Y, self.idx = filter_data(self.X, b, a)
        return Y

    def EMA(self, N=10, alpha=None):
        """
        Exponential moving average (type ???).
        
        If <alpha> is not given it is determined as equivalent to a N-SMA.
        """
        if (alpha is None):
            alpha = 2.0 / (N + 1.0)
        b = np.array([alpha])
        a = np.array([1.0, alpha - 1.0])
        Y, self.idx = filter_data(self.X, b, a)
        return Y

    def InstTrend(self, alpha=0.5):
        """
        Instantaneous Trendline (2nd order, IIR, low pass, Ehlers.)
        """
        b = np.array([alpha - alpha ** 2.0 / 4.0, alpha ** 2.0 / 2.0,
                      - alpha + 3.0 * alpha ** 2.0 / 4.0])
        a = np.array([1.0, - 2.0 * (1.0 - alpha), (1.0 - alpha) ** 2.0])
        Y, self.idx = filter_data(self.X, b, a)
        return Y

    def PassBand(self, P=5, delta=0.3):
        """
        Pass Band (type ???).

        P = cut-off period (50% power loss, -3 dB)
        delta = band centered in P and in percent
                (Example: 0.3 => 30% of P => 0.3*P, if P = 10 => 0.3*10 = 3)
        """
        beta = np.cos(2.0 * np.pi / P)
        gamma = np.cos(4.0 * np.pi * delta) / P
        alpha = 1.0 / gamma - np.sqrt(1.0 / gamma ** 2 - 1.0)
        b = np.array([(1.0 - alpha) / 2.0, 0.0, - (1.0 - alpha) / 2.0])
        a = np.array([1.0, - beta * (1.0 + alpha), alpha])
        Y, self.idx = filter_data(self.X, b, a)
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

    def Generalized(self, a, b):
        """
        Generic filter with transfer response coefficients <a> and <b>
        """
        b = np.asarray(b)
        a = np.asarray(a)
        Y, self.idx = filter_data(self.X, b, a)
        return Y

