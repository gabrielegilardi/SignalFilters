"""
Signal Filtering/Smoothing and Generation of Synthetic Time-Series.

Copyright (c) 2020 Gabriele Gilardi


X           (n_samples, n_series)       Dataset to filter
b           (n_b, )                     Numerator coefficients
a           (n_a, )                     Denominator coefficients
Y           (n_samples, n_series)       Filtered dataset
idx         scalar                      First filtered element in Y

n_samples       Number of data to filter
n_series        Number of series to filter
nb              Number of coefficients (numerator)
na              Number of coefficients (denominator)

Notes:
- the filter is applied starting from index.
- non filtered data are set equal to the original input, i.e.
  Y[0:idx-1,:] = X[0:idx-1,:]
- if n_series = 1 then must be ( ..., 1)

Filters:

Generic         b,a             Generic case
SMA             N               Simple Moving Average
EMA             N/alpha         Exponential Moving Average
WMA             N               Weighted moving average
MSMA            N               Modified Simple Moving Average
MLSQ            N               Modified Least-Squares Quadratic (N=5,7,9,11)
ButterOrig      P,N             Butterworth original (N=2,3)
ButterMod       P,N             Butterworth modified (N=2,3)
SuperSmooth     P,N             Super smoother (N=2,3)
GaussLow        P,N             Gauss low pass (P>=2)
GaussHigh       P,N             Gauss high pass (P>=5)
BandPass        P,delta         Band-pass filter
BandStop        P,delta         Band-stop filter
ZEMA1           N/alpha,K,Vn    Zero-lag EMA (type 1)
ZEMA2           N/alpha,K       Zero-lag EMA (type 2)
InstTrend       N/alpha         Instantaneous trendline
SincFunction    N               Sinc function
Decycler        P               Decycler, 1-GaussHigh (P>=5)
DecyclerOsc     P1,P2           Decycle oscillator, GH(P1) - GH(P2), (P1>=5)
ABG
Kalman

N               Order/smoothing factor/number of previous samples
alpha           Damping term
P, P1, P2       Cut-off/critical period (50% power loss, -3 dB)
delta           Band centered in P and in fraction
                (30% of P => 0.3, = 0.3*P, if P = 12 => 0.3*12 = 4)
K               Coefficient/gain
Vn              Look back sample (for the momentum)

correction = update = measurement
prediction = motion

X       (n_states, 1)           State estimate
P       (n_states, n_states)    Covariance estimate
F       (n_states, n_states)    State transition model
Z       (n_obs, 1)              Observations
H       (n_obs, n_states)       Observation model
R       (n_obs, n_obs)          Covariance of the observation noise
S       (n_obs, n_obs)          Covariance of the observation residual
K       (n_states, n_obs)       Optimal Kalman gain
Q       (n_states, n_states)    Covariance of the process noise matrix
Y       (n_obs, 1)              Observation residual (innovation)
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def plot_signals(signals, start=0):
    """
    signals must be a list
    """
    legend = []
    count = 0
    for signal in signals:
        signal = signal.flatten()
        end = len(signal)
        t = np.arange(start, end)
        plt.plot(t, signal[start:end])
        legend.append('Signal [' + str(count) + ']')
        count += 1
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(b=True)
    plt.legend(legend)
    plt.show()


def filter_data(data, b, a):
    """
    Applies a filter with transfer response coefficients <a> and <b>.
    """
    n_samples = len(data)
    nb = len(b)
    na = len(a)
    idx = np.amax([0, nb-1, na-1])
    Y = data.copy()

    for i in range(idx, n_samples):

        tmp = 0

        for j in range(nb):
            tmp += b[j] * data[i-j]         # Numerator term

        for j in range(1, na):
            tmp -= a[j] * Y[i-j]            # Denominator term

        Y[i] = tmp / a[0]

    return Y, idx


class Filter:

    def __init__(self, data):
        """
        """
        self.data = np.asarray(data)
        self.n_samples = len(data)

    def Generic(self, b=1.0, a=1.0):
        """
        Filter with generic transfer response coefficients <a> and <b>.
        """
        self.b = np.asarray(b)
        self.a = np.asarray(a)
        Y, self.idx = filter_data(self.data, self.b, self.a)

        return Y

    def SMA(self, N=10):
        """
        Simple moving average (?? order, FIR, ?? band).
        """
        self.b = np.ones(N) / N
        self.a = np.array([1.0])
        Y, self.idx = filter_data(self.data, self.b, self.a)

        return Y

    def EMA(self, N=10, alpha=None):
        """
        Exponential moving average (?? order, IIR, pass ??).
        If not given, <alpha> is determined as equivalent to a N-SMA.
        """
        if (alpha is None):
            alpha = 2.0 / (N + 1.0)

        self.b = np.array([alpha])
        self.a = np.array([1.0, - (1.0 - alpha)])
        Y, self.idx = filter_data(self.data, self.b, self.a)

        return Y

    def WMA(self, N=10):
        """
        Weighted moving average (?? order, FIR, pass ??).
        Example: N = 5  -->  [5.0, 4.0, 3.0, 2.0, 1.0] / 15.0
        """
        w = np.arange(N, 0, -1)

        self.b = w / np.sum(w)
        self.a = np.array([1.0])
        Y, self.idx = filter_data(self.data, self.b, self.a)

        return Y

    def MSMA(self, N=10):
        """
        Modified simple moving average (?? order, FIR, pass ??).
        Example: N = 4  -->  [0.5, 1.0, 1.0, 1.0, 0.5] / 4.0
        """
        w = np.ones(N+1)
        w[0] = 0.5
        w[N] = 0.5

        self.b = w / N
        self.a = np.array([1.0])
        Y, self.idx = filter_data(self.data, self.b, self.a)

        return Y

    def MLSQ(self, N=5):
        """
        Modified simple moving average (?? order, FIR, pass ??).
        Only N = 5, 7, 9, and 11 are implemented. If not returns the unfiltered
        dataset.
        """
        if (N == 5):
            w = np.array([7.0, 24.0, 34.0, 24.0, 7.0]) / 96.0

        elif (N == 7):
            w = np.array([1.0, 6.0, 12.0, 14.0, 12.0, 6.0, 1.0]) / 52.0

        elif (N == 9):
            w = np.array([-1.0, 28.0, 78.0, 108.0, 118.0, 108.0, 78.0, 28.0,
                          -1.0]) / 544.0

        elif (N == 11):
            w = np.array([-11.0, 18.0, 88.0, 138.0, 168.0, 178.0, 168.0, 138.0,
                           88.0, 18.0, -11.0]) / 980.0

        else:
            print("Warning: data returned unfiltered (wrong N)")
            self.idx = 0
            return self.data

        self.b = w
        self.a = np.array([1.0])
        Y, self.idx = filter_data(self.data, self.b, self.a)

        return Y

    def ButterOrig(self, N=2, P=10):
        """
        Butterworth original version (?? order, IIR, pass ??).
        Only N = 2 and 3 are implemented. If not returns the unfiltered dataset.
        """
        if (N == 2):
            beta = np.exp(-np.sqrt(2.0) * np.pi / P)
            alpha = 2.0 * beta * np.cos(np.sqrt(2.0) * np.pi / P)
            wb = np.array([1.0, 2.0, 1.0]) * (1.0 - alpha + beta ** 2.0) / 4.0
            wa = np.array([1.0, - alpha, beta ** 2.0])

        elif (N == 3):
            beta = np.exp(-np.pi / P)
            alpha = 2.0 * beta * np.cos(np.sqrt(3.0) * np.pi / P)
            wb = np.array([1.0, 3.0, 3.0, 1.0]) * (1.0 - beta ** 2.0) \
                 * (1.0 - alpha + beta ** 2.0) / 8.0
            wa = np.array([1.0, - (alpha + beta ** 2.0),
                          (1.0 + alpha) * beta ** 2.0, - beta ** 4.0])

        else:
            print("Warning: data returned unfiltered (wrong N)")
            self.idx = 0
            return self.data

        self.b = wb
        self.a = wa
        Y, self.idx = filter_data(self.data, self.b, self.a)

        return Y

    def ButterMod(self, N=2, P=10):
        """
        Butterworth modified version (?? order, IIR, pass ??).
        Only N = 2 and 3 are implemented. If not returns the unfiltered dataset.
        """
        if (N == 2):
            beta = np.exp(-np.sqrt(2.0) * np.pi / P)
            alpha = 2.0 * beta * np.cos(np.sqrt(2.0) * np.pi / P)
            wb = np.array([1.0 - alpha + beta ** 2.0])
            wa = np.array([1.0, - alpha, beta ** 2.0])

        elif (N == 3):
            beta = np.exp(-np.pi / P)
            alpha = 2.0 * beta * np.cos(np.sqrt(3.0) * np.pi / P)
            wb = np.array([1.0 - alpha * (1.0 - beta ** 2.0) - beta ** 4.0])
            wa = np.array([1.0, - (alpha + beta ** 2.0),
                          (1.0 + alpha) * beta ** 2.0, - beta ** 4.0])

        else:
            print("Warning: data returned unfiltered (wrong N)")
            self.idx = 0
            return self.data

        self.b = wb
        self.a = wa
        Y, self.idx = filter_data(self.data, self.b, self.a)

        return Y

    def SuperSmooth(self, N=2, P=10):
        """
        SuperSmooth (?? order, IIR, pass ??).
        Only N = 2 and 3 are implemented. If not returns the unfiltered dataset.
        """
        if (N == 2):
            beta = np.exp(-np.sqrt(2.0) * np.pi / P)
            alpha = 2.0 * beta * np.cos(np.sqrt(2.0) * np.pi / P)
            wb = np.array([1.0, 1.0]) * (1.0 - alpha + beta ** 2.0) / 2.0
            wa = np.array([1.0, - alpha, beta ** 2.0])

        elif (N == 3):
            beta = np.exp(-np.pi / P)
            alpha = 2.0 * beta * np.cos(1.738 * np.pi / P)
            wb = np.array([1.0, 1.0]) \
                 * (1.0 - alpha * (1.0 - beta ** 2.0) - beta ** 4.0) / 2.0
            wa = np.array([1.0, - (alpha + beta ** 2.0),
                          (1.0 + alpha) * beta ** 2.0, - beta ** 4.0])

        else:
            print("Warning: data returned unfiltered (wrong N)")
            self.idx = 0
            return self.data

        self.b = wb
        self.a = wa
        Y, self.idx = filter_data(self.data, self.b, self.a)

        return Y

    def GaussLow(self, N=1, P=2):
        """
        Gauss low pass (IIR, N-th order, low pass).
        Must be P > 1. If not returns the unfiltered dataset.
        """
        if (P < 2):
            print("Warning: data returned unfiltered (P < 2)")
            self.idx = 0
            return self.data

        A = 2.0 ** (1.0 / N) - 1.0
        B = 4.0 * np.sin(np.pi / P) ** 2.0
        C = 2.0 * (np.cos(2.0 * np.pi / P) - 1.0)
        alpha = (-B + np.sqrt(B ** 2.0 - 4.0 * A * C)) / (2.0 * A)

        self.b = np.array([alpha])
        self.a = np.array([1.0, - (1.0 - alpha)])
        Y = self.data.copy()
        for i in range(N):
            Y, self.idx = filter_data(Y, self.b, self.a)

        return Y

    def GaussHigh(self, N=1, P=5):
        """
        Gauss high pass (IIR, Nth order, high pass).
        Must be P > 4. If not returns the unfiltered dataset.
        """
        if (P < 5):
            print("Warning: data returned unfiltered (P < 5)")
            self.idx = 0
            return self.data

        A = 2.0 ** (1.0 / N) * np.sin(np.pi / P) ** 2.0 - 1.0
        B = 2.0 * (2.0 ** (1.0 / N) - 1.0) * (np.cos(2.0 * np.pi / P) - 1.0)
        C = - B
        alpha = (-B - np.sqrt(B ** 2.0 - 4.0 * A * C)) / (2.0 * A)

        self.b = np.array([1.0 - alpha / 2.0, -(1.0 - alpha / 2.0)])
        self.a = np.array([1.0, - (1.0 - alpha)])
        Y = self.data - self.data[0, :]
        for i in range(N):
            Y, self.idx = filter_data(Y, self.b, self.a)

        return Y

    def BandPass(self, P=5, delta=0.3):
        """
        Band-pass (type, order, IIR).
        Example: delta = 0.3, P = 12
                (30% of P => 0.3, = 0.3*P, if P = 12 => 0.3*12 = 4)
        """
        beta = np.cos(2.0 * np.pi / P)
        gamma = np.cos(4.0 * np.pi * delta / P)
        alpha = 1.0 / gamma - np.sqrt(1.0 / gamma ** 2 - 1.0)

        self.b = np.array([(1.0 - alpha) / 2.0, 0.0, - (1.0 - alpha) / 2.0])
        self.a = np.array([1.0, - beta * (1.0 + alpha), alpha])
        Y, self.idx = filter_data(self.data, self.b, self.a)

        return Y

    def BandStop(self, P=5, delta=0.3):
        """
        Band-stop (type, order, IIR)
        Example: delta = 0.3, P = 12
                (30% of P => 0.3, = 0.3*P, if P = 12 => 0.3*12 = 4)
        """
        beta = np.cos(2.0 * np.pi / P)
        gamma = np.cos(4.0 * np.pi * delta / P)
        alpha = 1.0 / gamma - np.sqrt(1.0 / gamma ** 2 - 1.0)

        self.b = np.array([1.0, - 2.0 * beta,  1.0]) * (1.0 + alpha) / 2.0
        self.a = np.array([1.0, - beta * (1.0 + alpha), alpha])
        Y, self.idx = filter_data(self.data, self.b, self.a)

        return Y

    def ZEMA1(self, N=10, alpha=None, K=1.0, Vn=5):
        """
        Zero lag Exponential Moving Average (type 1).
        If not given, <alpha> is determined as equivalent to a N-SMA.
        """
        if (alpha is None):
            alpha = 2.0 / (N + 1.0)

        w = np.zeros(Vn+1)
        w[0] = alpha * (1.0 + K)
        w[Vn] = - alpha * K

        self.b = w
        self.a = np.array([1.0, - (1.0 - alpha)])
        Y, self.idx = filter_data(self.data, self.b, self.a)

        return Y

    def ZEMA2(self, N=10, alpha=None, K=1.0):
        """
        Zero lag Exponential Moving Average (type 2).
        If not given, <alpha> is determined as equivalent to a N-SMA.
        """
        if (alpha is None):
            alpha = 2.0 / (N + 1.0)

        self.b = np.array([alpha * (1.0 + K)])
        self.a = np.array([1.0, alpha * K - (1.0 - alpha)])
        Y, self.idx = filter_data(self.data, self.b, self.a)

        return Y

    def InstTrend(self, N=10, alpha=None):
        """
        Instantaneous Trendline (2nd order, IIR, low pass).
        If not given, <alpha> is determined as equivalent to a N-SMA.
        """
        if (alpha is None):
            alpha = 2.0 / (N + 1.0)

        self.b = np.array([alpha - alpha ** 2.0 / 4.0, alpha ** 2.0 / 2.0,
                           - alpha + 3.0 * alpha ** 2.0 / 4.0])
        self.a = np.array([1.0, - 2.0 * (1.0 - alpha), (1.0 - alpha) ** 2.0])
        Y, self.idx = filter_data(self.data, self.b, self.a)

        return Y

    def SincFunction(self, N=10, nel=10):
        """
        Sinc function (order, FIR, pass).
        (N > 1, cut off at 0.5/N)
        """
        K = np.arange(1, nel)
        w = np.zeros(nel)
        w[0] = 1.0 / N
        w[1:] = np.sin(np.pi * K / N) / (np.pi * K)

        self.b = w
        self.a = np.array([1.0])
        Y, self.idx = filter_data(self.data, self.b, self.a)

        return Y

    def Decycler(self, P=10):
        """
        Decycler (?? order, IIR, pass ??). Gauss,HP,1st,P
        Built subtracting high pass Gauss filter from 1 (order 1)
        Must be P > 4. If not returns the unfiltered dataset.
        """
        if (P < 5):
            print("Warning: data returned unfiltered (P < 5)")
            self.idx = 0
            return self.data

        Y = self.data - self.GaussHigh(N=1, P=P)
        return Y

    def DecyclerOsc(self, P1=5, P2=10):
        """
        DecyclerOsc (?? order 2, IIR, pass ??).

        (Gauss, HP, 2nd order, Pmax - Gauss, HP, 2nd order, Pmin)
        P1 = 1st cut off period, P2 = 2nd cut off period. Automatically fixed.
        Must be P1, P2 > 4. If not returns the unfiltered dataset.
        """
        P_low = np.amin([P1, P2])
        P_high = np.amax([P1, P2])

        if (P_low < 5):
            print("Warning: data returned unfiltered (P_low < 5)")
            self.idx = 0
            return self.data

        Y = self.GaussHigh(N=2, P=P_low) - self.GaussHigh(N=2, P=P_high)
        return Y

    def ABG(self, alpha=0.0, beta=0.0, gamma=0.0, dt=1.0):
        """
        alpha-beta-gamma
        For numerical stability: 0 < alpha, beta < 1
        """
        # If necessary change scalars to arrays
        if (np.ndim(alpha) == 0):
            alpha = np.ones(self.n_samples) * alpha
        if (np.ndim(beta) == 0):
            beta = np.ones(self.n_samples) * beta
        if (np.ndim(gamma) == 0):
            gamma = np.ones(self.n_samples) * gamma

        # Initialize
        Y_corr = self.data.copy()
        Y_pred = self.data.copy()
        x0 = self.data[0, :]
        v0 = np.zeros(self.n_series)
        a0 = np.zeros(self.n_series)

        for i in range(1, self.n_samples):

            # Predictor (predicts state in <i>)
            x_pred = x0 + dt * v0 + 0.5 * a0 * dt ** 2.0
            v_pred = v0 + dt * a0
            a_pred = a0
            Y_pred[i, :] = x_pred

            # Residual (innovation)
            r = self.data[i, :] - x_pred

            # Corrector (corrects state in <i>)
            x_corr = x_pred + alpha[i] * r
            v_corr = v_pred + (beta[i] / dt) * r
            a_corr = a_pred + (2.0 * gamma[i] / dt ** 2.0) * r

            # Save value and prepare next iteration
            Y_corr[i, :] = x_corr
            x0 = x_corr
            v0 = v_corr
            a0 = a_corr

        self.idx = 1

        return Y_corr, Y_pred

    def Kalman(self, sigma_x, sigma_v, dt, abg_type="abg"):
        """
        Steady-state Kalman filter (also limited to one-dimension)
        """
        L = (sigma_x / sigma_v) * dt ** 2.0

        # Alpha filter
        if (abg_type == 'a'):
            alpha = (-L ** 2.0 + np.sqrt(L ** 4.0 + 16.0 * L ** 2.0)) / 8.0
            beta = 0.0
            gamma = 0.0

        # Alpha-Beta filter
        elif(abg_type == 'ab'):
            r = (4.0 + L - np.sqrt(8.0 * L + L ** 2.0)) / 4.0
            alpha = 1.0 - r ** 2.0
            beta = 2.0 * (2.0 - alpha) - 4.0 * np.sqrt(1.0 - alpha)
            gamma = 0.0

        # Alpha-Beta-Gamma filter
        else:
            b = (L / 2.0) - 3.0
            c = (L / 2.0) + 3.0
            d = - 1.0
            p = c - b ** 2.0 / 3.0
            q = (2.0 * b ** 3.0) / 27.0 - (b * c) / 3.0 + d
            v = np.sqrt(q ** 2.0 + (4.0 * p ** 3.0) / 27.0)
            z = - (q + v / 2.0) ** (1.0 / 3.0)
            s = z - p / (3.0 * z) - b / 3.0
            alpha = 1.0 - s ** 2.0
            beta = 2.0 * (1 - s) ** 2.0
            gamma = (beta ** 2.0) / (2.0 * alpha)

        # Apply filter
        Y = self.abg(alpha=alpha, beta=beta, gamma=gamma, dt=dt)

        return Y

    def plot_frequency(self):
        """
        """
        w, h = signal.freqz(self.b, self.a)
        h_db = 20.0 * np.log10(np.abs(h))
        wf = w / (2.0 * np.pi)
        plt.plot(wf, h_db)
        plt.axhline(-3.0, lw=1.5, ls='--', C='r')
        plt.grid(b=True)
        plt.xlim(np.amin(wf), np.amax(wf))
        plt.xlabel(r'$\omega$ [rad/sample]')
        plt.ylabel('$h$ [db]')
        legend = ['Filter', '-3dB']
        plt.legend(legend)
        plt.show()

    def plot_lag(self):
        """
        """
        w, gd = signal.group_delay((self.b, self.a))
        wf = w / (2.0 * np.pi)
        plt.plot(wf, gd)
        plt.grid(b=True)
        plt.xlim(np.amin(wf), np.amax(wf))
        plt.xlabel(r'$\omega$ [rad/sample]')
        plt.ylabel('$gd$ [samples]')
        plt.show()
