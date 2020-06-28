"""
Signal Filtering and Generation of Synthetic Time-Series.

Copyright (c) 2020 Gabriele Gilardi


X           (n_samples, )       Dataset to filter (input)
b           (n_b, )             Transfer response coefficients (numerator)
a           (n_a, )             Transfer response coefficients (denominator)
Y           (n_samples, )       Filtered dataset (output)
idx         scalar              First filtered element in Y

n_samples       Number of samples in the input dataset
nb              Number of coefficients in array <b>
na              Number of coefficients in array <a>

Notes:
- the filter is applied starting from index idx = MAX(0, nb-1, na-1).
- non filtered data are set equal to the input, i.e. Y[0:idx-1] = X[0:idx-1]
- X must be a 1D array.


Filter list:
-----------
Generic         b, a            Generic filter
SMA             N               Simple moving average
EMA             N/alpha         Exponential moving average
WMA             N               Weighted moving average
MSMA            N               Modified simple moving average
MLSQ            N               Modified least-squares quadratic (N = 5, 7, 9, 11)
ButterOrig      P, N            Butterworth original filter (N = 2, 3)
ButterMod       P, N            Butterworth modified filter (N = 2, 3)
SuperSmooth     P, N            Supersmoother filter (N = 2, 3)
GaussLow        P, N            Gauss low pass filter (P > 1)
GaussHigh       P, N            Gauss high pass filter (P > 4)
BandPass        P, delta        Band-pass filter
BandStop        P, delta        Band-stop filter
ZEMA1           N/alpha, K, Vn  Zero-lag EMA (type 1)
ZEMA2           N/alpha, K      Zero-lag EMA (type 2)
InstTrend       N/alpha         Instantaneous trendline
SincFilter      P, nel          Sinc function filter (N > 1)
Decycler        P               De-cycler filter (P >= 4)
DecyclerOsc     P1, P2          De-cycle oscillator (P >= 4)
ABG             alpha, beta,    Alpha-beta-gamma filter (0 < alpha, beta < 1)
                gamma, dt
Kalman          sigma_x, dt     One-dimensional steady-state Kalman filter
                sigma_v

N               Order/smoothing factor/number of previous samples
alpha           Damping term
P, P1, P2       Cut-off/critical period (50% power loss, -3 dB)
delta           Semi-band centered in P
K               Coefficient/gain
Vn              Look-back sample
nel             Number of frequencies in the sinc function
alpha           Parameter(s) to correct the position in the ABG filter
beta            Parameter(s) to correct the velocity in the ABG filter
gamma           Parameter(s) to correct the acceleration in the ABG filter
dt              Sampling interval in the ABG and Kalman filters
sigma_x         Process variance in the Kalman filter
sigma_v         Noise variance in the Kalman filter
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def plot_signals(signals, names=None, start=0, end=None):
    """
    Plot the signals specified in list <signals> with their names specified in
    list <names>. Each signal is plotted in its full length unless differently
    specified.
    """
    # Identify the signals by index if their name is not specified
    if (names is None):
        legend = []
        count = 0
    else:
        legend = names

    # Loop over the signals
    t_max = 0
    for signal in signals:

        signal = signal.flatten()
        t_max = np.amax([t_max, len(signal)])
        plt.plot(signal)

        # If no name is given use the list index to identify the signals
        if (names is None):
            legend.append('Signal [' + str(count) + ']')
            count += 1

    # Plot and format
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(b=True)
    plt.legend(legend)
    if (end is None):
        plt.xlim(start, t_max)
    else:
        plt.xlim(start, end)
    plt.show()


def filter_data(data, b, a):
    """
    Applies a filter with transfer response coefficients <b> (numerator) and
    <a> (denominator).
    """
    n_samples = len(data)
    nb = len(b)
    na = len(a)
    idx = np.amax([0, nb-1, na-1])          # Index of the 1st filtered sample
    Y = data.copy()

    # Loop over the samples
    for i in range(idx, n_samples):

        tmp = 0

        # Contribution from the numerator term (input samples)
        for j in range(nb):
            tmp += b[j] * data[i-j]

        # Contribution from the denominator term (previous output samples)
        for j in range(1, na):
            tmp -= a[j] * Y[i-j]

        Y[i] = tmp / a[0]

    return Y, idx


class Filter:

    def __init__(self, data):
        """
        Initialize the filter object.
        """
        self.data = np.asarray(data).flatten()
        self.idx = 0
        self.b = 0.0
        self.a = 0.0

    def Generic(self, b=1.0, a=1.0):
        """
        Filter with generic transfer response coefficients <b> and <a>.
        """
        self.b = np.asarray(b)
        self.a = np.asarray(a)
        Y, self.idx = filter_data(self.data, self.b, self.a)

        return Y

    def SMA(self, N=5):
        """
        Simple moving average.
        """
        self.b = np.ones(N) / N
        self.a = np.array([1.0])
        Y, self.idx = filter_data(self.data, self.b, self.a)

        return Y

    def EMA(self, N=5, alpha=None):
        """
        Exponential moving average.

        If not given, <alpha> is determined as equivalent to a N-SMA.
        """
        if (alpha is None):
            alpha = 2.0 / (N + 1.0)

        self.b = np.array([alpha])
        self.a = np.array([1.0, - (1.0 - alpha)])
        Y, self.idx = filter_data(self.data, self.b, self.a)

        return Y

    def WMA(self, N=5):
        """
        Weighted moving average.

        Example: N = 5  -->  [5, 4, 3, 2, 1] / 15.
        """
        w = np.arange(N, 0, -1)

        self.b = w / np.sum(w)
        self.a = np.array([1.0])
        Y, self.idx = filter_data(self.data, self.b, self.a)

        return Y

    def MSMA(self, N=5):
        """
        Modified simple moving average.

        Example: N = 5  -->  [1/2, 1, 1, 1, 1, 1/2] / 5.
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
        Modified least-squares quadratic.

        Must be N = 5, 7, 9, or 11. If wrong N, prints a warning and returns
        the unfiltered dataset.
        """
        if (N == 5):
            w = np.array([7, 24, 34, 24, 7]) / 96

        elif (N == 7):
            w = np.array([1, 6, 12, 14, 12, 6, 1]) / 52

        elif (N == 9):
            w = np.array([-1, 28, 78, 108, 118, 108, 78, 28, -1]) / 544

        elif (N == 11):
            w = np.array([-11, 18, 88, 138, 168, 178, 168, 138, 88, 18,
                          -11]) / 980

        else:
            print("Warning: data returned unfiltered (MLSQ - Wrong N)")
            self.idx = 0
            return self.data

        self.b = w
        self.a = np.array([1.0])
        Y, self.idx = filter_data(self.data, self.b, self.a)

        return Y

    def ButterOrig(self, N=2, P=10):
        """
        Butterworth original filter.

        Must be N = 2 or 3. If wrong N, prints a warning and returns the
        unfiltered dataset.
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
            print("Warning: data returned unfiltered (ButterOrig - Wrong N)")
            self.idx = 0
            return self.data

        self.b = wb
        self.a = wa
        Y, self.idx = filter_data(self.data, self.b, self.a)

        return Y

    def ButterMod(self, N=2, P=10):
        """
        Butterworth modified filter. It is derived from the Butterworth original
        filter deleting all but the constant term at the numerator.

        Must be N = 2 or 3. If wrong N, prints a warning and returns the
        unfiltered dataset.
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
            print("Warning: data returned unfiltered (ButterMod - Wrong N)")
            self.idx = 0
            return self.data

        self.b = wb
        self.a = wa
        Y, self.idx = filter_data(self.data, self.b, self.a)

        return Y

    def SuperSmooth(self, N=2, P=10):
        """
        Supersmoother filter. It is derived from the Butterworth modified
        filter adding a two-element moving average at the numerator.

        Must be N = 2 or 3. If wrong N, prints a warning and returns the
        unfiltered dataset.
        """
        if (N == 2):
            beta = np.exp(-np.sqrt(2.0) * np.pi / P)
            alpha = 2.0 * beta * np.cos(np.sqrt(2.0) * np.pi / P)
            wb = np.array([1.0, 1.0]) * (1.0 - alpha + beta ** 2.0) / 2.0
            wa = np.array([1.0, - alpha, beta ** 2.0])

        elif (N == 3):
            beta = np.exp(-np.pi / P)
            alpha = 2.0 * beta * np.cos(np.sqrt(3.0) * np.pi / P)
            wb = np.array([1.0, 1.0]) \
                 * (1.0 - alpha * (1.0 - beta ** 2.0) - beta ** 4.0) / 2.0
            wa = np.array([1.0, - (alpha + beta ** 2.0),
                          (1.0 + alpha) * beta ** 2.0, - beta ** 4.0])

        else:
            print("Warning: data returned unfiltered (SuperSmooth - Wrong N)")
            self.idx = 0
            return self.data

        self.b = wb
        self.a = wa
        Y, self.idx = filter_data(self.data, self.b, self.a)

        return Y

    def GaussLow(self, N=1, P=10):
        """
        Gauss low pass filter.

        Must be P > 1. If wrong P, prints a warning and returns the unfiltered
        dataset.
        """
        if (P <= 1):
            print("Warning: data returned unfiltered (GaussLow - Wrong P)")
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

    def GaussHigh(self, N=1, P=10):
        """
        Gauss high pass filter.

        Must be P > 4. If wrong P, prints a warning and returns the unfiltered
        dataset.
        """
        if (P <= 4):
            print("Warning: data returned unfiltered (GaussHigh - Wrong P)")
            self.idx = 0
            return self.data

        A = 2.0 ** (1.0 / N) * np.sin(np.pi / P) ** 2.0 - 1.0
        B = 2.0 * (2.0 ** (1.0 / N) - 1.0) * (np.cos(2.0 * np.pi / P) - 1.0)
        C = - B
        alpha = (-B - np.sqrt(B ** 2.0 - 4.0 * A * C)) / (2.0 * A)

        self.b = np.array([1.0 - alpha / 2.0, -(1.0 - alpha / 2.0)])
        self.a = np.array([1.0, - (1.0 - alpha)])
        Y = self.data - self.data[0]            # Shift to zero
        for i in range(N):
            Y, self.idx = filter_data(Y, self.b, self.a)

        return Y

    def BandPass(self, P=10, delta=0.3):
        """
        Band-pass filter.

        Example: delta = 0.3, P = 10  -->  0.3 * 10 = 3 -->  band is [7, 13]
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
        Band-stop filter.

        Example: delta = 0.3, P = 10  -->  0.3 * 10 = 3 -->  band is [7, 13]
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
        Zero-lag EMA (type 1). It is an alpha-beta type filter with sub-optimal
        parameters.

        If not given, <alpha> is determined as equivalent to a N-SMA.
        """
        if (alpha is None):
            alpha = 2.0 / (N + 1.0)

        w = np.zeros(Vn + 1)
        w[0] = alpha * (1.0 + K)
        w[Vn] = - alpha * K

        self.b = w
        self.a = np.array([1.0, - (1.0 - alpha)])
        Y, self.idx = filter_data(self.data, self.b, self.a)

        return Y

    def ZEMA2(self, N=10, alpha=None, K=1.0):
        """
        Zero-lag EMA (type 2). It is derived from the type 1 ZEMA removing the
        look-back term Vn.

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
        Instantaneous Trendline. It is created by removing the dominant cycle
        from the signal.

        If not given, <alpha> is determined as equivalent to a N-SMA.
        """
        if (alpha is None):
            alpha = 2.0 / (N + 1.0)

        self.b = np.array([alpha - alpha ** 2.0 / 4.0, alpha ** 2.0 / 2.0,
                           - alpha + 3.0 * alpha ** 2.0 / 4.0])
        self.a = np.array([1.0, - 2.0 * (1.0 - alpha), (1.0 - alpha) ** 2.0])
        Y, self.idx = filter_data(self.data, self.b, self.a)

        return Y

    def SincFilter(self, P=10, nel=10):
        """
        Sinc function filter. The cut off point is at 0.5/P.

        Must be P > 1. If wrong P, prints a warning and returns the unfiltered
        dataset.
        """
        if (P <= 1):
            print("Warning: data returned unfiltered (SincFilter - Wrong P)")
            self.idx = 0
            return self.data

        K = np.arange(1, nel)
        w = np.zeros(nel)
        w[0] = 1.0 / P
        w[1:] = np.sin(np.pi * K / P) / (np.pi * K)

        self.b = w
        self.a = np.array([1.0])
        Y, self.idx = filter_data(self.data, self.b, self.a)

        return Y

    def Decycler(self, P=10):
        """
        De-cycler filter. It is derived subtracting a 1st order high pass Gauss
        filter from 1.

        Must be P > 4. If wrong P, prints a warning and returns the unfiltered
        dataset.
        """
        if (P <= 4):
            print("Warning: data returned unfiltered (Decycler - Wrong P)")
            self.idx = 0
            return self.data

        Y = self.data - self.GaussHigh(N=1, P=P)
        return Y

    def DecyclerOsc(self, P1=5, P2=10):
        """
        De-cycler oscillator. It is derived subtracting a 2nd order high pass
        Gauss filter with higher cut-off period from a 2nd order high pass Gauss
        filter with higher cut-off period.

        Must be P > 4. If wrong P, prints a warning and returns the unfiltered
        dataset.
        """
        P_low = np.amin([P1, P2])
        P_high = np.amax([P1, P2])

        if (P_low <= 4):
            print("Warning: data returned unfiltered (DecyclerOsc - Wrong P)")
            self.idx = 0
            return self.data

        Y = self.GaussHigh(N=2, P=P_low) - self.GaussHigh(N=2, P=P_high)
        return Y

    def ABG(self, alpha=0.0, beta=0.0, gamma=0.0, dt=1.0):
        """
        Alpha-beta-gamma filter. It is a predictor-corrector type of filter.

        Arguments alpha, beta, and gamma can be a scalar (used for all samples)
        or an array with one value for each sample. For numerical stability it
        should be 0 < alpha, beta < 1.
        """
        n_samples = len(self.data)
        Y = np.zeros(n_samples)

        # Change scalar arguments to arrays if necessary
        if (np.ndim(alpha) == 0):
            alpha = np.ones(n_samples) * alpha
        if (np.ndim(beta) == 0):
            beta = np.ones(n_samples) * beta
        if (np.ndim(gamma) == 0):
            gamma = np.ones(n_samples) * gamma

        # Initialize
        x0 = self.data[0]
        v0 = 0.0
        a0 = 0.0
        Y[0] = x0

        for i in range(1, n_samples):

            # Predictor (predicts state in <i>)
            x_pred = x0 + dt * v0 + 0.5 * a0 * dt ** 2.0
            v_pred = v0 + dt * a0
            a_pred = a0

            # Residual (innovation)
            r = self.data[i] - x_pred

            # Corrector (corrects state in <i>)
            x_corr = x_pred + alpha[i] * r
            v_corr = v_pred + (beta[i] / dt) * r
            a_corr = a_pred + (2.0 * gamma[i] / dt ** 2.0) * r

            # Save value and prepare next iteration
            x0 = x_corr
            v0 = v_corr
            a0 = a_corr
            Y[i] = x_corr

        self.idx = 1

        return Y

    def Kalman(self, sigma_x, sigma_v, dt, abg_type="abg"):
        """
        One-dimensional steady-state Kalman filter. It is obtained from the
        alpha-beta-gamma filter using the process variance, the noise variance
        and optimizing the three parameters.

        Arguments sigma_x and sigma_v can be a scalar (used for all samples) or
        an array with one value for each sample.
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

        # Apply the alpha-beta-gamma filter
        Y = self.ABG(alpha=alpha, beta=beta, gamma=gamma, dt=dt)

        return Y

    def plot_response(self):
        """
        Plots the frequency response (in decibels) and the lag (group delay)
        of the filter with coefficients <b> and <a>.
        """
        # Frequency response
        w, h = signal.freqz(self.b, self.a)
        h_db = 20.0 * np.log10(np.abs(h))       # Convert to decibels

        # Lag / Group delay
        w, gd = signal.group_delay((self.b, self.a))

        # Scale frequency to [0, 0.5]
        wf = w / (2.0 * np.pi)

        # Plot and format frequency response
        plt.subplot(1, 2, 1)
        plt.plot(wf, h_db)
        plt.axhline(-3.0, lw=1.5, ls='--', C='r')   # -3 dB (50% power loss)
        plt.grid(b=True)
        plt.xlim(np.amin(wf), np.amax(wf))
        plt.xlabel(r'$\omega$ [rad/sample]')
        plt.ylabel('$h$ [db]')
        legend = ['Filter', '-3dB']
        plt.legend(legend)
        plt.title('Frequency Response')

        # Plot and format lag/group delay
        plt.subplot(1, 2, 2)
        plt.plot(wf, gd)
        plt.grid(b=True)
        plt.xlim(np.amin(wf), np.amax(wf))
        plt.xlabel(r'$\omega$ [rad/sample]')
        plt.ylabel('$gd$ [samples]')
        plt.title('Lag / Group Delay')

        # Show plots
        plt.suptitle('Example "Response"')
        plt.show()
