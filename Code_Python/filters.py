"""
Class for filter/smooth data.

Copyright (c) 2020 Gabriele Gilardi


References (both from John F. Ehlers):
[1] "Cycle Analytics for Traders: Advanced Technical Trading Concepts".
[2] "Signal Analysis, Filters And Trading Strategies".


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


Filters (ref. [1])
------------------

Generic         b, a        Generic case
SMA             N           Simple Moving Average
EMA             N           Exponential Moving Average
WMA             N           Weighted moving average
MSMA            N           Modified Simple Moving Average
MLSQ            N           Modified Least-Squares Quadratic (N = 5, 7, 9, 11)
ButterOrig      P, N        Butterworth original (N = 2, 3)
ButterMod       P, N        Butterworth modified (N = 2, 3)
SuperSmoother   P, N        Super smoother (N = 2, 3)
GaussLow        P, N        Gauss, low pass (P > 1)
GaussHigh       P, N        Gauss, high pass (P > 4)
Decycler        P           Decycler
DecyclerOsc     P1, P2      Decycle oscillator



ZEMA1           N, K, Vn    Zero-lag EMA (type 1)
ZEMA2           N, K        Zero-lag EMA (type 2)
MEMA            N, Ns       Modified EMA (with cubic velocity extimation)
PassBand        P, delta    Pass band filter
StopBand        P, delta    Stop band filter
InstTrendline   alpha       Instantaneous trendline
SincFunction    N           Sinc function (N > 1, cut off at 0.5/N)
Roofing         P1, P2      Gauss,HP,2nd,P1 + Supersmoother,P2

N               Order/smoothing factor/number of previous samples
alpha           Damping term
P, P1, P2       Cut-off/critical period (50% power loss, -3 dB)


K            Coefficient/gain
Vn           Look back bar for the momentum
delta        Band centered in P and in percent 
            (0.3 => 30% of P, = 0.3*P, if P = 10 => 0.3*10 = 3)
nt           Times the filter is called (order)
Ns           Look back bar/skip factor


"""

import sys
import numpy as np
import utils as utl


def filter_data(X, b, a):
    """
    Applies a filter with transfer response coefficients <a> and <b>.
    """
    n_samples, n_series = X.shape
    nb = len(b)
    na = len(a)
    idx = np.amax([0, nb - 1, na - 1])
    Y = X.copy()

    for i in range(idx, n_samples):
        tmp = np.zeros(n_series)

        for j in range(nb):
            tmp = tmp + b[j] * X[i-j,:]             # Numerator term

        for j in range(1, na):
            tmp = tmp - a[j] * Y[i-j, :]            # Denominator term

        Y[i,:] = tmp / a[0]

    return Y, idx


class Filter:

    def __init__(self, X):
        """
        """
        self.X = np.asarray(X)

        self.n_samples, self.n_series = X.shape
        self.idx = 0

    def Generic(self, b=1.0, a=1.0):
        """
        Filter with generic transfer response coefficients <a> and <b>.
        """
        b = np.asarray(b)
        a = np.asarray(a)
        Y, self.idx = filter_data(self.X, b, a)
        return Y

    def SMA(self, N=10):
        """
        Simple moving average (?? order, FIR, ?? band).
        """
        b = np.ones(N) / N
        a = np.array([1.0])
        Y, self.idx = filter_data(self.X, b, a)
        return Y

    def EMA(self, N=10, alpha=None):
        """
        Exponential moving average (?? order, IIR, pass ??).
        
        If not given, <alpha> is determined as equivalent to a N-SMA.
        """
        if (alpha is None):
            alpha = 2.0 / (N + 1.0)
        b = np.array([alpha])
        a = np.array([1.0, -(1.0 - alpha)])
        Y, self.idx = filter_data(self.X, b, a)
        return Y

    def WMA(self, N=10):
        """
        Weighted moving average (?? order, FIR, pass ??).
        
        Example: N = 5  -->  [5.0, 4.0, 3.0, 2.0, 1.0] / 15.0
        """
        w = np.arange(N,0,-1)
        b = w / np.sum(w)
        a = np.array([1.0])
        Y, self.idx = filter_data(self.X, b, a)
        return Y

    def MSMA(self, N=10):
        """
        Modified simple moving average (?? order, FIR, pass ??).
        
        Example: N = 4  -->  [0.5, 1.0, 1.0, 1.0, 0.5] / 4.0
        """
        w = np.ones(N+1)
        w[0] = 0.5
        w[N] = 0.5
        b = w / N
        a = np.array([1.0])
        Y, self.idx = filter_data(self.X, b, a)
        return Y

    def MLSQ(self, N=5):
        """
        Modified simple moving average (?? order, FIR, pass ??).
        
        Only N = 5, 7, 9, and 11 are implemented. If not return the unfiltered
        dataset.
        """
        if (N == 5):
            b = np.array([7.0, 24.0, 34.0, 24.0, 7.0]) / 96.0
        elif (N == 7):
            b = np.array([1.0, 6.0, 12.0, 14.0, 12.0, 6.0, 1.0]) / 52.0
        elif (N == 9):
            b = np.array([-1.0, 28.0, 78.0, 108.0, 118.0, 108.0, 78.0, 28.0, \
                          -1.0]) / 544.0
        elif (N == 11):     
            b = np.array([-11.0, 18.0, 88.0, 138.0, 168.0, 178.0, 168.0, \
                          138.0, 88.0, 18.0, -11.0]) / 980.0
        else:
            Y = self.X.copy()
            return Y
        a = np.array([1.0])
        Y, self.idx = filter_data(self.X, b, a)
        return Y

    def ButterOrig(self, N=2, P=10):
        """
        Butterworth original version (?? order, IIR, pass ??).

        Only N = 2 and 3 are implemented. If not return the unfiltered dataset.
        """
        if (N == 2):
            beta = np.exp(-np.sqrt(2.0) * np.pi / P)
            alpha = 2.0 * beta * np.cos(np.sqrt(2.0) * np.pi / P)
            b = np.array([1.0, 2.0, 1.0]) * (1.0 - alpha + beta ** 2.0) / 4.0
            a = np.array([1.0, -alpha, beta ** 2.0])
        elif (N == 3):
            beta = np.exp(-np.pi / P)
            alpha = 2.0 * beta * np.cos(np.sqrt(3.0) * np.pi / P)
            b = np.array([1.0, 3.0, 3.0, 1.0]) \
                * (1.0 - alpha + beta ** 2.0) * (1.0 - beta ** 2.0) / 8.0
            a = np.array([1.0, - (alpha + beta ** 2.0), \
                (1.0 + alpha) * beta ** 2.0, - beta ** 4.0])
        else:
            Y = self.X.copy()
            return Y
        Y, self.idx = filter_data(self.X, b, a)
        return Y


    def ButterMod(self, N=2, P=10):
        """
        Butterworth modified version (?? order, IIR, pass ??).

        Only N = 2 and 3 are implemented. If not return the unfiltered dataset.
        """
        if (N == 2):
            beta = np.exp(-np.sqrt(2.0) * np.pi / P)
            alpha = 2.0 * beta * np.cos(np.sqrt(2.0) * np.pi / P)
            b = np.array([1.0 - alpha + beta ** 2.0])
            a = np.array([1.0, -alpha, beta ** 2.0])
        elif (N == 3):
            beta = np.exp(-np.pi / P)
            alpha = 2.0 * beta * np.cos(np.sqrt(3.0) * np.pi / P)
            b = np.array([1.0 - alpha * (1.0 - beta ** 2.0) - beta ** 4.0])
            a = np.array([1.0, - (alpha + beta ** 2.0), \
                (1.0 + alpha) * beta ** 2.0, - beta ** 4.0])
        else:
            Y = self.X.copy()
            return Y
        Y, self.idx = filter_data(self.X, b, a)
        return Y


    def SuperSmoother(self, N=2, P=10):
        """
        SuperSmoother (?? order, IIR, pass ??).
        
        Only N = 2 and 3 are implemented. If not return the unfiltered dataset.
        """
        if (N == 2):
            beta = np.exp(-np.sqrt(2.0) * np.pi / P)
            alpha = 2.0 * beta * np.cos(np.sqrt(2.0) * np.pi / P)
            w = 1.0 - alpha + beta ** 2.0
            b = np.array([w, w]) / 2.0
            a = np.array([1.0, - alpha, beta ** 2.0])
        elif (N == 3):
            beta = np.exp(-np.pi / P)
            alpha = 2.0 * beta * np.cos(1.738 * np.pi / P)
            w = 1.0 - alpha * (1.0 - beta ** 2.0) - beta ** 4.0
            b = np.array([w, w]) / 2.0
            a = np.array([1.0, - (alpha + beta ** 2.0), \
                (1.0 + alpha) * beta ** 2.0, - beta ** 4.0])
        else:
            Y = self.X.copy()
            return Y
        Y, self.idx = filter_data(self.X, b, a)
        return Y

    def GaussLow(self, P=2, N=1):
        """
        Gauss low pass (IIR, N-th order, low pass).

        Must be P > 1. If not return the unfiltered dataset.
        """
        if (P < 2):
            Y = self.X.copy()
            return Y
        A = 2.0 ** (1.0 / N) - 1.0
        B = 4.0 * np.sin(np.pi / P) ** 2.0
        C = 2.0 * (np.cos(2.0 * np.pi/ P) - 1.0)
        alpha = (-B + np.sqrt(B ** 2.0 - 4.0 * A * C)) / (2.0 * A)
        b = np.array([alpha])
        a = np.array([1.0, - (1.0 - alpha)])
        Y = self.X.copy()
        for i in range(N):
            Y, self.idx = filter_data(Y, b, a)
        return Y, b, a

    def GaussHigh(self, P=5, N=1):
        """
        Gauss high pass (IIR, Nth order, high pass).
        
        Must be P > 4. If not return the unfiltered dataset.
        """
        if (P < 5):
            Y = self.X.copy()
            return Y
        A = 2.0 ** (1.0 / N) * np.sin(np.pi / P) ** 2.0 - 1.0
        B = 2.0 * (2.0 ** (1.0 / N) - 1.0) * (np.cos(2.0 * np.pi / P) - 1.0)
        C = - B
        alpha = (-B - np.sqrt(B ** 2.0 - 4.0 * A * C)) / (2.0 * A)
        b = np.array([1.0 - alpha / 2.0, -(1.0 - alpha / 2.0)])
        a = np.array([1.0, - (1.0 - alpha)])
        Y = self.X - self.X[0, :]
        for i in range(N):
            Y, self.idx = filter_data(Y, b, a)
        return Y, b, a



    def Decycler(self, P=10):
        """
        Decycler (?? order, IIR, pass ??). Gauss,HP,1st,P
        """
        pass

    # % Built subtracting high pass Gauss filter from 1 (only order 1)
    # case 'Decycler'
    #   P = param(1);
    #   [TMP,nLast] = FiltersBasic(IN,'GaussHigh',[P 1]);
    #   OUT = IN - TMP;

    def DecyclerOsc(self, Pmin=1, Pmax=5):
        """
        Decycler (?? order, IIR, pass ??). Gauss,HP,2nd,Pmax - Gauss,HP,2nd,Pmin
        """
        pass

    # % (Gauss, HP, 2nd, Pmax - Gauss, HP, 2nd Pmin)
    # % Pmax = larger cut off period
    # % Pmin = smaller cut off period
    # case 'DecyclerOsc'
    #   P1 = min(param);
    #   P2 = max(param);
    #   [TMP1,nLast] = FiltersBasic(IN,'GaussHigh',[P1 2]);
    #   [TMP2,nLast] = FiltersBasic(IN,'GaussHigh',[P2 2]);
    #   OUT = TMP2 - TMP1;




    def InstTrend(self, alpha=0.5):
        """
        Instantaneous Trendline (2nd order, IIR, low pass, Ehlers.).
        """
        b = np.array([alpha - alpha ** 2.0 / 4.0, alpha ** 2.0 / 2.0,
                      - alpha + 3.0 * alpha ** 2.0 / 4.0])
        a = np.array([1.0, - 2.0 * (1.0 - alpha), (1.0 - alpha) ** 2.0])
        Y, self.idx = filter_data(self.X, b, a)
        return Y

    def PassBand(self, P=5, delta=0.3):
        """
        Pass Band (type ???).
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
        """
        beta = cos(2.0*pi/float(P))
        gamma = cos(2.0*pi*(2.0*delta)/float(P))
        alpha = 1.0/gamma - sqrt(1.0/gamma**2 - 1.0)
        b = np.array([(1.0+alpha)/2.0, -2.0*beta*(1.0+alpha)/2.0,
                      (1.0+alpha)/2.0])
        a = np.array([1.0, -beta*(1.0+alpha), alpha])
        Y, self.idx = Generalized(self.X, b, a)
        return Y


    def ZEMA1(self, N=10, alpha=None, K=1.0, Vn=5):
        """
        Zero lag Exponential Moving Average (type 1).

        If not given, <alpha> is determined as equivalent to a N-SMA.
        """
        if (alpha is None):
            alpha = 2.0 / (N + 1.0)
        alpha = 2.0 / (float(N) + 1.0)
        b = np.zeros(Vn+1)
        b[0] = alpha * (1.0 + K)
        b[-1] = - alpha * K
        a = np.array([1.0, -(1.0-alpha)])
        Y, self.idx = Generalized(self.X, b, a)
        return Y



#     % Zero lag Exponential Moving Average (type 2)
#     % alpha is determined as equivalent to a N-SMA
#     case 'ZEMA2'
#       N = param(1);
#       alpha = 2/(N+1);
#       K = param(2);
#       b = alpha*(1+K);
#       a = [1  alpha*K-(1-alpha)];
#       [OUT,nLast] = GeneralizedFilter(IN,b,a,3);
     

 
      
            
#     % Sinc function
#     case 'SincFunction'
#       N = param(1);
#       nel = 50;
#       k=1:nel-1;
#       b = [ 1/N   sin(pi*k/N)./(pi*k) ];
#       a = 1;
#       [OUT,nLast] = GeneralizedFilter(IN,b,a,1);
     
    # % Roofing filter: Gauss high pass 2nd order filter + SuperSmoother
    # % P1 = cut off period for GaussHigh
    # % P2 = cut off period for SuperSmoother
    # case 'Roofing'
    #   P1 = param(1);
    #   P2 = param(2);
    #   [TMP,nLast] = FiltersBasic(IN,'GaussHigh',[P1 2]);
    #   [OUT,nLast] = FiltersBasic(TMP,'SuperSmoother',[P2 1]);


    # % MEMA - Uses cubic velocity extimation
    # case 'MEMA'
    #   N = param(1);
    #   Ns = param(2);
    #   alpha = 2/(N+1);
    #   [Vel,Acc,nLast] = FiltersMak(IN,'Cubic',Ns);
    #   for i = nLast:-1:1
    #     OUT(i) = alpha*(IN(i)+Vel(i)/Ns) + (1-alpha)*OUT(i+1);
    #   end
