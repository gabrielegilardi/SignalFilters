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


def synthetic_wave(per, amp=None, pha=None, tim=None):
    """
    Generates a multi-sinewave.
    P = [ P1 P2 ... Pn ]      Periods
    varargin = A, T, PH       Amplitudes, time, phases
    A = [ A1 A2 ... An ]      Amplitudes
    T = [ ts tf dt]           Time info: from ts to tf in dt steps
    PH = [PH1 PH2 ... PHn]    Phases (rad)

    Av = SUM(1 to n) of [ A*sin(2*pi*f*t + PH) ]
    Tv   Time (ts to tf with step dt)
    
    Default amplitudes are ones
    Default time is from 0 to largest period (1000 steps)
    Default phases are zeros
    """
    n_waves = len(per)
    per = np.asarray(per)

    # Check for amplitudes, times, and phases
    if (amp is None):
        amp = np.ones(n_waves)
    else:
        amp = np.asarray(amp)
    if (tim is None):
        t_start = 0.0
        t_end = np.amax(per)
        n_steps = 500
    else:
        t_start = tim[0]
        t_end = tim[1]
        n_steps = int(tim[2])
    if (pha is None):
        pha = np.zeros(n_waves)
    else:
        pha = np.asarray(pha)

    # Add all the waves
    t = np.linspace(t_start, t_end, num=n_steps)
    f = np.zeros(len(t))
    for i in range(n_waves):
        f = f + amp[i] * np.sin(2.0 * np.pi * t / per[i] + pha[i])

    return t, f


# function sP = SyntQT(P,type)
# %  Function: generate a random (synthetic) price curve
# %
# %  Inputs
# %  ------
# %  P        prices (for type equal to 'P' and 'R')
# %           normal distribution data (for type equal to 'N')
# %             P(1) = mean
# %             P(2) = std
# %             P(3) = length
# %  type     type of generation
# %             P    use returns from price 
# %             R    use returns normal distribution
# %             N    use specified normal distribution
# %
# %  Output
# %  ------
# %  sP       generated synthetic prices

#   % Check for number of arguments
#   if (nargin ~= 2)
#     fprintf(1,'\n');
#     error('Wrong number of arguments');
#   end

#   switch(type)
  
#     % Use actual returns from P to generate values 
#     case 'P'
#       R = Price2ret(P,'S');       % "simple" method
#       sR = phaseran(R,1);
    
#     % Use normal distribution to generate values
#     % (mean and std are from the actual returns of P)
#     case 'R'
#       R = Price2ret(P,'S');       % "simple" method
#       sR = normrnd(mean(R),std(R),length(R),1);
    
#     % Use defined normal distribution to generate values
#     % P(1)=mean, P(2)=std, P(3)=length
#     case 'N'
#       sR = normrnd(P(1),P(2),P(3),1);

#     otherwise
#       fprintf(1,'\n');
#       error('Type not recognized');
  
#   end
  
#   % Use 'simple' method and P0 = 1 to determine price
#   sP = Ret2price(sR,'S');  
  
# end      % End of function


# % Input data
# % ----------
# % recblk: is a 2D array. Row: time sample. Column: recording.
# % An odd number of time samples (height) is expected. If that is not
# % the case, recblock is reduced by 1 sample before the surrogate
# % data is created.
# % The class must be double and it must be nonsparse.

# % nsurr: is the number of image block surrogates that you want to 
# % generate.
 
# % Output data
# % ---------------------
# % surrblk: 3D multidimensional array image block with the surrogate
# % datasets along the third dimension

# % Example 1
# % ---------
# %   x = randn(31,4);
# %   x(:,4) = sum(x,2); % Create correlation in the data
# %   r1 = corrcoef(x) 
# %   surr = phaseran(x,10);
# %   r2 = corrcoef(surr(:,:,1)) % Check that the correlation is preserved

# %   Carlos Gias
# %   Date: 21/08/2011

# % Reference:
# % Prichard, D., Theiler, J. Generating Surrogate Data for Time Series
# % with Several Simultaneously Measured Variables (1994)
# % Physical Review Letters, Vol 73, Number 7

# function surrblk = phaseran(recblk,nsurr)

#   % Get parameters
#   [nfrms,nts] = size(recblk);
#   if ( rem(nfrms,2) == 0 )
#     nfrms = nfrms-1;
#     recblk = recblk(1:nfrms,:);
#   end
    
#   % Get parameters
#   len_ser = (nfrms-1)/2;
#   interv1 = 2:len_ser+1; 
#   interv2 = len_ser+2:nfrms;

#   % Fourier transform of the original dataset
#   fft_recblk = fft(recblk);

#   % Create the surrogate recording blocks one by one
#   surrblk = zeros(nfrms,nts,nsurr);
#   for k = 1:nsurr
#     ph_rnd = rand([len_ser 1]);
   
#     % Create the random phases for all the time series
#     ph_interv1 = repmat(exp(2*pi*1i*ph_rnd),1,nts);
#     ph_interv2 = conj(flipud( ph_interv1));
   
#     % Randomize all the time series simultaneously
#     fft_recblk_surr = fft_recblk;
#     fft_recblk_surr(interv1,:) = fft_recblk(interv1,:).*ph_interv1;
#     fft_recblk_surr(interv2,:) = fft_recblk(interv2,:).*ph_interv2;
   
#     % Inverse transform
#     surrblk(:,:,k)= real(ifft(fft_recblk_surr));
#   end
  
# end     % End of function




