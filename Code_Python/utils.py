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
    plt.xlabel('$\omega$ [rad/sample]')
    plt.ylabel('$h$ [db]')
    plt.title('b = ' + np.array_str(np.around(b, decimals=2)) \
               + ',   a = ' + np.array_str(np.around(a, decimals=2)))
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
    plt.title('b = ' + np.array_str(np.around(b, decimals=2)) \
               + ',   a = ' + np.array_str(np.around(a, decimals=2)))
    plt.show()



# function [y,t] = interpSignal(x,n1,n2,varargin)
#   if (nargin == 4)
#     dt = varargin{1};
#   else
#     dt = 20;
#   end
#   N = length(x);
#   Ts = (n2-n1)/(N-1);
#   n = n1:Ts:n2;
#   t = n1:Ts/dt:n2;
#   nt = length(t);
#   y = zeros(1,nt);
#   for i = 1:nt
#     a = x.*sinc( (t(i)-n)/Ts );
#     y(i) = sum(a);
#   end
# end


# function [f] = plotFn(type,func,varargin)
#   % 0     Plot real function 
#   % 1     Plot real/imag values
#   % 2     Plot real/imag values in polar form
#   % 3     Plot magnitude/phase

#   clf
#   tiny = 1e-7;
  
#   % Check if func is a function or data
#   if ( isa(func,"function_handle") )
#     N = varargin{1};
#     n = (0:N-1)';    
#     f = func(n);
#   else
#     N = length(func);
#     n = (0:N-1)';    
#     f = func;
#   end

#   % Clean data
#   xRe = real(f);  
#   xIm = imag(f);  
#   xRe( abs(xRe) < tiny ) = 0;
#   xIm( abs(xIm) < tiny ) = 0;

#   switch (type)

#     % Plot real function  
#     case 0    
      
#       stem(n,xRe,"b","filled")
#       xlim([n(1) n(N)])
#       grid on
#       xlabel("n")
#       ylabel("f")
#       box on

#     % Plot real/imag function
#     case 1
#       subplot(2,1,1)
#       stem(n,xRe,"b","filled")
#       xlim([n(1) n(N)])
#       grid on
#       xlabel("n")
#       ylabel("Re")
#       box on
#       subplot(2,1,2)
#       stem(n,xIm,"b","filled")
#       xlim([n(1) n(N)])
#       grid on
#       xlabel("n")
#       ylabel("Im")
#       box on

#     % Plot real/imag function in polar form
#     case 2
#       scatter(xRe,xIm,"b","filled")
#       maxRe = max( abs(xRe) );
#       maxIm = max( abs(xIm) );
#       m = max(maxRe,maxIm);
#       dx = 2*m/50;
#       text(xRe+dx,xIm,num2str(n))
#       xlim( [-m +m ])
#       ylim( [-m +m ])
#       axis("square")
#       grid on
#       hold on
#       plot([-m 0; +m 0],[0 -m; 0 +m],"k")
#       hold off
#       xlabel("Real")
#       ylabel("Imag")
#       box on
      
#     % Plot magnitude/phase 
#     case 3
#       xMa = sqrt( xRe.^2 + xIm.^2 );  
#       xAr = atan2(xIm,xRe);
#       subplot(2,1,1)
#       stem(n,xMa,"b","filled")
#       xlim([n(1) n(N)])
#       grid on
#       xlabel("n")
#       ylabel("Magnitude")
#       box on
#       subplot(2,1,2)
#       stem(n,xAr,"b","filled")
#       xlim([n(1) n(N)])
#       ylim([-pi pi])
#       grid on
#       xlabel("n")
#       ylabel("Phase [rad]")
#       box on
    
#   end
  
# end


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


# function [Tv,Av] = SyntWave(P,varargin)
# %  Function: generate a multi-sinewave
# %
# %  Inputs
# %  ------
# %  P = [ P1 P2 ... Pn ]      Periods
# %  varargin = A, T, PH       Amplitudes, time, phases
# %
# %  A = [ A1 A2 ... An ]      Amplitudes
# %  T = [ ts tf dt]           Time info: from ts to tf with step dt
# %  PH = [PH1 PH2 ... PHn]    Phases (rad)
# %
# %  Outputs
# %  -------
# %  Av = SUM(1 to n) of [ A*sin(2*pi*f*t + PH) ]
# %  Tv   Time (ts to tf with step dt)
# % 
# %  Default amplitudes are ones
# %  Default time is from 0 to largest period (1000 steps)
# %  Default phases are zeros

#   % Check for arguments
#   if (nargin == 1)
#     np = length(P);
#     A = ones(1,np);
#     T = [0 max(P) max(P)/1000];
#     PH = zeros(1,np);
#   elseif (nargin == 2)
#     np = length(P);
#     A = varargin{1};
#     T = [0 max(P) max(P)/1000];
#     PH = zeros(1,np);
#   elseif (nargin == 3)
#     np = length(P);
#     A = varargin{1};
#     T = varargin{2};
#     PH = zeros(1,np);
#   elseif (nargin == 4)
#     np = length(P);
#     A = varargin{1};
#     T = varargin{2};
#     PH = varargin{3};
#   else
#     fprintf(1,'\n');
#     error('Wrong number of arguments');
#   end    
 
#   % Add all waves
#   Tv = T(1):T(3):T(2);
#   Av = zeros(1,length(Tv));     
#   for j = 1:np
#     Av = Av + A(j)*sin(2*pi*Tv/P(j)+PH(j));
#   end
  
# end   % End of function


