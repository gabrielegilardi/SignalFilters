"""
Signal Filtering and Generation of Synthetic Time-Series.

Copyright (c) 2020 Gabriele Gilardi


References
----------

- John F. Ehlers, "Cycle Analytics for Traders: Advanced Technical Trading
  Concepts", @ http://www.mesasoftware.com/ehlers_books.htm.

- D. Prichard and J. Theiler, "Generating surrogate data for time series with
  several simultaneously measured variables",
  @ https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.73.951.

- H. Vinod and J. Lopez-de-Lacalle, "Maximum entropy bootstrap for time series:
  the meboot R package, @ https://www.jstatsoft.org/article/view/v029i05.

Characteristics
---------------
- The code has been written and tested in Python 3.7.7.
- Implementation of several digital signal filters and functions for the
  generation of synthetic (surrogate) time-series.
- Filters (file <filters.py>):
    Generic             Generic filter
    SMA                 Simple moving average
    EMA                 Exponential moving average
    WMA                 Weighted moving average
    MSMA                Modified simple moving average
    MLSQ                Modified least-squares quadratic
    ButterOrig          Butterworth original filter
    ButterMod           Butterworth modified filter
    SuperSmooth         Supersmoother filter
    GaussLow            Gauss low pass filter
    GaussHigh           Gauss high pass filter
    BandPass            Band-pass filter
    BandStop            Band-stop filter
    ZEMA1               Zero-lag EMA (type 1)
    ZEMA2               Zero-lag EMA (type 2)
    InstTrend           Instantaneous trendline
    SincFilter          Sinc function filter
    Decycler            De-cycler filter
    DecyclerOsc         De-cycle oscillator
    ABG                 Alpha-beta-gamma filter
    Kalman              One-dimensional steady-state Kalman filter
- Synthetic time-series (file <synthetic.py>):
    synthetic_wave          Generates multi-sine wave given periods,
                            amplitudes, and phases.
    synthetic_sampling      Generates surrogates using randomized-sampling
                            (bootstrap) with or without replacement.
    synthetic_FFT           Generates surrogates using the phase-randomized
                            Fourier transform algorithm.
    synthetic_MEboot        Generates surrogates using the maximum entropy
                            bootstrap algorithm.
- File <filters.py> includes also functions to plot the filter signal,
  frequency response, and group delay.
- File <synthetic.py> includes also functions to differentiate, integrate,
  normalize, and scale the discrete time-series.
- Usage: python test.py <example>.

Parameters
----------
example
    Name of the example to run (Filters, Kalman, FFT_boot, ME_boot, Response).
data_file
    File with the dataset (csv format). The extension is added automatically.
X
    Dataset to filter/time-series (input). It must be a 1D array, i.e. of shape
    (:, ) or (:, 1) or (1, :).
b
    Transfer response coefficients (numerator).
a
    Transfer response coefficients (denominator).
Y
    Filtered dataset (output).
X_synt
    Surrogate/synthetic generated time-series (output).
n_reps
    Number of surrogates/synthetic time-series to generate.

Examples
--------
There are five examples (all of them use the dataset in *spx.csv*). The
results are shown in file <Results_Examples.pdf>.

- Filter: example showing filtering using an EMA, a Butterworth modified
  filter, and a type 2 Zero-lag EMA.

- Kalman: example showing filtering using the three types of Kalman filter
  (alpha, alpha-beta, and alpha-beta-gamma).

- FFT_boot: example showing the generation of surrogates time-series using
  the Fourier-transform algorithm and discrete differences.

- ME_boot: example showing the generation of surrogates time-series using the
  maximum entropy bootstrap algorithm and discrete differences.

- Response: example showing the frequency response and lag/group delay for a
  band-pass filter.
"""

import sys
import warnings
import numpy as np

import filters as flt
import synthetic as syn

# Added to avoid message warning "The group delay is singular at frequencies
# [....], setting to 0" when plotting the lag for SMA filters.
warnings.filterwarnings('ignore')
np.random.seed(1294404794)

# Read example to run
if len(sys.argv) != 2:
    print("Usage: python test.py <example>")
    sys.exit(1)
example = sys.argv[1]

# Read data from a csv file
data_file = 'spx'
data = np.loadtxt(data_file + '.csv', delimiter=',')

# Example with an EMA, a Butterworth modified filter, and a type 2 Zero-lag EMA
if (example == 'Filters'):
    spx = flt.Filter(data)
    ema = spx.EMA(N=10)
    butter = spx.ButterMod(P=10, N=2)
    zema = spx.ZEMA2(N=10, K=2.0)
    signals = [spx.data, ema, butter, zema]
    names = ['SPX', 'EMA', 'ButterMod', 'ZEMA2']
    flt.plot_signals(signals, names=names, start=0, end=200)

# Example with the three types of Kalman filter
elif (example == 'Kalman'):
    spx = flt.Filter(data)
    a_type = spx.Kalman(sigma_x=0.1, sigma_v=0.1, dt=1.0, abg_type="a")
    ab_type = spx.Kalman(sigma_x=0.1, sigma_v=0.1, dt=1.0, abg_type="ab")
    abg_type = spx.Kalman(sigma_x=0.1, sigma_v=0.1, dt=1.0, abg_type="abg")
    signals = [spx.data, a_type, ab_type, abg_type]
    names = ['SPX', 'a-type', 'ab-type', 'abg-type']
    flt.plot_signals(signals, names=names, start=0, end=200)

# Example of surrogates time-series using the Fourier-transform algorithm
elif (example == 'FFT_boot'):
    n_reps = 4
    n_samples = len(data)

    # Generated the surrogates using the 1st discrete differences (in percent)
    dX = syn.value2diff(data, percent=True)
    dX_synt = syn.synthetic_FFT(dX, n_reps=n_reps)      # (n_reps, n_samples)

    # Rebuild the actual surrogate time-series
    signals = [data]
    names = ['SPX']
    for i in range(n_reps):
        X_synt = syn.diff2value(dX_synt[i, :], data[0], percent=True)
        signals.append(X_synt)
        names.append('Synth. #' + str(i+1))

    # Plot all
    flt.plot_signals(signals, names=names, start=0, end=200)

# Example of surrogates time-series using the maximum entropy bootstrap algorithm
elif (example == 'ME_boot'):
    n_reps = 4
    n_samples = len(data)

    # Generated the surrogates using the 1st discrete differences (in value)
    dX = syn.value2diff(data, percent=False)
    dX_synt = syn.synthetic_MEboot(dX, n_reps=n_reps, alpha=0.1, bounds=False,
                                   scale=False)         # (n_reps, n_samples)

    # Rebuild the actual surrogate time-series
    signals = [data]
    names = ['SPX']
    for i in range(n_reps):
        X_synt = syn.diff2value(dX_synt[i, :], data[0], percent=False)
        signals.append(X_synt)
        names.append('Synth. #' + str(i+1))

    # Plot all
    flt.plot_signals(signals, names=names, start=0, end=499)

# Example of frequency response and lag/group delay for a band-pass filter
elif (example == 'Response'):
    spx = flt.Filter(data)
    band = spx.BandPass(P=10, delta=0.3)
    spx.plot_response()

else:
    print("Example not found")
    sys.exit(1)
