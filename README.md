# Signal Filtering and Generation of Synthetic Time-Series

## Reference

- John F. Ehlers, "[Cycle Analytics for Traders: Advanced Technical Trading Concepts](http://www.mesasoftware.com/ehlers_books.htm)".

- D. Prichard and J. Theiler, "[Generating surrogate data for time series with several simultaneously measured variables](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.73.951)".

- H. Vinod and J. Lopez-de-Lacalle, "[Maximum entropy bootstrap for time series: the meboot R package](https://www.jstatsoft.org/article/view/v029i05)".

## Characteristics

- The code has been written and tested in Python 3.7.7.
- Implementation of several digital signal filters and functions for the generation of synthetic (surrogate) time-series.
- Filters (*filters.py*):
  - **Generic** Generic filter.
  - **SMA** Simple moving average.
  - **EMA** Exponential moving average.
  - **WMA** Weighted moving average.
  - **MSMA** Modified simple moving average.
  - **MLSQ** Modified least-squares quadratic.
  - **ButterOrig** Butterworth original filter.
  - **ButterMod** Butterworth modified filter.
  - **SuperSmooth** Supersmoother filter.
  - **GaussLow** Gauss low pass filter.
  - **GaussHigh** Gauss high pass filter.
  - **BandPass** Band-pass filter.
  - **BandStop** Band-stop filter.
  - **ZEMA1** Zero-lag EMA (type 1).
  - **ZEMA2** Zero-lag EMA (type 2).
  - **InstTrend** Instantaneous trendline.
  - **SincFilter** Sinc function filter.
  - **Decycler** De-cycler filter.
  - **DecyclerOsc** De-cycle oscillator.
  - **ABG** Alpha-beta-gamma filter.
  - **Kalman** One-dimensional steady-state Kalman filter.
- Synthetic time-series (*synthetic.py*):
  - **synthetic_wave** Generates multi-sine wave given periods, amplitudes, and phases.
  - **synthetic_sampling** Generates surrogates using randomized-sampling (bootstrap) with or without replacement.
  - **synthetic_FFT** Generates surrogates using the phase-randomized Fourier-transform algorithm.
  - **synthetic_MEboot** Generates surrogates using the maximum entropy bootstrap algorithm.
- File *filters.py* includes also functions to plot the filter signal, frequency response, and group delay.
- File *synthetic.py* includes also functions to differentiate, integrate, normalize, and scale the discrete time-series.
- Usage: *python test.py example*.

## Main Parameters

`example` Name of the example to run (Filters, Kalman, FFT_boot, ME_boot, Response).

`data_file` File with the dataset (csv format). The extension is added automatically.

`X` Dataset to filter/time-series (input). It must be a 1D array, i.e. of shape `(:, )` or `(:, 1)` or `(1, :)`.

`b` Transfer response coefficients (numerator).

`a` Transfer response coefficients (denominator).

`Y` Filtered dataset (output).

`X_synt` Surrogate/synthetic generated time-series (output).

`n_reps` Number of surrogates/synthetic time-series to generate.

## Examples

There are five examples (all of them use the dataset in *spx.csv*). The results are shown [here](Result_Examples.pdf).

- **Filter** Example showing filtering using an EMA, a Butterworth modified filter, and a type 2 Zero-lag EMA.

- **Kalman** Example showing filtering using the three types of Kalman filter (alpha, alpha-beta, and alpha-beta-gamma).

- **FFT_boot** Example showing the generation of surrogates time-series using the Fourier-transform algorithm and discrete differences.

- **ME_boot** Example showing the generation of surrogates time-series using the maximum entropy bootstrap algorithm and discrete differences.

- **Response** Example showing the frequency response and lag/group delay for a band-pass filter.
