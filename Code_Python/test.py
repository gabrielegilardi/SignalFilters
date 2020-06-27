"""
Signal Filtering and Generation of Synthetic Time-Series.

Copyright (c) 2020 Gabriele Gilardi

"""

import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt

import filters as flt
import synthetic as syn

# Added to avoid message warning "The group delay is singular at frequencies
# [....], setting to 0" when plotting the lag for SMA filters. 
# np.seterr(all='ignore')
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

# Example with a few filters
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

# Example of filter frequency and lag
elif (example == 'Response'):
    spx = flt.Filter(data)
    band = spx.BandPass(P=10, delta=0.3)
    spx.plot_response()

# Example of surrogates time-series using the Fourier-transform algorithm
elif (example == 'FFT_boot'):
    pass

# Example of surrogates time-series using maximum entropy bootstrap algorithm
elif (example == 'ME_boot'):
    pass

else:
    print("Example not found")
    sys.exit(1)
    

# dX = syn.value2diff(data, percent=True)
# # X = syn.diff2value(dX, data[0], percent=True)
# signals = [data, X]
# flt.plot_signals(signals, start=0, end=200)

# dX = syn.value2diff(data, percent=False)
# dX_synt = syn.synthetic_sampling(dX, n_reps=1, replace=True)
# dX_synt = syn.synthetic_FFT(dX, n_reps=2)
# dX_synt = syn.synthetic_MEboot(dX, n_reps=4, alpha=0.1, bounds=False, scale=False)
# for i in range(dX_synt.shape[0]):
#     # aa = dX_synt[:, i]
#     aa = syn.diff2value(dX_synt[i, :], data[0], percent=False)
#     plt.plot(aa)
# plt.xlim(0, 200)
# plt.plot(data)
# plt.show()
