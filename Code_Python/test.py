"""
Filters for time series.

Copyright (c) 2020 Gabriele Gilardi

ToDo:
- in comments write what filters do
- is necessary to copy X for Y untouched?
- decide default values in functions
- check conditions on P and N
- why lag plot gives errors
- fix plotting function
- example for alpha-beta-gamma using variable sigma as in financial time series
  (see Ehler)
- example using noisy multi-sine-waves
"""

import sys
import numpy as np
import filters as flt
import utils as utl
import matplotlib.pyplot as plt

# Read data to filter
if len(sys.argv) != 2:
    print("Usage: python test.py <data_file>")
    sys.exit(1)
data_file = sys.argv[1] + '.csv'

# Read data from a csv file
data = np.loadtxt(data_file, delimiter=',')
n_samples = data.shape[0]
data = data.reshape(n_samples, -1)

spx = flt.Filter(data)

# args = (spx.X, spx.SMA(N=5), spx.EMA(alpha=0.7))
# utl.plot_signals(args)

# alpha = 0.8
# bb = np.array([alpha])
# aa = np.array([1.0, alpha - 1.0])


# res, bb, aa = spx.SincFunction(2, 50)
# print(bb)
# print(aa)
# utl.plot_frequency_response(bb, aa)
# utl.plot_lag_response(bb, aa)
# sigma_x = 0.1
# sigma_v = 0.1 * np.ones(n_samples)
# res = spx.Kalman(sigma_x=sigma_x, sigma_v=sigma_v, dt=1.0, abg_type="abg")
alpha = 0.5
beta = 0.005
gamma = 0.0
Yc, Yp = spx.ABG(alpha=alpha, beta=beta, gamma=gamma, dt=1.0)
signals = (spx.data[:, 0], Yc[:, 0], Yp[:, 0])
utl.plot_signals(signals, 0, 50)


