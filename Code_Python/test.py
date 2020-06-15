"""
Filters for time series.

Copyright (c) 2020 Gabriele Gilardi

ToDo:
- use NaN/input values for points not filtered?
- return idx?
- util to test filter (impulse, utils)
- warning in filter when wrong order? or save flag with true/false if computed
- use self.a and self.b
- remove a and b from plots
- in comments write what filters do
- is necessary to copy X for Y untouched?
- decide default values in functions
- check conditions on P and N
"""

import sys
import numpy as np
import filters as flt
import utils as utl

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


res, bb, aa = spx.SincFunction(2, 50)
print(bb)
print(aa)
utl.plot_frequency_response(bb, aa)
utl.plot_lag_response(bb, aa)

# res = spx.DecyclerOsc(30, 60)
# print(res[0:10, :])
signals = (spx.X, res)
print(spx.idx)
utl.plot_signals(signals)
# print(spx.X[0:20])