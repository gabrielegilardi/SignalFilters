"""
Filters for time series.

Copyright (c) 2020 Gabriele Gilardi



ToDo:
- use NaN/input values for points not filtered?
- what to use as previous Y in recursive filters? i.e. set Y=X or set Y=0?
- plot filtered data (utils, with generic number of arguments passed)
- plot filter characteristics (utils)
- return idx?
- util to test filter (impulse, utils)
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
EMA = spx.EMA()
print(EMA[0:5,:])
# args = (spx.X, spx.SMA(N=5), spx.EMA(alpha=0.7))
# utl.plot_signals(args)