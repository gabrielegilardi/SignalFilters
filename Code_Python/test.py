"""
Signal Filtering/Smoothing and Generation of Synthetic Time-Series.

Copyright (c) 2020 Gabriele Gilardi


ToDo:
- add comments to the code
- in comments write what filters do
- is necessary to copy X for Y untouched?
- decide default values in functions
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

import filters as flt
import synthetic as syn

# Added to avoid message warning "The group delay is singular at frequencies
# [....], setting to 0" when plotting the lag for SMA filters. 
np.seterr(all='ignore')
np.random.seed(1294404794)

# Read data to filter
if len(sys.argv) != 2:
    print("Usage: python test.py <data_file>")
    sys.exit(1)
data_file = sys.argv[1] + '.csv'

# Read data from a csv file (one time-series each column)
data = np.loadtxt(data_file, delimiter=',')

t, f = syn.synthetic_wave([1., 2., 3.], A=None, phi=None, num=1000)
plt.plot(t,f)
plt.show()

# spx = flt.Filter(data)
# res = spx.EMA(N=10)
# signals = [spx.data, res[0:400]]
# flt.plot_signals(signals, start=100)

# spx.plot_frequency()
# spx.plot_lag()

# sigma_x = 0.1
# sigma_v = 0.1 * np.ones(n_samples)
# res = spx.Kalman(sigma_x=sigma_x, sigma_v=sigma_v, dt=1.0, abg_type="abg")
# alpha = 0.5
# beta = 0.005
# gamma = 0.0
# Yc, Yp = spx.ABG(alpha=alpha, beta=beta, gamma=gamma, dt=1.0)
# signals = (spx.data[:, 0], Yc[:, 0], Yp[:, 0])
# utl.plot_signals(signals, 0, 50)

# t, f = syn.synthetic_wave([1., 2., 3.], A=None, phi=None, num=100)
# plt.plot(t,f)
# plt.show()
# aa = np.array([
#         [ 0.8252,  0.2820],
#         [ 1.3790,  0.0335],
#         [-1.0582, -1.3337],
#         [-0.4686,  1.1275],
#         [-0.2725,  0.3502],
#         [ 1.0984, -0.2991],
#         [-0.2779,  0.0229],
#         [ 0.7015, -0.2620],
#         [-2.0518, -1.7502],
#         [-0.3538, -0.2857],
#         [-0.8236, -0.8314],
#         [-1.5771, -0.9792],
#         [ 0.5080, -1.1564]])
# synt_data = syn.synthetic_FFT(aa[0:5,0], n_reps=1)
# print(synt_data)

# plt.plot(synt_data1)
# plt.plot(synt_data2)
# plt.plot(data)
# names = ['syn1', 'syn2', 'spx']
# plt.legend(names)
# plt.show()
# percent = False
# print(data[0:10, :])
# bb = syn.value2diff(data, percent)
# print(bb[0:10, :])
# cc = syn.diff2value(bb, percent)
# print(cc[0:10, :]+1399.48)

# i = np.arange(aa.shape[1])
# bb[:, i] = aa[idx[:, i], i]
# aa = np.array([4, 12, 36, 20, 8])
# bb = syn.synthetic_sampling(aa, n_reps=2, replace=True)
# print(bb)
# aa = np.array([4, 12, 36, 20, 8])
# W = syn.synthetic_MEboot(aa, n_reps=1, alpha=0.1, bounds=False, scale=False)
# print('W=')
# print(W)

