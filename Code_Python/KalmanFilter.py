"""
- generalize to N and M

M = 1
N = 2
"""
import numpy as np

dt = 1.

measurements = np.array([ 1., 2., 3., 4., 5., 6., 7., 8, 9, 10])

# State vector
# (N, 1)
x = np.array([[ 0. ],
              [ 0. ]])

# Prediction uncertainty (covariance matrix of x)
# (N, N)
P = np.array([[ 1000.,    0. ],
              [    0., 1000. ]])

# External motion
# (N, 1)
U = np.array([[ 0. ],
              [ 0. ]]) 

# Update matrix (state transition matrix)
# (N, N)
F = np.array([[ 1., dt ], 
              [ 0., 1. ]])

# Measurement function (extraction matrix)
# (M, N)
H = np.array([[ 1., 0. ]])

# Measurement uncertainty/noise (covariance matrix of z)
# (M, M)
R = np.array([[ 1. ]])

# z = measurament vector
# (M, 1)

def filter(x, P):

    step = 0
    for z in (measurements):
        step += 1
        print("step = ", step, "  meas. = ", z)
        
        # Measurement
        S = H @ P @ H.T + R                 # (M, M)
        K = P @ H.T @ np.linalg.inv(S)      # (N, M)
        y = z - H @ x
        xp = x + K @ y
        Pp = P - K @ H @ P

        # Prediction
        x = F @ xp + U
        P = F @ Pp @ F.T

        print('x =')
        print(x)
        print('P =')
        print(P)

filter(x, P)
