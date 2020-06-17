"""

correction = update = measurement
prediction = motion

X       (n_states, 1)           State vector
P       (n_states, n_states)    Covariance matrix of X
F       (n_states, n_states)    State transition matrix
U       (n_states, 1)           Input/control/drift vector
Z       (n_meas, 1)             Measurament vector
H       (n_meas, n_states)      Measurament matrix
R       (n_meas, n_meas)        Covariance matrix of Z
S       (n_meas, n_meas)        Covariance matrix (?)
K       (n_states, m)           Kalman gain
Q       (n_states, n_states)    Covariance matrix (?)

Data    (n_meas, n_samples)     Measurements
Fext    (n_states, n_samples)   External driver
X0      (n_states, 1)           Initial state vector
P0      (n_states, n_states)    Initial covariance matrix of X
"""

import numpy as np


class KalmanFilter:

    def __init__(self, F, H, Q, R):
        """
        """
        self.F = F
        self.Q = Q
        self.H = H
        self.R = R


    def prediction(self, X, P, U):
        X = self.F @ X + U
        P = self.F @ P @ self.F.T + self.Q
        return X, P


    def update(self, X, P, Z):
        """
        """
        S = self.H @ P @ self.H.T + self.R
        K = P @ self.H.T @ np.linalg.inv(S)
        X = X + K @ (Z - self.H @ X)
        P = P - K @ self.H @ P
        return X, P

    
    def applyFilter(self, Data, Fext, X0, P0):
        """
        """
        pass


# Measurements
data = np.array([[5., 6., 7., 8., 9., 10.],
                 [10., 8., 6., 4., 2., 0.]])

# Initial state vector
X0 = np.array([[4. ],
               [12.],
               [0. ],
               [0. ]])

# Initial covariance matrix of X
P0 = np.array([[0., 0.,    0.,    0.],
               [0., 0.,    0.,    0.],
               [0., 0., 1000.,    0.],
               [0., 0.,    0., 1000.]])

# External motion
Fext = np.zeros_like(data)

# Next state function
dt = 0.1
F = np.array([[ 1., 0., dt, 0. ], 
              [ 0., 1., 0., dt ],
              [ 0., 0., 1., 0. ],
              [ 0., 0., 0., 1. ]])
# Measurement function
H = np.array([[ 1., 0., 0., 0. ],
              [ 0., 1., 0., 0. ]])
# Measurement uncertainty
R = np.array([[ 0.1, 0.  ],
              [ 0. , 0.1 ]])

def filter(x, P):

    step = 0
    for z in (measurements):
        step += 1
        print("step = ", step, "  meas. = ", z)
        
        # Update


        print('x =')
        print(x)
        print('P =')
        print(P)

filter(x, P)
