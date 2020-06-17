
# import numpy as np

# measurements = np.array([5., 6., 7., 9., 10.])
# motion = np.array([1., 1., 2., 1., 1.])
# measurement_sigma = 4.
# motion_sigma = 2.
# mu = 0.
# sigma = 1000.

# # Measurement
# def Update( mean1, var1, mean2, var2 ):
#     mean = (var2*mean1 + var1*mean2) / (var1 + var2)
#     var = 1.0 / (1.0/var1 + 1.0/var2)
#     return [mean, var]

# # Motion
# def Predict( mean1, var1, U, varU ):
#     mean = mean1 + U
#     var = var1 + varU
#     return [mean, var]

# for n in range(len(measurements)):
#     [mu, sigma] = Update(mu, sigma, measurements[n], measurement_sigma)
#     print('Update : ', n, [mu, sigma])
#     [mu, sigma] = Predict(mu, sigma, motion[n],motion_sigma)
#     print('Predict: ', n, [mu, sigma])

# print(' ')
# print(Update(1,1,3,1))

# -------------------------------------------------------

# import numpy as np

# measurements = [ 1., 2., 3. ]
# dt = 1.

# # Initial state (location and velocity)
# x = np.array([[ 0. ],
#               [ 0. ]])
# # Initial uncertainty
# P = np.array([[ 1000.,    0. ],
#               [    0., 1000. ]])
# # External motion
# U = np.array([[ 0. ],
#               [ 0. ]]) 
# # Next state function
# F = np.array([[ 1., dt ], 
#               [ 0., 1. ]])
# # Measurement function
# H = np.array([[ 1., 0. ]])
# # Measurement uncertainty
# R = np.array([[ 1. ]])
# # Identity matrix
# I = np.eye(2)


# def filter(x, P):

#     step = 0
#     for z in (measurements):
#         step += 1
#         print("step = ", step, "  meas. = ", z)
        
#         # Measurement
#         Htra = H.T
#         S = H.dot(P.dot(Htra)) + R
#         Sinv = np.linalg.inv(S)
#         K = P.dot(Htra.dot(Sinv))
#         y = z - H.dot(x)
#         xp = x +K.dot(y)
#         Pp = P - K.dot(H.dot(P))

#         # Prediction
#         x = F.dot(xp) + U
#         Ftra = F.T
#         P = F.dot(Pp.dot(Ftra))

#         print('x =')
#         print(x)
#         print('P =')
#         print(P)

# filter(x, P)

# # -------------------------------------------------------

import numpy as np

x0 = 4.
y0 = 12.
measurements = np.array([[  5., 10. ],
                         [  6.,  8. ],
                         [  7.,  6. ],
                         [  8.,  4. ],
                         [  9.,  2. ],
                         [ 10.,  0. ]])
# x0 = -4.
# y0 = 8.
# measurements = np.array([[  1.,  4. ],
#                          [  6.,  0. ],
#                          [ 11., -4. ],
#                          [ 16., -8. ]])
# x0 = 1.
# y0 = 19.
# measurements = np.array([[  1., 17. ],
#                          [  1., 15. ],
#                          [  1., 13. ],
#                          [  1., 11. ]])
# x0 = 1.
# y0 = 19.
# measurements = np.array([[  2., 17. ],
#                          [  0., 15. ],
#                          [  2., 13. ],
#                          [  0., 11. ]])
# Time step                         
dt = 0.1
# Initial state (location and velocity)
x = np.array([[ x0 ],
              [ y0 ],
              [  0. ],
              [  0. ]])
# Initial uncertainty
P = np.array([[ 0., 0.,    0.,    0. ],
              [ 0., 0.,    0.,    0. ],
              [ 0., 0., 1000.,    0. ],
              [ 0., 0.,    0., 1000. ]])
# External motion
U = np.array([[ 0. ],
              [ 0. ],
              [ 0. ], 
              [ 0. ]]) 
# Next state function
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
# Measurement vector
z = np.zeros((2,1))


def filter(x, P):
   
   for n in range(len(measurements)):

      z[0][0] = measurements[n][0]
      z[1][0] = measurements[n][1]

      # Prediction
      xp = F.dot(x) + U
      Ftra = F.T
      Pp = F.dot(P.dot(Ftra))

      # Measurement
      Htra = H.T
      S = H.dot(Pp.dot(Htra)) + R
      Sinv = np.linalg.inv(S)
      K = Pp.dot(Htra.dot(Sinv))
      y = z - H.dot(xp)
      x = xp +K.dot(y)
      P = Pp - K.dot(H.dot(Pp))
      # print(z)
      # print('x = ')
      # print(x)
      # print('P = ')
      # print(P)
      # print(' ')

   return x, P


x_final, P_final = filter(x, P)
print('x = ')
print(x_final)
print('P = ')
print(P_final)
