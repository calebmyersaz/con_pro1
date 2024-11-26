import numpy as np
# import scipy
from scipy.linalg import solve_continuous_are

# Define matrices
Af = np.array([[0, 1], [-2, -3]])
Bf = np.array([[0], [1]])
Cf = np.array([[1, 0]])
Nm = 1
Ne = 0.1

# Define equivalent Q and R
Q = Cf.T @ np.linalg.inv(Nm) @ Cf
R = Ne

# Solve Riccati Equation
M = solve_continuous_are(Af, Bf, Q, R)

# Print result
print("Solution M:")
print(M)