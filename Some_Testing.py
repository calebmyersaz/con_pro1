import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are

# Define constants
m1, m2 = 1.0, 1.0  # Masses of the links (kg)
l = 1.0  # Link length (m)
I1, I2 = 0.1, 0.1  # Moments of inertia (kg·m^2)
g = 9.81  # Gravity (m/s^2)

# State-space matrices
def state_matrices(theta2):
    # Adjust theta2 to avoid singularities
    theta2_safe = max(np.abs(theta2), 1e-6)

    # Inertia matrix J
    J = np.array([
        [I1 + 0.25 * m1 * l**2 + m2 * l**2, 0.5 * m2 * l**2 * np.cos(theta2_safe)],
        [0.5 * m2 * l**2 * np.cos(theta2_safe), I2 + 0.25 * m2 * l**2]
    ])

    # Check for singularity in J
    if np.linalg.cond(J) > 1 / np.finfo(float).eps:
        raise ValueError("Matrix J is nearly singular or ill-conditioned!")

    # State matrices A and B
    A = np.zeros((4, 4))
    A[2:, :2] = np.eye(2)

    B = np.zeros((4, 2))
    B[2:, :] = np.linalg.inv(J) @ np.array([
        [l * np.sin(theta2_safe / 2), -l * np.sin(theta2_safe / 2)],
        [0.5, 0.5]
    ])
    d = np.zeros((4, 1))
    d[2, 0] = -0.5 * l * (theta2**2) * np.cos(theta2 / 2)

    # return A, B
    return A, B, d

# LQR gain calculation
def compute_lqr_gain(A, B, Q, R):
    # Check controllability
    controllability_matrix = np.hstack([B, A @ B, A @ A @ B, A @ A @ A @ B])
    if np.linalg.matrix_rank(controllability_matrix) < A.shape[0]:
        raise ValueError("The system is not controllable!")

    # Solve the Algebraic Riccati Equation
    try:
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P
    except np.linalg.LinAlgError as e:
        print("Failed to solve Riccati equation. Ensure matrices are correct.")
        raise e
    return K

# Control law
def control_law(x, xd, K, A, B, d):
    u = -K @ (x - xd)
    return u

# Kalman filter
class KalmanFilter:
    def __init__(self, A, B, C, Q, R):
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R
        self.P = np.eye(A.shape[0])
        self.x_hat = np.zeros((A.shape[0], 1))

    def update(self, u, y):
        # Prediction
        x_hat_pred = self.A @ self.x_hat + self.B @ u
        P_pred = self.A @ self.P @ self.A.T + self.Q
        
        # Update
        K = P_pred @ self.C.T @ np.linalg.inv(self.C @ P_pred @ self.C.T + self.R)
        self.x_hat = x_hat_pred + K @ (y - self.C @ x_hat_pred)
        self.P = (np.eye(self.A.shape[0]) - K @ self.C) @ P_pred
        return self.x_hat

# Simulate the system
def simulate_system(initial_state, time, xd_func, K, A, B, d):
    def dynamics(t, x):
        theta2 = x[1]
        A_dyn, B_dyn, d_dyn = state_matrices(theta2)
        u = control_law(x, xd_func(t), K, A_dyn, B_dyn, d_dyn)
        dxdt = A_dyn @ x + B_dyn @ u + d_dyn.flatten()
        return dxdt

    sol = solve_ivp(dynamics, [time[0], time[-1]], initial_state, t_eval=time)
    return sol.t, sol.y.T

# Parameters for simulation
time = np.linspace(0, 10, 1000)
initial_state = np.array([0.2, 0.0, 0.0, 0.0])  # Initial [r, theta_r, r_dot, theta_r_dot]
xd_func = lambda t: np.array([0.2 + 0.015 * np.sin(2 * np.pi * 0.5 * t), 0.0, 0.0, 0.0])  # Desired trajectory
Q = np.eye(4)  # State weighting matrix for LQR
R = np.eye(2)  # Control effort weighting matrix for LQR

# Simulation
A, B, d = state_matrices(0.5)
K = compute_lqr_gain(A, B, Q, R)
t, x = simulate_system(initial_state, time, xd_func, K, A, B, d)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(t, x[:, 0], label="r (actual)")
plt.plot(t, [xd_func(ti)[0] for ti in t], '--', label="r (desired)")
plt.plot(t, x[:, 1], label="theta_r (actual)")
plt.plot(t, [xd_func(ti)[1] for ti in t], '--', label="theta_r (desired)")
plt.xlabel("Time (s)")
plt.ylabel("State")
plt.legend()
plt.grid()
plt.title("Tracking Control of Two-DoF Manipulator")
plt.show()