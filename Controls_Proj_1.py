
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp

# System parameters
m1, m2 = 1.0, 2.0
l = 1.0
I1, I2 = 0.5, 0.5
g = 9.81

# State-space matrices
def state_matrices(theta2):
    theta2_safe = max(abs(theta2), 1e-6)
    J = np.array([
        [I1 + 0.25 * m1 * l**2 + m2 * l**2, 0.5 * m2 * l**2 * np.cos(theta2_safe)],
        [0.5 * m2 * l**2 * np.cos(theta2_safe), I2 + 0.25 * m2 * l**2]
    ])
    A = np.zeros((4, 4))
    A[0, 2], A[1, 3] = 1, 1
    B = np.zeros((4, 2))
    B[2:, :] = np.linalg.inv(J) @ np.array([
        [l * np.sin(theta2_safe / 2), -l * np.sin(theta2_safe / 2)],
        [0.5, 0.5]
    ])
    d = np.zeros((4, 1))
    d[2, 0] = -0.5 * l * theta2_safe**2 * np.cos(theta2_safe / 2)
    return A, B, d

# LQR control
def calc_lqr_gain(A, B, Q, R):
    P = solve_continuous_are(A, B, Q, R)
    return np.linalg.inv(R) @ B.T @ P

# Reference trajectory
def ref_trajectory(t):
    return np.array([0.2, 0.1*np.sin(2*np.pi*1.5*t), 0, 0])

# Control law
def control_law(t, x):
    x_des = ref_trajectory(t)
    return -K_f @ (x - x_des)

# System dynamics
def dynamics(t, x):
    A, B, d = state_matrices(theta2)
    v = control_law(t, x)
    return (A @ x + B @ v + d.flatten())

# Simulation parameters
theta2 = np.pi / 4
A, B, d = state_matrices(theta2)
Q = np.diag([100000, 100000, 10, 10])
R = np.diag([0.01, 0.01])
K_f = calc_lqr_gain(A, B, Q, R)

# Initial conditions
x0 = np.array([0.2, 0.0, 0.1, 0.0])
t_span = [0, 5]
t_eval = np.linspace(t_span[0], t_span[1], 2000)

# Solve dynamics
sol = solve_ivp(dynamics, t_span, x0, t_eval=t_eval)

# Plot results
time = sol.t
states = sol.y
reference = np.array([ref_trajectory(t) for t in time])

fig, axs = plt.subplots(2, 1, figsize=(8, 6))  # 2 rows, 1 column

# Plot data on the first subplot (top)
axs[0].plot(time, states[0, :], label="r (actual)")
axs[0].plot(time, reference[:, 0], '--', label="r (reference)")
axs[0].set_title("Position Tracking")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Position (r)")
axs[0].set_ylim(0.16,0.24)
axs[0].legend()

# Plot data on the second subplot (bottom)
axs[1].plot(time, states[1, :], label="r (actual)")
axs[1].plot(time, reference[:, 1], '--', label="r (reference)")
axs[1].set_title("Angle Tracking")
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("Angle (theta_r)")
axs[1].set_ylim(-0.12,0.12)
axs[1].legend()

# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()