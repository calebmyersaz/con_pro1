import numpy as np

# Initialize system matrices
A = np.array([[1, 1], [0, 1]])  # Example state transition matrix
B = np.array([[0.5], [1]])      # Control input matrix
C = np.array([[1, 0]])          # Measurement matrix
Q = np.eye(2) * 0.1             # Process noise covariance
R = np.eye(1) * 1               # Measurement noise covariance

# Initial state and covariance
x_est = np.array([[0], [0]])    # Initial state estimate
P = np.eye(2)                   # Initial error covariance

# Time steps and measurements
time_steps = 10
measurements = np.random.randn(time_steps, 1)  # Simulated noisy measurements
control_inputs = np.ones((time_steps, 1))      # Simulated control inputs

# Kalman filter loop
for k in range(time_steps):
    # Prediction step
    u = control_inputs[k]
    x_pred = A @ x_est + B @ u
    P_pred = A @ P @ A.T + Q

    # Update step
    z = measurements[k]
    S = C @ P_pred @ C.T + R  # Innovation covariance
    K = P_pred @ C.T @ np.linalg.inv(S)  # Kalman gain
    y = z - C @ x_pred  # Innovation
    x_est = x_pred + K @ y
    P = (np.eye(P_pred.shape[0]) - K @ C) @ P_pred

    # Print results
    print(f"Time step {k+1}:")
    print(f"State Estimate: \n{x_est}")
    print(f"Error Covariance: \n{P}\n")
