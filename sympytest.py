from sympy import symbols, Matrix, eye, simplify, pprint

# Define symbols for state variables, input, measurement, noise covariances
x1, x2 = symbols('x1 x2')  # State vector components
u = symbols('u')           # Control input
z = symbols('z')           # Measurement
x_pred1, x_pred2 = symbols('x_pred1 x_pred2')  # Predicted states

# State vector (x) and predicted state vector (x_pred)
x = Matrix([x1, x2])
x_pred = Matrix([x_pred1, x_pred2])

# State-space matrices (symbolic)
A = Matrix([[1, 1], [0, 1]])  # Example state transition matrix
B = Matrix([[0.5], [1]])      # Control input matrix
C = Matrix([[1, 0]])          # Measurement matrix
Q = eye(2) * 0.1              # Process noise covariance (symbolic)
R = eye(1) * 1                # Measurement noise covariance (symbolic)

# Initial state estimate and covariance
x_est = Matrix([0, 0])        # Initial state estimate
P = eye(2)                    # Initial error covariance

# Prediction equations
x_pred = A * x_est + B * u
P_pred = A * P * A.T + Q

# Display results
print("Predicted State (x_pred):")
pprint(x_pred)
print("\nPredicted Covariance (P_pred):")
pprint(P_pred)

# Compute Kalman gain
S = C * P_pred * C.T + R  # Innovation covariance
K = P_pred * C.T * S.inv()  # Kalman gain

# Update state estimate
y = z - C * x_pred  # Measurement residual (innovation)
x_est = x_pred + K * y

# Update error covariance
P = (eye(P_pred.shape[0]) - K * C) * P_pred

# Display results
print("\nKalman Gain (K):")
pprint(K)
print("\nUpdated State Estimate (x_est):")
pprint(x_est)
print("\nUpdated Covariance (P):")
pprint(P)


# Symbolic iteration for one measurement
u_value = 1  # Example input value
z_value = 2  # Example measurement value

# Substitute values into prediction step
x_pred_eval = x_pred.subs(u, u_value)
P_pred_eval = P_pred

# Substitute values into update step
y_eval = z_value - C * x_pred_eval
K_eval = simplify(P_pred_eval * C.T * (C * P_pred_eval * C.T + R).inv())
x_est_eval = simplify(x_pred_eval + K_eval * y_eval)
P_eval = simplify((eye(P_pred_eval.shape[0]) - K_eval * C) * P_pred_eval)

print("\nPredicted State with Substituted Values (x_pred):")
pprint(x_pred_eval)
print("\nUpdated State Estimate with Substituted Values (x_est):")
pprint(x_est_eval)
print("\nUpdated Covariance with Substituted Values (P):")
pprint(P_eval)