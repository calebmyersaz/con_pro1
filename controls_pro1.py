import sympy as sp
import numpy as np

def Calc_M(A_f,B_f,C_f,N_m,N_e):
    np_A = np.array(A_f).astype(np.float64)
    np_B = np.array(B_f).astype(np.float64)
    np_C = np.array(C_f).astype(np.float64)
    np_Nm = np.array(N_m).astype(np.float64)
    np_Ne = np.array(N_e).astype(np.float64)

    def matrix_eq(M_flat):
        M = M_flat.reshape(4, 4)  # Convert the flat array back to a matrix
        term1 = np.dot(np_A, M) + np.dot(M, np_A.T)
        term2 = np.dot(M, np.dot(np_C.T, np.dot(np.linalg.inv(np_Nm), np.dot(np_C, M))))
        term3 = np.dot(np_B, np.dot(np_Ne, np_B.T))
        return (term1 + term2 - term3).flatten()  # Flatten the result for numerical solving

    # Initial guess for the matrix M (a 16-element vector)
    initial_guess = np.eye(4).flatten()

    # Solve using fsolve to minimize the residual of the equation
    from scipy.optimize import fsolve
    M_solution_flat = fsolve(matrix_eq, initial_guess)

    # Convert the solution back to a 4x4 matrix
    M_solution = M_solution_flat.reshape(4, 4)

    # print(M_solution)
    return M_solution



# Define sympy variable:
I_1, I_2, m_1, m_2, l, theta_1, theta_2, theta_12 = sp.symbols('I1 I2 m1 m2 l theta1 theta2 theta_12')
theta_1dot, theta_12dot, theta_2dot = sp.symbols('theta1dot theta12dot theta_2dot')

# define J, w, and R_f
# Equation 2
J = sp.Matrix([
    [I_1 + (1/4)*m_1*l**2+m_2*l**2, (1/2)*m_2*l**2*sp.cos(theta_2)],
    [(1/2)*m_2*l**2*sp.cos(theta_2), I_2 +(1/4)*m_2*l**2]

])
# Equation 3
w = sp.Matrix([
    [(1/2)*m_2**2*l**2*sp.sin(theta_2)*theta_12dot**2],
    [-(1/2)*m_2**2*l**2*sp.sin(theta_2)*theta_1dot**2]

])
# Equation 4
R_f = l*sp.Matrix([
    [-sp.sin(theta_2/2),sp.cos(theta_2/2)],
    [sp.sin(theta_2/2),sp.cos(theta_2/2)]

]) 
# State Variables
r, theta_r, rdot, theta_rdot = sp.symbols('r theta_r rdot theta_rdot')
x,u,d,n = sp.symbols('x u d n')
tau_m, tau_b = sp.symbols('tau_m tau_b')
x = sp.Matrix([[r],[theta_r],[rdot],[theta_rdot]])
# sp.pprint(x)
u = sp.Matrix([[tau_m],[tau_b]])

# Equation 7
# State Matrices
B = sp.Matrix([[0, 0],
               [0, 0],
               [l*sp.sin(theta_2/2),-l*sp.sin(theta_2/2)],
               [1/2, 1/2]])
B = B*J.inv()
# Equation 8
d = sp.Matrix([[0],
               [0],
               [-(1/2)*l*theta_2dot**2*sp.cos(theta_2/2)],
               [0]])
# Equation 6
A = sp.Matrix([[0,0,1,0],
               [0,0,0,1],
               [0,0,0,0],
               [0,0,0,0]])

v,v_1,v_2 = sp.symbols('v v_1 v_2')
v = sp.Matrix([[v_1],[v_2]])



# Equation 9
ctrlp1 = J * sp.Matrix([[1/(2*l*sp.sin(theta_2/2)),1],
                        [-1/(2*l*sp.sin(theta_2/2)),1]]) * v
ctrlp2 = (sp.cot(theta_2/2)/4)*J*sp.Matrix([[-1],[1]])*(theta_2dot**2)

u_ctrl = ctrlp1 - w -ctrlp2




# sp.pprint(sp.simplify(u_ctrl)) 

# Establish symbols
a_x, a_y, = sp.symbols('a_x a_y')
theta_1ddot, theta_12ddot = sp.symbols('theta_1ddot theta_12ddot')
l_20 = sp.symbols('l_20')
# Equation 13
K1 = sp.Matrix([[l*sp.sin(theta_2), 0],
                [l*sp.cos(theta_2), l_20]])
K2 = sp.Matrix([[-l*sp.cos(theta_2), -l_20],
                [l*sp.sin(theta_2), 0]])
# Equation 14
accel = sp.Matrix([[theta_1ddot],
                   [theta_12ddot]])
accel = K1.inv()*(sp.Matrix([[a_x],[a_y]])-K2*sp.Matrix([[theta_1dot**2],[theta_12dot**2]])) 
# print('Equation 14')
# sp.pprint(accel)



Kal_th,Kal_thdot,Kth_e,Kth_edot, n_e, n_m,Kal_y = sp.symbols('Kal_th Kal_thdot Kth_e Kth_edot n_e n_m y_Kal')

Kal_th = sp.Matrix([[theta_1],[theta_12],[theta_1dot],[theta_12dot]])
# print('Kalman Theta Def')
# sp.pprint(Kal_th)
n = sp.Matrix([[1],[1]])

# n_e definition
n_e = -K1.inv()*n
# sp.pprint(n_e)




print('n_e def')
# sp.pprint(n_e)

A_f = sp.Matrix([[0,0,1,0],
                 [0,0,0,1],
                 [0,0,0,0],
                 [0,0,0,0]])

B_f = sp.Matrix([[0, 0],
                 [0, 0],
                 [0, 1],
                 [1, 0]])

C_f = sp.Matrix([[0,1,0,0],
               
                [1,0,0,0]])

# Equation 15
Kal_thdot = sp.Matrix([[0,0,1,0],
                       [0,0,0,1],
                       [0,0,0,0],
                       [0,0,0,0]]) * Kal_th + sp.Matrix([[0,0],
                                                         [0,0],
                                                         [0,1],
                                                         [1,0]])*(accel+n_e)
# sp.pprint(Kal_thdot)

n_m = sp.Matrix([[1],[-1]])

# Equation 16
Kal_y = sp.Matrix([[1, 0, 0, 0],
                   [0, 1, 0, 0]])*Kal_th + n_m

th12_e, th1_e, th2_e = sp.symbols('th12e th1e th2e')
# th12_edot, th1_edot, th2_edot = sp.symbols('th12edot th1edot th2edot')

F,Kal_th_e,Kal_th_e_dot, a = sp.symbols('F Kal_th_e Kal_th_e_dot, a')
th12_edot, th1_edot, th2_edot = sp.symbols('th12edot th1edot th2edot')

Kal_th_e = sp.Matrix([[th1_e],
                      [th12_e],
                      [th1_edot],
                      [th12_edot]])
a = accel
F = sp.Matrix([[1,0],
               [0,1],
               [1,0],
               [0,1]])
# Kal_th_e_dot = A_f*Kal_th_e + B_f*a + F*(Kal_y - C_f*Kal_th_e)
# G = (Kal_y - C_f*Kal_th_e)

# sp.pprint(Kal_th_e_dot)
# np_matrix = np.array(n_e).astype(np.float64)
# N_cov = np.cov(np_matrix.T)
N_e = sp.Matrix([[1, -.5],[-1, 2]])

N_m = sp.Matrix([[1, -.33],[-1, .75]])
# M = sp.MatrixSymbol('M',4,4)
M_soln = Calc_M(A_f,B_f,C_f,N_m,N_e)
M=M_soln
eq = A_f*M + M* A_f.T - M*C_f.T*N_m.inv()*C_f*M + B_f*N_e*B_f.T

# soln = sp.linsolve(eq,M)
# sp.pprint(eq)
F = M*C_f.T*N_m.inv()

Kal_th_e_dot = A_f*Kal_th_e + B_f*a + F*(Kal_y - C_f*Kal_th_e)

# sp.pprint(Kal_th_e_dot)
# print(M_soln)
# Equation 20
x_est = sp.Matrix([[2*l*sp.cos(0.5*th2_e)],
                   [0.5*(th1_e + th12_e)],
                   [-l*th2_edot*sp.sin(0.5*th2_e)],
                   [0.5*(th1_edot+th12_edot)]])
# print('Equation 20')
# sp.pprint(x_est)













# NUMERICAL
# Establish Initial Conditions:

m1_num =1
m2_num =2
l_num = 1
l20_num = 1
I1_num = 0.5
I2_num = 0.5
theta1 = np.pi/4
theta2 = np.pi/4
theta12 = np.pi/2
theta12d = 0.157
theta1d = 0.157
theta2d = 0.157
# sp.pprint(J)
J = J.subs({I_1:I1_num,I_2:I2_num,l: l_num, m_1:m1_num,m_2:m2_num,theta_2: theta2,theta_12dot:theta12d,theta_1dot:theta1d})
w = w.subs({I_1:I1_num,I_2:I2_num,l: l_num, m_1:m1_num,m_2:m2_num,theta_2: theta2,theta_12dot:theta12d,theta_1dot:theta1d})
B = B.subs({I_1:I1_num,I_2:I2_num,l: l_num, m_1:m1_num,m_2:m2_num,theta_2: theta2,theta_12dot:theta12d,theta_1dot:theta1d})

sp.pprint(J)
sp.pprint(w)
sp.pprint(B)
sp.pprint(x)

# eq 10, still need v inputs
u_ctrl = u_ctrl.subs({I_1:I1_num,I_2:I2_num,l: l_num, m_1:m1_num,m_2:m2_num,theta_2: theta2,theta_12dot:theta12d,theta_1dot:theta1d,theta_2dot: theta2d})
sp.pprint(u_ctrl)

tau_m = u_ctrl[0]
tau_b = u_ctrl[1]

d = d.subs({I_1:I1_num,I_2:I2_num,l: l_num, m_1:m1_num,m_2:m2_num,theta_2: theta2,theta_12dot:theta12d,theta_1dot:theta1d,theta_2dot: theta2d})
sp.pprint(d)

# eq 14 Still needs linear acceleration vals
accel = accel.subs({I_1:I1_num,I_2:I2_num,l: l_num,l_20:l20_num, m_1:m1_num,m_2:m2_num,theta_2: theta2,theta_12dot:theta12d,theta_1dot:theta1d,theta_2dot: theta2d})


# est_th1 = 0.5
# est_th12 = 0.5
est_th1d = 0.5
est_th12d = 0.5
ax = 0
ay = 0

Kalman_y = Kal_y.subs({theta_1: theta1, theta_12:theta12})
est_th1 = Kalman_y[0]
est_th12 = Kalman_y[1]
est_th2 = est_th12 - est_th1
est_th2d = est_th12d - est_th1d

Kalman_Thdot = Kal_th_e_dot.subs({I_1:I1_num,I_2:I2_num,l: l_num,l_20:l20_num, m_1:m1_num,m_2:m2_num,theta_2: theta2,theta_12dot:theta12d,theta_1dot:theta1d,
                             theta_2dot: theta2d, theta_1:theta1, theta_12: theta12, th1_e:est_th1,th12_e:est_th12,th1_edot:est_th1d,th12_edot:est_th12d, a_x:ax,a_y:ay})

estimated_x = x_est.subs({th1_e:est_th1,th12_e:est_th12,th2_e:est_th2,l:l_num,th12_edot:est_th12d,th1_edot:est_th1d,th2_edot:est_th2d})

K_f = sp.Matrix([[10,10,10,10],
                 [5,5,5,5]])
x_d = sp.Matrix([[0],[0],[0],[0]])
v = K_f*(x_d - estimated_x)

# sp.pprint(Kalman_Thdot)