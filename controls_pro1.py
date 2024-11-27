import sympy as sp
import numpy as np





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





print('n_e def')
sp.pprint(n_e)

A_f = sp.Matrix([[0,0,1,0],
                 [0,0,0,1],
                 [0,0,0,0],
                 [0,0,0,0]])

B_f = sp.Matrix([[0],[0],[1],[1]])

C_f = sp.Matrix([[1, 1, 0, 0]])

# Equation 15
Kal_thdot = sp.Matrix([[0,0,1,0],
                       [0,0,0,1],
                       [0,0,0,0],
                       [0,0,0,0]]) * Kal_th + sp.Matrix([[0,0],
                                                         [0,0],
                                                         [0,1],
                                                         [1,0]])*(accel+n_e)
sp.pprint(Kal_thdot)

n_m = sp.Matrix([[1],[1]])

# Equation 16
Kal_y = sp.Matrix([[1, 0, 0, 0],
                   [0, 1, 0, 0]])*Kal_th + n_m

th12_e, th1_e, th2_e = sp.symbols('th12e th1e th2e')
th12_edot, th1_edot, th2_edot = sp.symbols('th12edot th1edot th2edot')
x_est = sp.Matrix([[2*l*sp.cos(0.5*th2_e)],
                   [0.5*(th1_e + th12_e)],
                   [-l*th2_edot*sp.sin(0.5*th2_e)],
                   [0.5*(th1_edot+th12_edot)]])
# print('Equation 20')
# sp.pprint(x_est)













# NUMERICAL
# Establish Initial Conditions:

# m1_num =1
# m2_num =2
# l_num = 1
# I1_num = 0.5
# I2_num = 0.5
# # sp.pprint(J)
# J = J.subs({I_1:I1_num,I_2:I2_num,l: l_num, m_1:m1_num,m_2:m2_num})
# w = w.subs({I_1:I1_num,I_2:I2_num,l: l_num, m_1:m1_num,m_2:m2_num})
# B = B.subs({I_1:I1_num,I_2:I2_num,l: l_num, m_1:m1_num,m_2:m2_num})

# # sp.pprint(J)
# # sp.pprint(w)
# # sp.pprint(B)
# # sp.pprint(x)

# u_ctrl = u_ctrl.subs({I_1:I1_num,I_2:I2_num,l: l_num, m_1:m1_num,m_2:m2_num})
# # sp.pprint(u_ctrl)

# tau_m = u_ctrl[0]
# tau_b = u_ctrl[1]

# # tau_m = tau_m.subs({theta_2:0.5})
# # sp.pprint(tau_m)
