import sympy as sp
import numpy as np





# Define sympy variable:
I_1, I_2, m_1, m_2, l, theta_1, theta_2, theta_12 = sp.symbols('I1 I2 m1 m2 l theta1 theta2 theta_12')
theta_1dot, theta_12dot, theta_2dot = sp.symbols('theta1dot theta12dot theta_2dot')

# define J, w, and R_f
J = sp.Matrix([
    [I_1 + (1/4)*m_1*l**2+m_2*l**2, (1/2)*m_2*l**2*sp.cos(theta_2)],
    [(1/2)*m_2*l**2*sp.cos(theta_2), I_2 +(1/4)*m_2*l**2]

])
w = sp.Matrix([
    [(1/2)*m_2**2*l**2*sp.sin(theta_2)*theta_12dot**2],
    [-(1/2)*m_2**2*l**2*sp.sin(theta_2)*theta_1dot**2]

])
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


# State Matrices
B = sp.Matrix([[0, 0],
               [0, 0],
               [l*sp.sin(theta_2/2),-l*sp.sin(theta_2/2)],
               [1/2, 1/2]])
B = B*J.inv()

d = sp.Matrix([[0],
               [0],
               [-(1/2)*l*theta_2dot**2*sp.cos(theta_2/2)],
               [0]])
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


Kal_th,Kal_thdot, n_e, n_m = sp.symbols('Kal_th Kal_thdot n_e n_m')

Kal_th = sp.Matrix([[theta_1],[theta_12],[theta_1dot],[theta_12dot]])


# n_e definition
n_e = -K1.inv()*n

A_f = {}

B_f = []

C_f = []