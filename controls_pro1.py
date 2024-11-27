import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

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
r_st, theta_r, rdot, theta_rdot = sp.symbols('r theta_r rdot theta_rdot')
x,u,d,n = sp.symbols('x u d n')
tau_m, tau_b = sp.symbols('tau_m tau_b')
x = sp.Matrix([[r_st],[theta_r],[rdot],[theta_rdot]])
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

# Rquation 5
x_d = A*x + B*(u+w) + d
# print(555)
# sp.pprint(x_d)



v,v_1,v_2 = sp.symbols('v v_1 v_2')
v = sp.Matrix([[v_1],[v_2]])
A_r = sp.Matrix([[0,0,1,0],
                [0,0,0,1],
                [0,0,0,0],
                [0,0,0,0]])
B_r = sp.Matrix([[0, 0],
               [0, 0],
               [0,1],
               [1, 0]])
C_r = sp.Matrix([[0,1,0,0],
                [1,0,0,0]])
x_r = A_r*x+B_r*v
y_r = C_r*x


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




# print('n_e def')
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
# F = sp.Matrix([[1,0],
#                [0,1],
#                [1,0],
#                [0,1]])
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

sp.pprint(Kal_th_e_dot)
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

# sp.pprint(J)
# sp.pprint(w)
# sp.pprint(B)
# sp.pprint(x)
x_d = x_d.subs({I_1:I1_num,I_2:I2_num,l: l_num, m_1:m1_num,m_2:m2_num,theta_2: theta2,theta_12dot:theta12d,theta_1dot:theta1d,theta_2dot: theta2d})
# print(666)
# sp.pprint(x_d)
# eq 10, still need v inputs
u_ctrl = u_ctrl.subs({I_1:I1_num,I_2:I2_num,l: l_num, m_1:m1_num,m_2:m2_num,theta_2: theta2,theta_12dot:theta12d,theta_1dot:theta1d,theta_2dot: theta2d})
# sp.pprint(u_ctrl)

tau_m = u_ctrl[0]
tau_b = u_ctrl[1]

d = d.subs({I_1:I1_num,I_2:I2_num,l: l_num, m_1:m1_num,m_2:m2_num,theta_2: theta2,theta_12dot:theta12d,theta_1dot:theta1d,theta_2dot: theta2d})
# sp.pprint(d)

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

K_f = sp.Matrix([[1,1,1,1],
                 [.5,.5,.5,.5]])
x_des = sp.Matrix([[0],[0],[0],[0]])
v = K_f*(x_des - estimated_x)

# sp.pprint(Kalman_Thdot)









# *****************************************************
# Initial COnditions:
# Initial Thetas
# theta1 = np.pi/4
# theta2 = np.pi/4
# theta12 = np.pi/2
# r = 0.75
phi = np.pi/8
rd = 0
phid = 0
# theta2 = np.arccos((2*l_num**2-r**2)/(2*l_num*l_num))
# theta1 = phi - np.arccos((r**2)/(2*l_num*r))
# theta12 = theta1 + theta2

theta12d = 0.157
theta1d = 0.157
theta2d = 0.157

taum = 0.1
taub = 0.1

# B = B.subs({I_1:I1_num,
#             I_2:I2_num,
#             l: l_num, 
#             m_1:m1_num,
#             m_2:m2_num,
#             theta_2:theta2,
#             theta_12dot:theta12d,
#             theta_1dot:theta1d})
# u=u.subs({tau_m:5})
x_d = x_d.subs({I_1:I1_num,
                I_2:I2_num,
                l: l_num, 
                m_1:m1_num,
                m_2:m2_num,
                theta_2: theta2,
                theta_12dot:theta12d,
                theta_1dot:theta1d,
                theta_2dot: theta2d,
                rdot:rd,
                theta_rdot:phid,
                
                })
sp.pprint(tau_m)

State_dot = x_d
rd = State_dot[0]
phid = State_dot[1]
rdd = State_dot[2]
phidd = State_dot[3]
# sp.pprint(x_d)

u_ctrl = u_ctrl.subs({I_1:I1_num,
                      I_2:I2_num,
                      l:l_num, 
                      m_1:m1_num,
                      m_2:m2_num,
                      theta_2:theta2,
                      theta_12dot:theta12d,
                      theta_1dot:theta1d,
                      theta_2dot:theta2d})
r_num=1
phi= np.pi/4
rd = 0.1
phid=0
ax = 0
ay = 0
phi_rec = []
r_rec = []
ref_rec = []
xhat_rec = []
x_des = sp.Matrix([[0.34],[0.5],[0.2],[0.4]])
estth1=0
estth12=0
estth2=0
estth12dot=0
estth1dot=0
estth2dot=0
# sp.pprint(x_est)
for t in np.arange(0,5,0.1):
    f = 2
    ref_r = -0.2+0.015*np.sin(2*np.pi*f*t)
    x_des = sp.Matrix([[ref_r],[0],[0],[0]])
    # sp.pprint(x_des)
    ref_rec.append(ref_r)
    
    
    theta2 = np.arccos((2*l_num**2-r_num**2)/(2*l_num*l_num))
    theta1 = phi - np.arccos((r_num**2)/(2*l_num*r_num))
    theta12 = theta1 + theta2
    
    # sp.pprint(x.subs({r_st:r_num,theta_r:phi,theta_rdot:phid,rdot:rd}))
    
    x_cur = x.subs({r_st:r_num,theta_r:phi,theta_rdot:phid,rdot:rd})
    # # r_num=1
    # # phi= np.pi/4
    # rd = x[0]
    # phid=x[1]
    # y_r
    
    
    Acceleration = accel.subs({a_x:ax,a_y:ay})
    Th1Accel = Acceleration[0]
    Th12Accel = Acceleration[1]
    
    Est_Kal = Kal_th_e_dot.subs({I_1:I1_num,
                                 I_2:I2_num,
                                 l: l_num,
                                 l_20:l20_num, 
                                 m_1:m1_num,
                                 m_2:m2_num,
                                 theta_2: theta2,
                                 theta_12dot:theta12d,
                                 theta_1dot:theta1d,
                                    theta_2dot: theta2d, 
                                    theta_1:theta1,
                                    theta_12: theta12,
                                    th1_e:estth1,
                                    th12_e:estth12,
                                    th1_edot:estth1dot,
                                    th12_edot:estth12dot,
                                    a_x:ax,
                                    a_y:ay})
    # sp.pprint(Est_Kal)
    KalmanY = Kal_y.subs({
                            theta_12dot:theta12d,
                            theta_1dot:theta1d,
                            theta_1:theta1,
                            theta_12: theta12,})
    # sp.pprint(Kalman_y)
    estth1=Kalman_y[0]
    estth12=Kalman_y[1]
    estth2=estth12-estth1
    estth12dot=Est_Kal[2]
    estth1dot=Est_Kal[3]
    estth2dot=estth12dot-estth1dot
    
    xhat= x_est.subs({th1_e:estth1,th12_e:estth12,th2_e:estth2,l:l_num,th12_edot:estth12dot,th1_edot:estth1dot,th2_edot:estth2dot})
    # sp.pprint(xhat)
    xhat_rec.append(xhat[0])
    v_eval = K_f*(x_des - xhat)
    
    #  V Vectpr Created
    # sp.pprint(v)
    new_xdot = x_r.subs({rdot:rd,theta_rdot: phid,v_1 :v_eval[0],v_2 :v_eval[1]}) 
    # sp.pprint(new_xdot)
    new_x = x_cur + new_xdot*0.1
    # sp.pprint(new_x)
    
    r_num = float(new_x[0])
    # print(5)
    # sp.pprint(type(r_num))
    phi = float(new_x[1])
    rd = float(new_x[2])
    phid = float(new_x[3])
    
    
    
    # sp.pprint(x_r)
    # sp.pprint(y_r)
    
    # tau_m = tau_m.subs({v_1 :v_eval[0],v_2 :v_eval[1]})
    # sp.pprint(tau_m)
    # tau_b = tau_b.subs({v_1 :v_eval[0],v_2 :v_eval[1]})
    # sp.pprint(tau_b)
    sp.pprint(f"t: {t}, r_num: {r_num}, phi: {phi}, rd: {rd}, phid: {phid}")
    sp.pprint(f"estth1: {estth1}, estth12: {estth12}, estth2: {estth2}")
    sp.pprint(f"xhat: {xhat}, v_eval: {v_eval}")
    
    
    # Find v
    phi_rec.append(phi)
    r_rec.append(r_num)
    
    # sp.pprint(xhat)
    if t> 0.5:
        break
    
# plt.plot(r_rec)
# plt.plot(ref_rec)
# plt.plot(xhat_rec)
# plt.show()
    
plt.plot(ref_rec, label="Reference")
plt.plot(xhat_rec, label="Estimate")
plt.legend()
plt.show()
    