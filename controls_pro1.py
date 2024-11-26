import sympy as sp
import numpy as np





# Define sympy variable:
I_1, I_2, m_1, m_2, l, theta_1, theta_2 = sp.symbols('I1 I2 m1 m2 l theta1 theta2')

J = sp.Matrix([
    [I_1 + (1/4)*m_1*l**2+m_2*l**2, (1/2)*m_2*l**2*sp.cos(theta_2)],
    [(1/2)*m_2*l**2*sp.cos(theta_2), I_2 +(1/4)*m_2*l**2]

])
sp.pprint(J)