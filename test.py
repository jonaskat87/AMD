import numpy as np
from scipy import linalg, integrate
from sympy import *
from sympy.abc import a, d, p, q, l, n, t
import numpy as np
from sympy import lambdify
init_printing(use_unicode=True)

A_F = Matrix([[0, 0, 2*a], [0, 0, 0], [0, 4*a, 0]])*l
A_G = Matrix([[0, 0, 0], [0, 0, -2*d], [-4*d, 0, 0]])*l
Basis = Matrix([p**2, q**2, p*q])
F = Matrix([a,0,0])
G = Matrix([0,d,0])

expA_F = A_F.exp()
expA_G = A_G.exp()
print("expA_F: ", latex(expA_F))
print("expA_G: ", latex(expA_G))

L = expA_G @ expA_F
P, D = L.diagonalize()
print("P=",latex(P))
print("D=",latex(D))
Pinv = P.inv()
print("Complicated:", latex(simplify(Pinv @ (G + expA_G @ F))))
eyeminusD = eye(3)-D
print("I-D=",latex(simplify(eyeminusD)))

# new P
P = Matrix([[-1/(2*d*l), -sin(t)*exp(j*t)/(2*d*l), sin(t)*exp(-j*t)/(2*d*l)], 
            [-1/(2*a*l), sin(t)*exp(-j*t)/(2*a*l), -sin(t)*exp(-j*t)/(2*a*l)], 
            [1, 1, 1]])
print(latex(simplify(P.inv())))