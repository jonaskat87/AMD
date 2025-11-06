# create script which generates terms of modified Hamiltonian up to and at a certain order

from sympy import *
from sympy.combinatorics import Permutation
import numpy as np
init_printing(use_unicode=True)

# define vectors p and q of variables by index p and q up to d dimensions
def init_pq(d):
    if d == 1:
        p = symbols('p')
        q = symbols('q')
    else:
        p = symbols(''.join(['p' + str(i) + ' ' for i in np.arange(d)]))
        q = symbols(''.join(['q' + str(i) + ' ' for i in np.arange(d)]))
    return p, q

# define Poisson bracket (appropriately scaled by eta)
def P_bracket(eta, first, second, p, q):
    if isinstance(p, Symbol): # check if one-dimensional and therefore not in tuple
        return eta * (diff(first, q) * diff(second, p) - diff(first, p) * diff(second, q))
    else:
        div1 = np.array([diff(first, q[i]) for i in np.arange(len(p))])
        div2 = np.array([diff(second, p[i]) for i in np.arange(len(p))])
        div3 = np.array([diff(first, p[i]) for i in np.arange(len(p))])
        div4 = np.array([diff(second, q[i]) for i in np.arange(len(p))])
        return eta * (np.dot(div1, div2) - np.dot(div3, div4))
# p, q = init_pq(2)
# print(P_bracket(0.01, sin(q[0])+cos(q[1]), sin(p[0])*cos(p[1]), p, q))
# f, g = symbols('f g', cls=Function)
# eta = symbols('eta')
# p, q = init_pq(1)
# p = (p,)
# q = (q,)
# print(latex(P_bracket(eta,f(*q)*g(*p),P_bracket(eta,P_bracket(eta, f(*q), g(*p),p,q),g(*p),p,q),p,q)))

# simplified Reinsch algorithm to generate BCH series from Van–Brunt and Visser (2016)
# first and second should take in separate symbol arguments, NOT a tuple/list/etc.
def Reinsch(N, eta, first, second, p, q):
    x = symbols(''.join(['x' + str(i) + ' ' for i in np.arange(N + 1)]))
    y = symbols(''.join(['y' + str(i) + ' ' for i in np.arange(N + 1)]))
    perm = Permutation(*np.roll(np.arange(N + 2),1))
    X = MatrixPermute(diag(0, diag(*x)), perm, axis = 0).as_explicit()
    Y = MatrixPermute(diag(0, diag(*y)), perm, axis = 0).as_explicit()
    EX = X.exp()
    EY = Y.exp()
    EXY = simplify(EX @ EY)
    # shows terms to be added
    Z = Add.make_args(sum(EXY.log().row(0)))
    BCH = 0 # full BCH series
    for expr in Z: # loop through expressions in the sum
        length = expr.args.__len__()
        if length == 0: # if only one term
            if str(expr)[0] == 'x':
                BCH += first
            else:
                BCH += second
        else: # if not the terms with only x0 or only y0
            length -= 1 # don't count coefficient in length
            expr = list(Mul.make_args(expr)) # define list with all expressions in it
            const = expr[0]
            terms = expr[1:]
            terms.sort(key=lambda x: int(str(x)[1]), reverse=True) # sort terms by the subscript (second character in symbol)
            # evaluate sorted terms
            acc = 'flag' # the result of recursively applying Poisson bracket in order as indicated by the ordering of the symbols
            for term in terms:
                if acc == 'flag': # to initialize acc
                    if str(term)[0] == 'x':
                        acc = const * first
                    else:
                        acc = const * second
                else:
                    if str(term)[0] == 'x':
                        acc = P_bracket(eta, first, acc, p, q)
                    else:
                        acc = P_bracket(eta, second, acc, p, q)
            BCH += acc / length
    return BCH
p, q = init_pq(1)
f, g = symbols('f g', cls=Function)
eta = symbols('eta')
# print(latex(Reinsch(4, eta, sin(q), cos(p), p, q)))

# get Nth-order term from BCH series
def BCH_Nth(N, eta, first, second, p, q):
    series = Reinsch(N, eta, first, second, p, q)
    return LC(Poly(series, eta)) # return leading coefficient when interpreted as polynomial in eta
print(latex(BCH_Nth(3, eta, f(*p), g(*q), p, q)))

def H_k(k, eta):
    if k == 0:
        H = F + G
    elif k == 1:
        H = P_bracket(eta, F, G) / 2
    elif k == 2:
        H = P_bracket(eta, P_bracket(eta, F, G), G) / 12 + P_bracket(eta, P_bracket(eta, G, F), F) / 12
    elif k == 3:
        H = -P_bracket(eta, P_bracket(eta, P_bracket(eta, F, G), G), F) / 24
    elif k == 4:
        H = -P_bracket(eta,P_bracket(eta, P_bracket(eta, P_bracket(eta, G, F), F), F), F) / 720 - \
        P_bracket(eta,P_bracket(eta, P_bracket(eta, P_bracket(eta, F, G), G), G), G) / 720 + \
        P_bracket(eta,P_bracket(eta, P_bracket(eta, P_bracket(eta, G, F), F), F), G) / 360 + \
        P_bracket(eta,P_bracket(eta, P_bracket(eta, P_bracket(eta, F, G), G), G), F) / 360 + \
        P_bracket(eta,P_bracket(eta, P_bracket(eta, P_bracket(eta, F, G), F), G), F) / 120 + \
        P_bracket(eta,P_bracket(eta, P_bracket(eta, P_bracket(eta, G, F), G), F), G) / 120 
    elif k == 5:
        H = P_bracket(eta, P_bracket(eta,P_bracket(eta, P_bracket(eta, P_bracket(eta, F, G), F), G), F), G) / 240 + \
            P_bracket(eta, P_bracket(eta,P_bracket(eta, P_bracket(eta, P_bracket(eta, F, G), G), G), F), G) / 720 - \
            P_bracket(eta, P_bracket(eta,P_bracket(eta, P_bracket(eta, P_bracket(eta, F, G), F), F), G), G) / 720 + \
            P_bracket(eta, P_bracket(eta,P_bracket(eta, P_bracket(eta, P_bracket(eta, F, G), F), F), F), G) / 1440 - \
            P_bracket(eta, P_bracket(eta,P_bracket(eta, P_bracket(eta, P_bracket(eta, F, G), G), F), G), G) / 720
    return simplify(H)

    
    
    

