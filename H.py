# create script which generates terms of modified Hamiltonian up to and at a certain order

from sympy import *
from sympy.abc import p, q, F, G # the two coordinates and the two functions F=F(p) and G=G(q)
from copy import copy
import numpy as np
from re import findall, finditer
from itertools import chain, combinations
init_printing(use_unicode=True)

# define derivatives in terms of indices as subscripts _{1_..._n} (like Einstein index notation)
# var is the variable to take derivatives of (to be inputed as a variable, not string). 
# Must be p (in which cases derivative of F terms) or q ("" G "")
# N is the numbered index through which you want to take the gradient (should be greater than those currently present)
def index_deriv(fun, var, N):
    # decompose fun into terms to be multiplied (inner) then added (outer)
    decomp = [[str(mult_term) for mult_term in Mul.make_args(add_term)] for add_term in Add.make_args(fun)]
    res = []
    for i, term in enumerate(decomp): # replicate term to take derivatives
        # list terms which require derivatives with respect to variable
        if var == p and len(list(filter(lambda x: "F" in x, term))) == 0: # if no F terms, then derivative is zero
           continue
        elif var == q and len(list(filter(lambda x: "G" in x, term))) == 0: # if no G terms, then derivative is zero
           continue
        for i in np.arange(len(term)):
            tmp = copy(term)
            if (var == p and ("F" in term[i])) or (var == q and ("G" in term[i])):
                if (var == p and ("F" in term[i]) and len(term[i])==1): # if only F
                    tmp[i] = tmp[i].replace("F", "F_{"+str(N)+"}")
                elif (var == q and ("G" in term[i]) and len(term[i])==1): # if only G
                    tmp[i] = tmp[i].replace("G", "G_{"+str(N)+"}")
                else:
                    tmp[i] = tmp[i].replace("}", "_"+str(N)+"}")
                res += [tmp]
    if len(res) == 0: # if whole expression vanishes
        return 0
    else:
        return sum([prod([sympify(Symbol(l)) for l in L]) for L in res])

def powerset(iterable):
    "list(powerset([1,2,3])) --> [(), (1,), (2,), (3,), (1,2), (1,3), (2,3), (1,2,3)]"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

# creates symbols for F and G up to order N
def create_symbols(N):
    tmp = list(powerset(np.arange(N)+1))[1:]
    inds = []
    for L in tmp:
        L = [str(n) for n in L]
        inds += ['F_{' + '_'.join(list(L)) + '}']
        inds += ['G_{' + '_'.join(list(L)) + '}']
    return ' '.join(inds)
F1, G1, F2, G2, F3, G3, F12, G12, F13, G13, F23, G23, F123, G123 = symbols(create_symbols(3))
F, G = symbols("F G")
# print(index_deriv(F+F1*G12*F2+F12*G1*G2, p, 3))
# print(index_deriv(F+F1*G12*F2+F12*G1*G2, q, 3))

# define Poisson bracket (appropriately scaled by eta)
# derivatives are marked by adding indices to the bottom (like Einstein index notation)
# N is the index to multiply by
def P_bracket(eta, first, second):
    search = findall(r'{\d+}', str(first)+str(second))
    # get index to add for dot products
    if len(search) == 0: # if no derivatives yet
        N = 1
    else:
        N = np.max([int(n.replace('{', '').replace('}', '')) for n in search]) + 1
    return eta * (index_deriv(first, q, N) * index_deriv(second, p, N) - index_deriv(first, p, N) * index_deriv(second, q, N))
eta = symbols("eta", positive=True)
# print(P_bracket(eta, F+F1*G12*F2+F12*G1*G2, G+G1*G123*F2*G3))
# print(latex(P_bracket(eta, G, F)))
# print(latex(P_bracket(eta, P_bracket(eta, G, F), F)))
# print(latex(P_bracket(eta, P_bracket(eta, P_bracket(eta, G, F), F), G)))

# def Reinsch(N, eta, first, second):
#     x = symbols(''.join(['x' + str(i) + ' ' for i in np.arange(N + 1)]))
#     y = symbols(''.join(['y' + str(i) + ' ' for i in np.arange(N + 1)]))
#     perm = Permutation(*np.roll(np.arange(N + 2),1))
#     X = MatrixPermute(diag(0, diag(*x)), perm, axis = 0).as_explicit()
#     Y = MatrixPermute(diag(0, diag(*y)), perm, axis = 0).as_explicit()
#     EX = X.exp()
#     EY = Y.exp()
#     EXY = simplify(EX @ EY)
#     # shows terms to be added
#     Z = Add.make_args(sum(EXY.log().row(0)))
#     BCH = 0 # full BCH series
#     for expr in Z: # loop through expressions in the sum
#         length = expr.args.__len__()
#         if length == 0: # if only one term
#             if str(expr)[0] == 'x':
#                 BCH += first
#             else:
#                 BCH += second
#         else: # if not the terms with only x0 or only y0
#             length -= 1 # don't count coefficient in length
#             expr = list(Mul.make_args(expr)) # define list with all expressions in it
#             const = expr[0]
#             terms = expr[1:]
#             terms.sort(key=lambda x: int(str(x)[1]), reverse=True) # sort terms by the subscript (second character in symbol)
#             # evaluate sorted terms
#             acc = 'flag' # the result of recursively applying Poisson bracket in order as indicated by the ordering of the symbols
#             for term in terms:
#                 if acc == 'flag': # to initialize acc
#                     if str(term)[0] == 'x':
#                         acc = const * first
#                     else:
#                         acc = const * second
#                 else:
#                     if str(term)[0] == 'x':
#                         acc = P_bracket(eta, first, acc)
#                     else:
#                         acc = P_bracket(eta, second, acc)
#             BCH += acc / length
#     return simplify(BCH)
# print(latex(Reinsch(3, eta, F, G)))

# Direct computation of the modified Hamiltonian at order k
# (Faster than the Reinsch algorithm)
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
        
# return inner operator (operator to power of i in equation 12)
# n is how far to take the exponential expansion inside
# func is arbitrary function
def op_deriv(eta, n, func):
    # Define series exponential of inner derivative operator
    inner_operator = F
    for i in (np.arange(n) + 1):
        bracket = F
        for k in np.arange(i):
            bracket = P_bracket(eta, bracket, G)
        inner_operator += (eta ** i) * bracket / factorial(i) 
    search = findall(r'{\d+}', str(inner_operator)+str(func))
    # get index to add for dot products
    if len(search) == 0: # if no derivatives yet
        N = 1
    else:
        N = np.max([int(n.replace('{', '').replace('}', '')) for n in search]) + 1
    return -index_deriv(G, q, N) * index_deriv(func, p, N) + index_deriv(inner_operator, p, N) * index_deriv(func, q, N)

# compute C_{kn} coefficient (see equation 14)
def C_kn(eta, k, n):
    # Define series exponential of inner derivative operator
    terms = 0
    for i in (np.arange(n) + 1):
        stuff = H_k(k, eta)
        for k in np.arange(i):
            stuff = op_deriv(eta, n, stuff)
        terms += (eta ** i) * stuff / factorial(i) 
    terms = list(Add.make_args(terms.expand(basic=True)))
    for k, expr in enumerate(terms):
        exp = 0
        # loop through all instances of eta
        flag = 1
        for i in [m.start() for m in finditer('eta', str(expr))]:
            if flag:
                temp = str(expr)[:(i-1)]
                flag = 0
            try:
                if str(expr)[(i+3):(i+5)] == '**': # if exponent
                    exp += int(str(expr[i+5]))
                elif str(expr)[i+3] == '*': # if exponent is not written out (1)
                    exp += 1
            except: # if out of range, then just eta
                exp += 1
                continue
        if exp == n:
            terms[k] = sympify(Symbol(temp))
        else:
            terms[k] = 0
    return sum(terms)
eta = symbols("eta", positive=True)
print(C_kn(eta, 0, 2))
print(C_kn(eta, 1, 1))
