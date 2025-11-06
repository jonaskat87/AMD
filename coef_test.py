# script which checks growth in k

from sympy import *
# from sympy.combinatorics import Permutation
from sympy.abc import p,q
import numpy as np
from sympy.utilities.iterables import multiset_permutations, ordered_partitions
from sympy.functions.combinatorial.numbers import bernoulli
init_printing(use_unicode=True)

f, g = symbols('f g', cls=Function)
eta = symbols('eta', positive=True)

# define Poisson bracket (appropriately scaled by eta)
def P_bracket(eta, first, second):
    return eta * (diff(first, q) * diff(second, p) - diff(first, p) * diff(second, q))

# return the modified Hamiltonian term at order (m - 1) using recursive formula
def H_m(m, eta, F, G):
    if m == 1:
        return F + G
    else:
        func = P_bracket(eta, H_m(m - 1, eta, F, G), G - F) / 2
        for p in (np.arange((m - 1) / 2) + 1): 
            adZ2p = 0
            # sum over all integer partitions. To save memory, do not save each one generated
            for part in ordered_partitions(m - 1, int(2 * p)): 
                # loop over all permutations of this partition
                for perm in list(multiset_permutations(part)):
                    bracket = F + G
                    # compute iterated Poisson bracket
                    for i in perm:
                        bracket = P_bracket(eta, bracket, H_m(i, eta, F, G))
                    adZ2p += bracket
            func += bernoulli(int(2 * p)) * adZ2p / factorial(2 * p)
        # ensure that all rational fractions are kept and appropriate terms are combined
        return simplify(nsimplify(func / m)) 
    
# return the modified Hamiltonian term at order k without the eta ** k
def H_k(k, eta, F, G):
    return simplify(H_m(k + 1, eta, F, G) / eta ** k)

# returns operator exponential of Poisson bracket
# n is how far to take the exponential expansion inside
# func is arbitrary function
def op_exp(eta, n, F, G):
    # Define series exponential of Poisson bracket operator
    inner_operator = F
    for i in (np.arange(n) + 1):
        bracket = F
        for k in np.arange(i):
            bracket = P_bracket(eta, bracket, G)
        inner_operator +=  bracket / factorial(i) 
    return inner_operator

def inner_operation(eta, n, F, G, func):
    return eta * (-diff(G, q) * diff(func, p) + diff(op_exp(eta, n + 1, F, G), p) * diff(func, q))

# function which is then to be evaluated on H_k to generate C_kn
def nested_op(eta, n, F, G, func):
    # Define series exponential of operator in outer exponential
    series = 0
    for i in np.arange(n + 1):
        for j in np.arange(n - i + 1):
            series += (-eta * diff(G, q)) ** i * (eta * diff(op_exp(eta, n, F, G), p)) ** j * diff(diff(func, p, i), q, j) / \
                (factorial(i) * factorial(j))
    return series

# compute C_{kn} coefficient (see equation 14)
def C_kn(eta, k, n, F, G):
    return expand(diff(nested_op(eta, n + 1, F, G, H_k(k, eta, f(p), g(q))), eta, n).subs(eta, 0) / factorial(n))

# script which sums coefficients in absolute value form 
for i in range(1,13):
    print('Diagonal ', i, ':')
    sum = 0
    for k in np.arange(i - 1):
        res = simplify(C_kn(eta, k, i - k, f(p), g(q))).args
        res = [Abs(term).subs([(f, exp), (g, exp), (p, 0), (q, 0)]).doit() for term in res]
        sum += Add(*res)
    simplify(sum)
    print(sum)

    
    

