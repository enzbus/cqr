#
# Trick to get solver object
#
import numpy as np
import cvxpy as cp
from cqr.cvxpy_interface import CQR

SOC = False #True #True #False
PROGRAM_ONE = True #False# True #False
PROGRAM_TWO = False # True #True #True

if SOC:

    np.random.seed(0)
    m, n = 200, 100
    x = cp.Variable(n)
    A = np.random.randn(m, n)
    b = np.random.randn(m)
    objective = cp.norm2(A @ x - b) + 1. * cp.norm1(x)
    constraints = []
    program = cp.Problem(cp.Minimize(objective), constraints)
    #prog.solve(solver='SCS', verbose=True, eps=3e-10)
    #scs_obj = objective.value
    program.solve(solver=CQR())
if PROGRAM_ONE:
    seed=0
    m=410
    n=300
    np.random.seed(seed)
    x = cp.Variable(n)
    A = np.random.randn(m, n)
    b = np.random.randn(m)
    objective = cp.norm1(A @ x - b)
    d = np.random.randn(n, 25)
    constraints = [cp.abs(x) <= .75, x @ d == 2.,]
    program = cp.Problem(cp.Minimize(objective), constraints)
    program.solve(solver=CQR())


if PROGRAM_TWO:
    seed=0
    m=70
    n=40
    x = cp.Variable(n)
    A = np.random.randn(m, n)
    b = np.random.randn(m)
    objective = cp.norm1(A @ x - b) + 1. * cp.norm1(x)
    # adding these constraints, which are inactive at opt,
    # cause cg loop to stop early
    constraints = []  # x <= 1., x >= -1]
    program = cp.Problem(cp.Minimize(objective), constraints)
    program.solve(solver=CQR())


###
# Simple DR
###


from cqr.cvxpy_interface import solvers
self = solvers[-1]

def loss(x):
    return np.linalg.norm(self.douglas_rachford_step(x))**2/2.

def grad(x):
    return self.douglas_rachford_step_derivative(x).T @ self.douglas_rachford_step(x)

def func(x):
    tmp = self.douglas_rachford_step(x)
    return np.linalg.norm(tmp)**2/2., self.douglas_rachford_step_derivative(x).T @ tmp

from scipy import optimize as opt

losses = []

def callback(x):
    losses.append(np.linalg.norm(self.douglas_rachford_step(x)))

result = opt.fmin_l_bfgs_b(func, x0=np.zeros(self.m * 2), approx_grad=False, factr=1e-32, pgtol=1e-32, maxfun=100000000, maxiter=100000000,callback=callback)

import matplotlib.pyplot as plt
plt.semilogy(losses)
plt.show()

print(result[1])
result[2].pop('grad')
print(result[2])
self.var_reduced=self._var_reduced_from_sy(result[0])
self.new_toy_solve()
self.decide_solution_or_certificate()