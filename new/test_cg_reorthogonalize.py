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
    m=801
    n=600
    np.random.seed(seed)
    x = cp.Variable(n)
    A = np.random.randn(m, n)
    b = np.random.randn(m)
    objective = cp.norm1(A @ x - b)
    d = np.random.randn(n, 100)
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

import scipy as sp
from cqr.cvxpy_interface import solvers
self = solvers[-1]
self.var_reduced = np.zeros(self.m-1)

print(np.linalg.norm(self.newres(self.var_reduced)))
step = np.zeros_like(self.var_reduced)

TOTAL = 0
def _counter(_):
    global TOTAL
    TOTAL += 1

for i in range(10000):
    
    res = self.newres(self.var_reduced)
    J = self.newjacobian_linop(self.var_reduced)
    print('init loss', np.linalg.norm(J @ step + res))
    
    # orthogonalization
    tmp = self.newjacobian_linop_nocones() @ step
    err = tmp - self.coneproje_linop(self.var_reduced) @ tmp
    # breakpoint()
    step -= self.newjacobian_linop_nocones().T @ err

    print('post orth loss', np.linalg.norm(J @ step + res))
    OLDTOTAL = int(TOTAL)
    result = sp.sparse.linalg.cg(
        J.T @ J,
        -J.T @ res,
        callback=_counter,
        # x0=step,
        rtol=min(0.5, np.linalg.norm(J.T @ res)**0.5),
        maxiter=100,
        # atol=0.,
        # btol=min(0.5, np.linalg.norm(res**0.5)),
        # iter_lim=None
        )
    print('ITER', i, 'CG ITERS', TOTAL-OLDTOTAL)
    # print(result[1:-1])
    # TOTAL += result[2]
    step[:] = result[0]
    for _ in range(100):
        test_new_var = self.var_reduced + step/(2.**_)
        if np.linalg.norm(self.newres(test_new_var)) < np.linalg.norm(res):
            self.var_reduced = test_new_var
            # self.var_reduced = self.var_reduced + step/(2.**(_+2))
            break
    print(np.linalg.norm(self.newres(self.var_reduced)))
    if np.linalg.norm(self.newres(self.var_reduced)) < 1e-12:
        break
print(TOTAL)