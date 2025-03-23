#
# Trick to get solver object
#
import numpy as np
import logging
logging.basicConfig(level='INFO')
import cvxpy as cp
from cqr.cvxpy_interface import CQR

SOC = True # True #False #True #True #False
PO = True # False #True
PROGRAM_ONE = False#False#True # True #False
PROGRAM_TWO = False # False # True #True #True


MYMEMORY = 10

if SOC:
    np.random.seed(0)
    m, n = 70, 40
    x = cp.Variable(n)
    A = np.random.randn(m, n)
    b = np.random.randn(m)
    objective = cp.norm2(A @ x - b) + 1. * cp.norm1(x)
    constraints = []
    program = cp.Problem(cp.Minimize(objective), constraints)
    #prog.solve(solver='SCS', verbose=True, eps=3e-10)
    #scs_obj = objective.value
    program.solve(solver=CQR())

if PO:
    np.random.seed(0)
    n = 50
    w = cp.Variable(n)
    w0 = np.random.randn(n)
    # w0 = np.maximum(w0,0.)
    w0 -= np.sum(w0)/len(w0)
    w0 /= np.sum(np.abs(w0))
    mu = np.random.randn(n) * 1e-3
    Sigma = np.random.randn(n,n)
    Sigma = Sigma.T @ Sigma
    eival, eivec = np.linalg.eigh(Sigma)
    eival *= 1e-4
    eival[:-len(eival)//10] = 0.
    Sigma = eivec @ np.diag(eival) @ eivec.T
    # chol = np.linalg.cholesky(Sigma)
    # objective = cp.quad_form(w, Sigma) + w.T @ mu + 1e-4 * cp.norm1(w-w0)
    objective = w.T @ mu + 1e-5 * cp.norm1(w-w0)
    # w_max = np.random.randn(n)
    # w_max = np.maximum(w_max,0.01)
    # w_max = np.minimum(w_max,0.05)
    constraints = [#w >=0, #w<=w_max,
     cp.sum(w)==0, cp.norm1(w-w0)<=0.05, 
        cp.norm1(w)<=1, cp.sum_squares((np.diag(np.sqrt(eival)) @ eivec.T) @ w) <= 0.00005]
    program = cp.Problem(cp.Minimize(objective), constraints)
    #prog.solve(solver='SCS', verbose=True, eps=3e-10)
    #scs_obj = objective.value
    program.solve(solver=CQR())
    # breakpoint()

if PROGRAM_ONE:
    seed=0
    m=401
    n=300
    np.random.seed(seed)
    x = cp.Variable(n)
    A = np.random.randn(m, n)
    b = np.random.randn(m)
    objective = cp.norm1(A @ x - b)
    d = np.random.randn(n, 50)
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
# Using LM
###
from cqr.cvxpy_interface import solvers
self = solvers[-1]

zio = self._sy_from_var_reduced(np.zeros(self.m-1))
print(zio)
zia = zio[np.abs(zio) > 2.2e-16]
print(np.max(np.abs(zia))/np.min(np.abs(zia)))

import matplotlib.pyplot as plt
plt.plot(zio); plt.show()



def _densify(linear_operator):
    """Create Numpy 2-d array from a sparse LinearOperator."""
    result = np.zeros(linear_operator.shape, dtype=float)
    for j in range(result.shape[1]):
        ej = np.zeros(result.shape[1])
        ej[j] = 1.
        result[:, j] = linear_operator @ ej
    return result



def J(x):
    return _densify(self.douglas_rachford_step_derivative(x))
    # if len(x.shape)>1:
    #     x = x.flatten()
    # breakpoint()
    # return self.newjacobian_linop(x)
import scipy.optimize as opt

result = opt.least_squares(fun=self.douglas_rachford_step, jac=J, x0=np.zeros(2*self.m), method='lm', verbose=2,ftol=1e-15,xtol=1e-15,gtol=1e-15)
# takes 3000 iters on PO

print(np.linalg.norm(self.douglas_rachford_step(result.x)))


raise Exception

from cqr.cvxpy_interface import solvers
self = solvers[-1]

def _densify(linear_operator):
    """Create Numpy 2-d array from a sparse LinearOperator."""
    result = np.zeros(linear_operator.shape, dtype=float)
    for j in range(result.shape[1]):
        ej = np.zeros(result.shape[1])
        ej[j] = 1.
        result[:, j] = linear_operator @ ej
    return result


def J(x):
    return _densify(self.newjacobian_linop(x))
    # if len(x.shape)>1:
    #     x = x.flatten()
    # breakpoint()
    # return self.newjacobian_linop(x)



import scipy.optimize as opt

# result = opt.least_squares(fun=self.newres, jac=J, x0=np.zeros(self.m-1), method='trf', verbose=2)
# takes 3000 iters on PO

###
# USING LBFGS
###

def func(x):
    res = self.newres(x)
    return 2 * np.sum(res**2), self.newjacobian_linop(x).T @ res

result = opt.fmin_l_bfgs_b(
    func=func,
    x0=np.zeros(self.m-1),
    approx_grad=False,
    m=100,
    pgtol=0.,
    factr=0.,
    maxls=100,
)
print()
print()
print('NORM RES', np.sqrt(result[1]))
result[2].pop('grad')
print('STATS', result[2])

import matplotlib.pyplot as plt
# plt.plot(self.newres(result[0])); plt.show()

x = result[0]
res = self.newres(x)
print(np.linalg.norm(res))
jac = J(x)
u, s, v = np.linalg.svd(jac, full_matrices=False)
_ = u.t @ res
_ /= s
_[s<1e-12] = 0.
step = v @ _

