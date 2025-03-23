#
# Trick to get solver object
#
import numpy as np
import logging
logging.basicConfig(level='INFO')
import cvxpy as cp
from cqr.cvxpy_interface import CQR

SOC = False # True #False #True #True #False
PO = False #True
PROGRAM_ONE = False#False#True # True #False
PROGRAM_TWO = True #True #True


MYMEMORY = 10

if SOC:
    np.random.seed(0)
    m, n = 2000, 1000
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
    n = 100
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
        cp.norm1(w)<=1, cp.sum_squares((np.diag(np.sqrt(eival)) @ eivec.T) @ w) <= 0.0001]
    program = cp.Problem(cp.Minimize(objective), constraints)
    #prog.solve(solver='SCS', verbose=True, eps=3e-10)
    #scs_obj = objective.value
    program.solve(solver=CQR())

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
# Simple DR
###


from cqr.cvxpy_interface import solvers
self = solvers[-1]


dr_y = np.zeros(2 * self.m)
losses = []
for i in range(10000):
    step = self.douglas_rachford_step(dr_y)
    losses.append(np.linalg.norm(step))
    print(losses[-1])
    if losses[-1] < 1e-10:
        print(f'converged in {i} iterations')
        break
    dr_y += step
else:
    pass #raise Exception

var_reduced = self._var_reduced_from_sy(dr_y)
print('SQNORM RESIDUAL OF SOLUTION',
    np.linalg.norm(self.newres(var_reduced))**2)


import matplotlib.pyplot as plt
plt.semilogy(losses)
plt.show()



np.random.seed(0)
N = 5
x = np.random.randn(N, self.m * 2)
z = np.empty_like(x)
for i in range(N):
    z[i] = self.admm_cone_project(x[i])
y = np.empty_like(x)
for i in range(N):
    y[i] = self.admm_linspace_project(x[i])

for j in range(10000):

    import cvxpy as cp
    alpha = cp.Variable(len(z))
    beta = cp.Variable(len(y))
    obj = cp.Minimize(cp.sum_squares(z.T @ alpha - y.T @ beta))
    constr = [alpha >= 0., cp.sum(beta) == 1.]
    print('objective', cp.Problem(obj, constr).solve())
    print('alpha', alpha.value)
    print('beta', beta.value)

    in_cone = z.T @ alpha.value
    # assert np.allclose(test0, self.admm_cone_project(test0))
    in_linspace = y.T @ beta.value
    # test1 = y.T @ beta.value
    # assert np.allclose(test1, self.admm_linspace_project(test1))

    INDEX = j % N
    # x[INDEX] = (z.T @ alpha.value + y.T @ beta.value)/2.
    z[INDEX] = self.admm_cone_project(3 * in_linspace -2 * in_cone)
    y[INDEX] = self.admm_linspace_project(3* in_cone - 2*in_linspace)