#
# Trick to get solver object
#
import numpy as np
import logging
logging.basicConfig(level='INFO')
import cvxpy as cp
from cqr.cvxpy_interface import CQR

SOC = False # True #False #True #True #False
PO = True # False #True
PROGRAM_ONE = False#False#True # True #False
PROGRAM_TWO = False # True # False # True #True #True


MYMEMORY = 100

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
    n = 1000
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
# Simple DR
###


from cqr.cvxpy_interface import solvers
self = solvers[-1]


dr_y = np.zeros(2 * self.m)
losses = []
xs = []
steps = []
for i in range(120):
    xs.append(np.copy(dr_y))
    step = self.douglas_rachford_step(dr_y)
    steps.append(np.copy(step))
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
# plt.semilogy(losses)
# plt.show()

oldlosses = np.array(losses)

if False:

    oldlosses = np.array(losses)
    # self.var_reduced = var_reduced
    var_reduced = self.inexact_levemberg_marquardt(self.newres, self.newjacobian_linop, self._var_reduced_from_sy(dr_y), max_iter=100)

    dr_y = self._sy_from_var_reduced(var_reduced)
    for i in range(2000):
        xs.append(np.copy(dr_y))
        step = self.douglas_rachford_step(dr_y)
        steps.append(np.copy(step))
        losses.append(np.linalg.norm(step))
        # print(losses[-1])
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

    raise Exception
    breakpoint()

MEMORY = 0

import scipy as sp

for i in range(100000):
    xs.append(np.copy(dr_y))
    step = self.douglas_rachford_step(dr_y)
    steps.append(np.copy(step))
    losses.append(np.linalg.norm(step))
    print(i, losses[-1])
    if losses[-1] < 1e-13:
        print(f'converged in {i} iterations')
        break
    test_new_x = {}
    if i % 1000 == -1:
        dr_y = self._sy_from_var_reduced(self.inexact_levemberg_marquardt(self.newres, self.newjacobian_linop, self._var_reduced_from_sy(dr_y), max_iter=1))
        xs = []
        steps = []
        continue

    # if i % 27 == 0:
    #     dr_y = self._sy_from_var_reduced(self.inexact_levemberg_marquardt(self.refinement_residual, self.refinement_jacobian, self._var_reduced_from_sy(dr_y), max_iter=20))
    #     continue

    for MEMORY in [MYMEMORY]:#0, 1, 2, 5, 10]:#, 20, 50, 100]:
        if MEMORY == 0:
            test_new_x[MEMORY] = np.copy(xs[-1] + steps[-1])#@*0.15)
            continue

        mystep = -np.array(steps[-MEMORY-1:])
        Y = np.diff(mystep, axis=0)
        myxs = np.array(xs[-MEMORY-1:])
        S = np.diff(myxs, axis=0)
        # newstep = steps[-1] + (S.T -Y.T) @ np.linalg.lstsq(S @ Y.T,# + np.eye(9)*1e-4,
        #  S @ steps[-1], rcond=None)[0]
        # u,s,v = np.linalg.svd(Y, full_matrices=False)
        # TOPPINO = (v @ steps[-1]) / s
        # TOP1 = S.T @ (u @ (TOPPINO)) - v.T @ (v @ steps[-1])
        #TOP = (S.T -Y.T) @ np.linalg.lstsq(Y @ Y.T,Y @ steps[-1], rcond=None)[0]
        # mat = Y @ Y.T
        # diag = 0. #np.mean(np.diag(mat))
        # TOP2 = (S.T -Y.T) @ np.linalg.solve(mat + np.eye(len(mat)) * diag,Y @ steps[-1])
        Q,R = np.linalg.qr(Y.T)
        mytmp = Q.T @ steps[-1]
        TOP3 = S.T @ sp.linalg.solve_triangular(R, mytmp, lower=False) - Q @ mytmp
        # breakpoint()

        # last attempt
        _ = np.vstack([(mystep @ mystep.T), -np.ones(MEMORY+1)])
        _ = np.hstack([_, -np.ones((MEMORY+2,1))])
        _[-1,-1] = 0.
        rhs = np.zeros(MEMORY+2)
        rhs[-1] = -1.
        alphamu = np.linalg.solve(_, rhs)
        alpha = alphamu[:-1]
        zio = (myxs - mystep).T @ alpha

        # TEST WITH QR
        myq, myr = np.linalg.qr(mystep.T)
        one = np.linalg.solve(myr.T, np.ones(MEMORY+1))
        zaius = myr * np.linalg.norm(one)
        alpha = np.linalg.solve(zaius.T @ zaius, np.ones(MEMORY+1))
        newnewzio = (myxs - mystep).T @ alpha

        one /= np.linalg.norm(one) # any norming here
        two = np.linalg.solve(myr, one)
        normer = sum(two)
        # zio = (myxs - mystep).T @ two
        newzio = myxs.T @ (two / normer) - myq @ (one / normer)
        # breakpoint()
        

        # breakpoint()

        # breakpoint()
        # breakpoint()
        # mat = S @ Y.T
        # diag = np.mean(np.diag(mat))
        # TOP3 = (S.T -Y.T) @ np.linalg.solve(mat + np.eye(len(mat)) * diag,S @ steps[-1])
        # assert np.allclose(TOP3, TOP2)
        newstep = steps[-1] + TOP3
        test_new_x[MEMORY] = np.copy(xs[-1] + newstep)#@*0.15)
        
        # test_new_x[MEMORY] = zio
        test_new_x[MEMORY] = newzio
        # test_new_x[MEMORY] = newnewzio

        # breakpoint()

    test_new_losses = {MEMORY:np.linalg.norm(self.douglas_rachford_step(test_new_x[MEMORY])) for MEMORY in test_new_x}

    BEST_MEMORY = np.argmin(list(test_new_losses.values()))
    #print('BEST MEMORY', list(test_new_losses.keys())[BEST_MEMORY])
    dr_y = test_new_x[list(test_new_losses.keys())[BEST_MEMORY]]

self.var_reduced=self._var_reduced_from_sy(dr_y)
self.refine()
# self.new_toy_solve()
# self.decide_solution_or_certificate()

plt.semilogy(losses)
plt.semilogy(oldlosses)
plt.show()

# mystep = np.array(steps[-10:])
# Y = np.diff(mystep, axis=0)
# myxs = np.array(xs[-10:])
# S = np.diff(myxs, axis=0)

# newstep = steps[-1] - (S.T -Y.T) @ np.linalg.lstsq(S @ Y.T, S @ steps[-1], rcond=None)[0]
# x = np.copy(xs[-1] - newstep*0.15)

# print(np.linalg.norm(self.douglas_rachford_step(xs[-1] + steps[-1])))
# print(np.linalg.norm(self.douglas_rachford_step(xs[-1] + newstep)))


###
# LEVEMBERG-MAR
###

if False:

    import scipy as sp
    dr_y = np.zeros(2 * self.m)
    TOT = 0
    losses = []

    for i in range(100):
        result = sp.sparse.linalg.lsqr(self.douglas_rachford_step_derivative(dr_y), self.douglas_rachford_step(dr_y), atol=0, btol=0.)#001)#, iter_lim=30)
        #print(result[1:-1])
        TOT += result[2]
        leng = -np.logspace(-12,0)#np.linspace(-3,3,10000)
        vals = [np.linalg.norm(self.douglas_rachford_step(dr_y + result[0] * l))**2 for l in leng]
        print((np.min(vals), leng[np.argmin(vals)]))
        losses.append(np.sqrt(np.min(vals)))
        if losses[-1] < 1e-12:
            break
        #plt.plot(leng,vals)
        dr_y = np.copy(dr_y + result[0] * leng[np.argmin(vals)])

    #plt.show()
    print("TOTAL", TOT)
    plt.figure()
    plt.semilogy(losses)
    plt.show()

###
# Anderson
###

def f(x):
    return np.copy(x + self.douglas_rachford_step(x))

def anderson(xs, fs, memory=10):
    ...